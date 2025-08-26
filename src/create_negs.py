import torch
import itertools
import os
import random
import clip
from tqdm import tqdm

from class_names import CLASS_NAME, preset_noun_prompt_templates, \
    preset_adj_prompt_templates, preset_noun_prompt_templates_for_sketch, csp_templates


def create_initial_negative_labels(clip_model, train_dataset='imagenet', neg_top_p=0.15, seed=0, device='cuda:0', emb_batchsize=1000, pencentile=1, 
                                   wordnet_database='txtfiles/', txt_exclude='noun.person.txt,noun.quantity.txt,noun.group.txt,adj.pert.txt'):
    
    class_name = CLASS_NAME[train_dataset]

    if train_dataset == 'imagenet_sketch':
        noun_prompt_templates = preset_noun_prompt_templates_for_sketch
    else:
        noun_prompt_templates = preset_noun_prompt_templates
        
    id_prompts = [pair[0].format(pair[1]) for pair in list(itertools.product(noun_prompt_templates, class_name))]
    text_inputs_pos = torch.cat([clip.tokenize(f"{c}") for c in id_prompts]).to(device)
    with torch.no_grad():
        text_features_pos = clip_model.encode_text(text_inputs_pos).to(torch.float32)
        feat_dim = text_features_pos.shape[-1]
        text_features_pos = text_features_pos.view(-1, len(class_name), feat_dim)
        text_features_pos = text_features_pos.mean(dim=0)
        text_features_pos /= text_features_pos.norm(dim=-1, keepdim=True)

    
    txtfiles = os.listdir(wordnet_database)
    if txt_exclude:
        file_names = txt_exclude.split(',')
        for file in file_names:
            txtfiles.remove(file)
    words_noun = []
    words_adj = []

    raw_words_noun = []
    raw_words_adj = []

    prompt_templete = dict(
        adj=csp_templates,
        noun=noun_prompt_templates,
    )

    dedup = dict()
    random.seed(seed)
    noun_length = 0
    adj_length = 0
    for file in txtfiles:
        filetype = file.split('.')[0]
        if filetype not in prompt_templete:
            continue
        with open(os.path.join(wordnet_database, file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('_', ' ')
                if line.strip() in dedup:
                    continue
                dedup[line.strip()] = None
                if filetype == 'noun':
                    if line.strip() in class_name:
                        continue
                    noun_length += 1
                    raw_words_noun.append(line.strip())
                    for template in prompt_templete[filetype]:
                        words_noun.append(template.format(line.strip()))
                elif filetype == 'adj':
                    adj_length += 1
                    candidate = random.choice(prompt_templete[filetype]).format(line.strip())
                    raw_words_adj.append(candidate)
                    for template in preset_adj_prompt_templates:
                        words_adj.append(template.format(candidate))
                else:
                    raise TypeError


    text_inputs_neg_noun = torch.cat([clip.tokenize(f"{c}") for c in tqdm(words_noun)]).to(device)
    text_inputs_neg_adj = torch.cat([clip.tokenize(f"{c}") for c in tqdm(words_adj)]).to(device)
    text_inputs_neg = torch.cat([text_inputs_neg_noun, text_inputs_neg_adj], dim=0)
    ensemble_noun_length = len(text_inputs_neg_noun)

    with torch.no_grad():
        text_features_neg = []
        for i in tqdm(range(0, len(text_inputs_neg), emb_batchsize)):
            x = clip_model.encode_text(text_inputs_neg[i: i + emb_batchsize])
            text_features_neg.append(x)
        text_features_neg = torch.cat(text_features_neg, dim=0)

        noun_text_features_neg = text_features_neg[:ensemble_noun_length].view(-1, len(noun_prompt_templates), feat_dim).mean(dim=1)
        adj_text_features_neg = text_features_neg[ensemble_noun_length:].view(-1, len(preset_adj_prompt_templates), feat_dim).mean(dim=1)
        text_features_neg = torch.cat([noun_text_features_neg, adj_text_features_neg], dim=0)
        text_features_neg /= text_features_neg.norm(dim=-1, keepdim=True)
        text_features_neg = text_features_neg.to(torch.float32)
        
        neg_sim = []
        for i in range(0, noun_length + adj_length, emb_batchsize):
            tmp = text_features_neg[i: i + emb_batchsize] @ text_features_pos.T
            tmp = tmp.to(torch.float32)
            sim = torch.quantile(tmp, q=pencentile, dim=-1)
            maximum = torch.max(tmp, dim=1)[0]
            sim[maximum > 0.95] = 1.0
            neg_sim.append(sim)

        neg_sim = torch.cat(neg_sim, dim=0)
        neg_sim_noun = neg_sim[:noun_length]
        neg_sim_adj = neg_sim[noun_length:]


        ind_noun = torch.argsort(neg_sim_noun)
        ind_adj = torch.argsort(neg_sim_adj)

        selected_words = [raw_words_noun[i] for i in ind_noun[0:int(len(ind_noun) * neg_top_p)].cpu().tolist()] + \
                        [raw_words_adj[i] for i in ind_adj[0:int(len(ind_adj) * neg_top_p)].cpu().tolist()]  

        adj_start_idx = int(len(ind_noun) * neg_top_p)
        selected_words_noun = selected_words[:adj_start_idx]
        selected_words_adj = selected_words[adj_start_idx:]

    return selected_words_noun, selected_words_adj