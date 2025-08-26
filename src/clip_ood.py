import clip
import torch
import pickle
import time
import os
import random

from class_names import CLASS_NAME, preset_noun_prompt_templates, preset_adj_prompt_templates, preset_noun_prompt_templates_for_sketch
from create_negs import create_initial_negative_labels
from neg_filter import neg_filter


def _save_labels(labels, file_name):
    # Save in a txt file to see the labels
    f = open(file_name+'.txt', 'w')
    for w in labels:
        f.write(w + '\n')
    f.close()

    # Save using pickle for easier reload
    with open(file_name + '.pkl', 'wb') as fp:
        pickle.dump(labels, fp)

    return


def _load_labels(file_name):
    if not os.path.exists(file_name + '.pkl'):
        raise FileNotFoundError(f"Label file not found: {file_name}.pkl")

    with open(file_name + '.pkl', 'rb') as fp:
        labels = pickle.load(fp)
    return labels


class CLIPood():
    def __init__(self,
                 train_dataset='imagenet',
                 arch='ViT-B/16',
                 seed=0,
                 device='cuda:0',
                 output_folder='output/',
                 load_saved_labels=True):
        
        self.device = device
        random.seed(seed)

        self.clip_model, self.clip_preprocess = clip.load(arch, self.device, jit=False)
        self.clip_model.eval()
        
        class_name = CLASS_NAME[train_dataset]
        
        if train_dataset == 'imagenet_sketch':
            self.noun_prompt_templates = preset_noun_prompt_templates_for_sketch
        else:
            self.noun_prompt_templates = preset_noun_prompt_templates
        
        # Number of top positive/negative labels to use when generating multi-matching texts
        self.k_pos = min(5, len(CLASS_NAME[train_dataset]))
        self.k_neg = 5

        # NegLabel paramters
        self.ngroup = 100
        self.group_fuse_num = None
        
        if load_saved_labels is True:
            print(f"\n--- Loading negative labels from folder: {output_folder}")
            neg_labels_noun = _load_labels(output_folder+'neg_labels_noun')
            neg_labels_adj = _load_labels(output_folder+'neg_labels_adj')
        else:
            print("\n--- Creating initial negative labels using CSP")
            start_time = time.time()
            initial_neg_labels_noun, initial_neg_labels_adj = create_initial_negative_labels(self.clip_model, train_dataset=train_dataset, neg_top_p=0.15, seed=seed, device=self.device)
            print(f"time: {time.time() - start_time:.2f} seconds")
            print(f"number of initial negative labels — nouns: {len(initial_neg_labels_noun)}, adjectives: {len(initial_neg_labels_adj)} - total: {len(initial_neg_labels_noun)+len(initial_neg_labels_adj)}")
            
            _save_labels(initial_neg_labels_noun, output_folder+'initial_neg_labels_noun')
            _save_labels(initial_neg_labels_adj, output_folder+'initial_neg_labels_adj')

            print("\n--- Filtering negative labels using NegFilter")
            start_time = time.time()
            neg_labels_noun, neg_labels_adj = neg_filter(initial_neg_labels_noun, initial_neg_labels_adj, class_name, self.clip_model, seed=seed, device=self.device, output_folder=output_folder)
            print(f"time: {time.time() - start_time:.2f} seconds")
            _save_labels(neg_labels_noun, output_folder+'neg_labels_noun')
            _save_labels(neg_labels_adj, output_folder+'neg_labels_adj')
            
        
        print(f"number of final negative labels — nouns: {len(neg_labels_noun)}, adjs: {len(neg_labels_adj)} - total: {len(neg_labels_noun)+len(neg_labels_adj)}")
        print("examples of negative labels nouns:", neg_labels_noun[:3])
        print("examples of negative labels adjs:", neg_labels_adj[:3])

        self.pos_features = self._embed_text_labels(labels=class_name, prompt_templates=self.noun_prompt_templates)
        
        neg_noun_features = self._embed_text_labels(labels=neg_labels_noun, prompt_templates=self.noun_prompt_templates)
        neg_adj_features = self._embed_text_labels(labels=neg_labels_adj, prompt_templates=preset_adj_prompt_templates)
        self.neg_features = torch.cat([neg_noun_features, neg_adj_features], dim=0)

        self.pos_labels = class_name
        self.neg_labels = neg_labels_noun + neg_labels_adj
        
        print("pos_features shape: ", self.pos_features.shape)
        print("neg_features shape: ", self.neg_features.shape)


    def _embed_text_labels(self, labels, prompt_templates, batch_size=1000):
        texts = []
        for w in labels:
            for template in prompt_templates:
                texts.append(template.format(w))
        
        tokenized = torch.cat([clip.tokenize(f"{c}") for c in texts]).to(self.device)

        with torch.no_grad():
            features = []
            for i in range(0, len(tokenized), batch_size):
                x = self.clip_model.encode_text(tokenized[i: i + batch_size])
                features.append(x)
            features = torch.cat(features, dim=0)

            features = features.view(-1, len(prompt_templates), features.shape[-1]).mean(dim=1)
            features /= features.norm(dim=-1, keepdim=True)
            features = features.to(torch.float32)
        
        return features


    def _grouping(self, pos, neg, num, ngroup=10, random_permute=False):
        group = ngroup
        drop = neg.shape[1] % ngroup
        if drop > 0:
            neg = neg[:, :-drop]
        if random_permute:
            SEED=0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            idx = torch.randperm(neg.shape[1], device=self.device)
            neg = neg.T
            negs = neg[idx].T.reshape(pos.shape[0], group, -1).contiguous()
        else:
            negs = neg.reshape(pos.shape[0], group, -1).contiguous()
        scores = []
        for i in range(group):
            full_sim = torch.cat([pos, negs[:, i, :]], dim=-1)
            full_sim = full_sim.softmax(dim=-1)
            pos_score = full_sim[:, :pos.shape[1]].sum(dim=-1)
            scores.append(pos_score.unsqueeze(-1))
        scores = torch.cat(scores, dim=-1)
        if num is not None:
            scores = scores[:,0:num-1]
        score = scores.mean(dim=-1)
        
        return score


    def _neg_label_score(self, pos_sim, neg_sim):
        full_sim = torch.cat([pos_sim, neg_sim], dim=-1)

        if self.ngroup > 1:
            score = self._grouping(pos_sim, neg_sim, num=self.group_fuse_num, ngroup=self.ngroup, random_permute=True)
        else:
            full_sim = full_sim.softmax(dim=-1)
            pos_score = full_sim[:, :pos_sim.shape[1]].sum(dim=1)
            score = pos_score

        return score
    

    def _multi_matching_score(self, pos_sim, neg_sim, image_features):
        vals, ind = torch.topk(pos_sim, self.k_pos)
        top_pos_labels = [self.pos_labels[ind[0, i]] for i in range(self.k_pos)]

        vals, ind = torch.topk(neg_sim, self.k_neg)
        top_neg_labels = [self.neg_labels[ind[0, i]] for i in range(self.k_neg)]

        
        multi_matching_texts = []
        neg_sims = []
        for p in top_pos_labels:
            for i, n in enumerate(top_neg_labels):
                w = p + " and " + n
                neg_sims.append(vals[0, i])
                multi_matching_texts.append(w)
                
        multi_matching_features = self._embed_text_labels(multi_matching_texts, self.noun_prompt_templates)
        neg_sims = torch.tensor(neg_sims).to(self.device)

        sim = (100.0 * image_features @ multi_matching_features.T)
        ss = torch.exp(sim) / (torch.exp(sim) + torch.exp(neg_sims))
        score = torch.max(ss)

        return score


    # Final detection score
    def detection_score(self, img):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(img)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.to(torch.float32)

            pos_sim = (100.0 * image_features @ self.pos_features.T)            
            neg_sim = (100.0 * image_features @ self.neg_features.T)
            
            neg_label_score = self._neg_label_score(pos_sim, neg_sim)
            multi_matching_score = self._multi_matching_score(pos_sim, neg_sim, image_features)
            
        # return neg_label_score.item() + 2*multi_matching_score.item(), neg_label_score.item() + 4*multi_matching_score.item()
        return neg_label_score.item() + 2*multi_matching_score.item()
