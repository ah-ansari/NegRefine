import torch
import clip
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from tqdm import tqdm

from class_names import CLASS_NAME


def _embed_text(text_list, clip_model, device, batch_size=1000):
    tokens = torch.cat([clip.tokenize(t) for t in text_list]).to(device)
    features = []

    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            batch_features = clip_model.encode_text(tokens[i:i + batch_size])
            features.append(batch_features)

    features = torch.cat(features, dim=0)
    features /= features.norm(dim=-1, keepdim=True)
    features = features.to(torch.float32)
    
    return features


def _clean_words(word_list):
    cleaned = []

    # Keep only letters, underscores, hyphens, and spaces. Also, discard words with fewer than 3 characters.
    for word in word_list:
        w = ''.join(c for c in word if c.isalpha() or c in {'_', '-', ' '})
        if len(w) > 2:
            cleaned.append(w)
    
    cleaned = list(set(cleaned))

    return cleaned


def _find_top_matching_pairs(neg_labels, pos_labels, clip_model, device="cuda:0"):
    pos_features = _embed_text(pos_labels, clip_model, device)
    neg_features = _embed_text(neg_labels, clip_model, device)

    # For each negative label, select its top-n most similar positive labels
    n = min(10, len(pos_labels))

    with torch.no_grad():
        similarity = neg_features @ pos_features.T

    pairs = []
    for i in range(len(neg_labels)):
        top_pos_labels = [pos_labels[j] for j in torch.topk(similarity[i], n).indices]
        pairs.append((neg_labels[i], top_pos_labels))
    
    return pairs


def _query_llm(prompt, tokenizer, model, device):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answering only Yes/No."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response



def _llm_filter(label_pairs, seed=0, device="cuda:0", log_file="llm_process.txt"):
    # from transformers.utils import logging
    # logging.set_verbosity_error()

    # set the seed for transformers
    set_seed(seed)

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct-1M", torch_dtype="auto", device_map=device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct-1M")
        
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M", torch_dtype="auto", device_map=device)
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")

    # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype="auto", device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    filtered_neg_labels = []

    n_proper = 0
    n_same = 0
    n_subcatgeory = 0

    f = open(log_file + '.txt', 'w')

    for i in tqdm(range(len(label_pairs))):
        neg_label = label_pairs[i][0]
        pos_labels = label_pairs[i][1]

        should_include = True

        # proper noun query
        prompt = f"Is '{neg_label}' a proper noun, like the name of a city, country, mountain, river, or any unique place or entity?"
        response = _query_llm(prompt, tokenizer, model, device)
        if response.lower().startswith("yes"):
            n_proper += 1
            should_include = False
            f.write(f"{neg_label} | proper noun \n")
            

        if should_include is True:
            for l in pos_labels:
                # check if they are the same words
                if neg_label == l:
                    n_same += 1
                    should_include = False
                    f.write(f"{neg_label} | {l} | the same \n")
                    break

                # subcategory query
                prompt = f"Is '{neg_label}' a subcatgeory of '{l}'?"
                response = _query_llm(prompt, tokenizer, model, device)
                if response.lower().startswith("yes"):
                    n_subcatgeory += 1
                    should_include = False
                    f.write(f"{neg_label} | {l} | subcategory \n")                
                    break

                
                f.write(f"{neg_label} | {l} | \n")


        if should_include == True:
            filtered_neg_labels.append(neg_label)
            
    
    f.write("---------------------------------------------\n")
    f.write("n_proper: " + str(n_proper) + "\n")
    f.write("n_same: " + str(n_same) + "\n")
    f.write("n_subcatgeory: " + str(n_subcatgeory) + "\n")
    f.close()

    return filtered_neg_labels
    

def neg_filter(neg_labels_noun, neg_labels_adj, pos_labels, clip_model, seed=0, device='cuda:0', output_folder='output/'):
    # step 1: Find the top matching positive labels for each negative label, used in the second step for subcategory check
    noun_pairs = _find_top_matching_pairs(_clean_words(neg_labels_noun), pos_labels, clip_model, device)
    adj_pairs = _find_top_matching_pairs(_clean_words(neg_labels_adj), pos_labels, clip_model, device)

    # step 2: Use the LLM to filter out negative labels that are proper nouns or subcategory of one of their top-matching positive labels
    filtered_neg_labels_noun = _llm_filter(noun_pairs, seed=seed, device=device, log_file=output_folder+'llm_process_noun')
    filtered_neg_labels_adj = _llm_filter(adj_pairs, seed=seed, device=device, log_file=output_folder+'llm_process_adj')

    return filtered_neg_labels_noun, filtered_neg_labels_adj
