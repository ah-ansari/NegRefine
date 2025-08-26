import torch
import torchvision as tv
from torchvision.datasets import ImageNet
import numpy as np
import time
from tqdm import tqdm
import argparse

import ood_evaluate
import clip_ood


def load_dataset(dataset_name):
    data_path = 'data/'

    if dataset_name == 'imagenet':
        dataset = ImageNet(root=data_path+'ImageNet/', split='val')
    elif dataset_name == 'inaturalist':
        dataset = tv.datasets.ImageFolder(data_path+"iNaturalist/")
    elif dataset_name == 'sun':
        dataset = tv.datasets.ImageFolder(data_path+"SUN/images/")
    elif dataset_name == 'texture':
        dataset = tv.datasets.ImageFolder(data_path+"dtd/images/")
    elif dataset_name == 'places':
        dataset = tv.datasets.ImageFolder(data_path+"Places/")
    elif dataset_name == 'clean':
        dataset = tv.datasets.ImageFolder(data_path+"NINCO_popular_datasets_subsamples/")
    elif dataset_name == 'ninco':
        dataset = tv.datasets.ImageFolder(data_path+"NINCO_OOD_classes/")
    elif dataset_name == 'open_image':
        dataset = tv.datasets.ImageFolder(data_path+"openimage_o/")
    elif dataset_name == 'imagenet10':
        dataset = tv.datasets.ImageFolder(data_path+"ImageNet10/")
    elif dataset_name == 'imagenet20':
        dataset = tv.datasets.ImageFolder(data_path+"ImageNet20/")
    elif dataset_name == 'imagenet100':
        dataset = tv.datasets.ImageFolder(data_path+"ImageNet100/")
    elif dataset_name == 'imagenetA':
        dataset = tv.datasets.ImageFolder(data_path+"imagenet-a/")
    elif dataset_name == 'imagenetR':
        dataset = tv.datasets.ImageFolder(data_path+"imagenet-r/")
    elif dataset_name == 'imagenetv2':
        dataset = tv.datasets.ImageFolder(data_path+"imagenetv2-matched-frequency/imagenetv2-matched-frequency-format-val/")
    elif dataset_name == 'imagenet_sketch':
        dataset = tv.datasets.ImageFolder(data_path+"imagenet-sketch/sketch/")
    elif dataset_name == 'car196':
        dataset = tv.datasets.ImageFolder(data_path+"cars/")
    elif dataset_name == 'pet37':
        dataset = tv.datasets.ImageFolder(data_path+"oxford_pet/")
    elif dataset_name == 'bird200':
        dataset = tv.datasets.ImageFolder(data_path+"CUB_200_2011/CUB_200_2011/images/")
    elif dataset_name == 'food101':
        dataset = tv.datasets.ImageFolder(data_path+"food101/images/")
    elif dataset_name == 'waterbird':
        dataset = tv.datasets.ImageFolder(data_path+"waterbird_complete95_forest2water2/")
    elif dataset_name == 'placesbg':
        dataset = tv.datasets.ImageFolder(data_path+"placesbg/")
    elif dataset_name == 'imagenet99':
        dataset = tv.datasets.ImageFolder(data_path+"ImageNet99/")
    else:
        print(f"Dataset name {dataset_name} does not match with any of the datasets." )
        exit(1)
    
    return dataset


def get_scores(dataset, ood_model):
    scores = []
    for i in tqdm(range(len(dataset))):
        image, class_id = dataset[i]

        image = ood_model.clip_preprocess(image).unsqueeze(0).to(ood_model.device)
        result = ood_model.detection_score(image)

        scores.append(result)
    
    return scores


def print_result(scores_in, scores_ood):
    labels = np.ones(len(scores_in) + len(scores_ood))
    labels[:len(scores_in)] = -1

    scores_in = np.array(scores_in)
    scores_ood = np.array(scores_ood)

    if scores_in.ndim == 1:
        scores_in = scores_in.reshape(-1, 1)
    
    if scores_ood.ndim == 1:
        scores_ood = scores_ood.reshape(-1, 1)

    for i in range(scores_in.shape[1]):
        auroc, aupr_in, aupr_out, fpr = ood_evaluate.get_measures(scores_in[:, i], scores_ood[:, i])
        print(f"score{i}     auroc: {auroc}    fpr: {fpr}")

    return


def evaluate(in_dataset_name='imagenet', ood_dataset_name_list=['clean'], arch='ViT-B/16', seed=0, device='cuda:0', output_folder='output/', load_saved_labels=True):
    in_dataset = load_dataset(in_dataset_name)
    ood_dataset_list = []
    for dataset in ood_dataset_name_list:
        ood_dataset_list.append(load_dataset(dataset))

    ood_model = clip_ood.CLIPood(train_dataset=in_dataset_name, arch=arch, seed=seed, device=device, output_folder=output_folder, load_saved_labels=load_saved_labels)
    
    print("\n--- Starting predictions on testing images")
    start_time = time.time()

    print(f"in ({in_dataset_name}) - number: ", len(in_dataset))
    scores_in = get_scores(in_dataset, ood_model)

    scores_ood_list = []
    for i, data in enumerate(ood_dataset_list):
        print(f"ood{i} ({ood_dataset_name_list[i]}) - number: ", len(data))
        s = get_scores(data, ood_model)
        scores_ood_list.append(s)
    
    print(f"time: {time.time() - start_time:.2f} seconds")

    print("\n\n--- Final results\n")
    for i, scores in enumerate(scores_ood_list):
        print(f"------------------ {ood_dataset_name_list[i]} -------------------")
        print_result(scores_in, scores)
        print("")
    
    return
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dataset_name", type=str, default="imagenet", help="Name of in-distribution dataset (default: imagenet)")
    parser.add_argument("--ood_dataset_name_list", type=str, nargs='+', default=["clean"], help="List of OOD dataset names (default: ['clean'])")
    parser.add_argument("--arch", type=str, default="ViT-B/16", help="CLIP model architecture (default: ViT-B/16)")
    parser.add_argument("--seed", type=int, default=0, help="Seed value (default: 0)")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device (default: cuda:0)")
    parser.add_argument("--output_folder", type=str, default="output/", help="Folder to save outputs (default: output/)")
    parser.add_argument("--load_saved_labels", action="store_true", help="Load previously saved negative labels from output folder")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    print("------------------------------------------")
    evaluate(in_dataset_name=args.in_dataset_name,
           ood_dataset_name_list=args.ood_dataset_name_list,
           arch=args.arch, seed=args.seed, device=args.device,
           output_folder=args.output_folder, load_saved_labels=args.load_saved_labels)
