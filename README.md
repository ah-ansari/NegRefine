# NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection

[![Paper](https://img.shields.io/badge/Paper-arXiv:2507.09795-b31b1b.svg)](https://arxiv.org/abs/2507.09795)
[![ICCV 2025](https://img.shields.io/badge/ICCV-2025-7b1fa2.svg)](https://iccv.thecvf.com/)
[![License](https://img.shields.io/github/license/ah-ansari/NegRefine)](https://github.com/ah-ansari/NegRefine/blob/main/LICENSE)

Official implementation of **NegRefine**, accepted to **ICCV 2025**.  
> 📄 [Paper on arXiv](https://arxiv.org/abs/2507.09795)

NegRefine improves negative label-based zero-shot OOD detection by:  
- **Filtering** subcategories and proper nouns from the *negative label* set using an LLM
- **Multi-matching-aware scoring** that accounts for images matching multiple labels  

With these improvements, NegRefine achieves **state-of-the-art results** on large-scale **ImageNet-1K** benchmark.  


## 📂 Code Overview

The repository is structured as follows:

```
neg_refine/
├─ data/                     # Dataset root (add datasets here)
├─ output/                   # Save folder for outputs and results per dataset/seed
│  └─ imagenet/seed_0/       # Example folder for ImageNet with seed 0
├─ scripts/                  # Bash scripts for running experiments
│  └─ ...
├─ src/                      # Python source code
│  ├─ class_names.py         # Dataset class names and prompt templates
│  ├─ clip_ood.py            # Main method for CLIP-based zero-shot OOD detection
│  ├─ create_negs.py         # Generates initial negative labels (CSP-based)
│  ├─ eval.py                # Entry point for experiments and evaluation
│  ├─ neg_filter.py          # LLM-based refinement of negative labels
│  └─ ood_evaluate.py        # OOD evaluation metrics (AUROC, FPR@95, etc.)
├─ txtfiles/                 # WordNet lexicon text files (adjectives/nouns)
│  └─ ...
```


## ⚙️ Environment Setup

This project was developed with **Python 3.10.12** and **PyTorch 2.6.0** on **Ubuntu 22.04**.  

- **CLIP**: We used the [OpenAI CLIP implementation](https://github.com/openai/CLIP).  
- **LLM**: For negative label filtering, we primarily used [Qwen2.5-14B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M) via Hugging Face.  
- **Other dependencies**: See [requirements.txt](./requirements.txt) for the full list of packages.



## 📦 Dataset Downloads

Below are the sources for downloading the datasets used in our experiments:

- **ImageNet-1K**: Download from the [ImageNet Challenge 2012 website](https://www.image-net.org/challenges/LSVRC/2012/index). Only the validation data is required.

- **NINCO & Clean**: Available from the [NINCO GitHub](https://github.com/j-cb/NINCO). The provided `.tar.gz` file includes both: **NINCO dataset** (`NINCO_OOD_classes`) and **Clean Collection** (`NINCO_popular_datasets_subsamples`, obtained through manual analysis of random samples from 11 common OOD datasets).

- **OpenImage-O**: Can be downloaded from [OpenOOD](https://github.com/Jingkang50/OpenOOD) using the provided [download script](https://github.com/Jingkang50/OpenOOD/blob/main/scripts/download/download.py).

- **ImageNet-10, ImageNet-20, ImageNet-100**: Refer to the [MCM GitHub](https://github.com/deeplearning-wisc/MCM) for instructions to create these subsets of ImageNet-1K classes.  
  *Note: In our experiments, we modified **ImageNet-100** to create **ImageNet-99** by removing the “race car” class (class n04037443).*  

- **iNaturalist, SUN, Places, Textures**: Download links available on the [MOS GitHub](https://github.com/deeplearning-wisc/large_scale_ood).  

- **CUB-200, Stanford Cars, Food-101, Oxford Pets**: Download links available on the [MCM GitHub](https://github.com/deeplearning-wisc/MCM).  

- **Waterbirds (Spurious OOD)**: Refer to this [MCM GitHub issue](https://github.com/deeplearning-wisc/MCM/issues/7).

After downloading, place all datasets in the `data/` folder.   
Refer to (or modify) the `load_dataset()` function in [`src/eval.py`](./src/eval.py) for the exact folder structure and naming conventions used for data loading.



## 🚀 Running Experiments

The script to run each experiment from the main paper is provided in the [`scripts/`](./scripts) folder.  
Scripts are named after the in-distribution datasets used in the experiments.  

For example, to reproduce the ImageNet-1K benchmark, run:
```bash
sh scripts/imagenet.sh
```

The results of each experiment—including evaluation metrics, logs, and negative label files—will be saved in the `output/` folder.


## 📊 Example Results

As an illustration, we provide the saved results for **ImageNet-1K** with **seed 0**, available in [`output/imagenet/seed_0/`](./output/imagenet/seed_0/). These include the saved negative labels, LLM refinement logs, and final evaluation results.

**Results (In-Distribution: ImageNet-1K, Seed 0):**
| OOD Dataset       | AUROC (%) | FPR@95 (%) |
|-------------------|-----------|------------|
| ⭐ **iNaturalist** | 99.57     | 1.51       |
| ⭐ **OpenImage-O** | 95.02     | 24.03      |
| ⭐ **Clean**       | 90.70     | 33.04      |
| ⭐ **NINCO**       | 81.90     | 62.11      |
| SUN               | 94.64     | 22.93      |
| Places            | 90.42     | 39.10      |
| Textures          | 94.69     | 21.15      |

> **Note:** Only the first four datasets are considered valid OOD data and are included in the main paper results, as they contain minimal or no in-distribution contamination. In contrast, **SUN, Places, and Textures** contain notable overlap with ImageNet-1K classes, leading to in-distribution contamination. For further discussion, refer to [our paper](https://arxiv.org/abs/2507.09795) and the [NINCO paper](https://arxiv.org/abs/2306.00826).

The table above shows results for **ImageNet-1K with seed 0**.  
For the complete set of experiments and results, averaged over 10 seeds, please refer to our main paper.


## 🙏 Acknowledgements

Our code is built on the excellent work of [CSP](https://github.com/MengyuanChen21/NeurIPS2024-CSP) and [NegLabel](https://github.com/XueJiang16/NegLabel). We sincerely thank the authors.


## 📖 Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{ansari2025negrefine,
  title={NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection},
  author={Ansari, Amirhossein and Wang, Ke and Xiong, Pulei},
  journal={arXiv preprint arXiv:2507.09795},
  year={2025}
}
```
