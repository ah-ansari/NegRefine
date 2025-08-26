#!/bin/sh

# Define settings
TRAIN_DATASET="car196"
SEED=0
OUTPUT_FOLDER="output/car/seed_${SEED}/"
DEVICE="cuda:0"
OOD_DATASETS="inaturalist sun places texture"

# Create output directory
mkdir -p "$OUTPUT_FOLDER"

python src/eval.py --in_dataset_name "$TRAIN_DATASET" --ood_dataset_name_list $OOD_DATASETS --seed "$SEED" --device "$DEVICE" --output_folder "$OUTPUT_FOLDER" | tee "${OUTPUT_FOLDER}result.txt"

# # Use --load_saved_labels option to reload previously created negative labels
# python src/eval.py --in_dataset_name "$TRAIN_DATASET" --ood_dataset_name_list $OOD_DATASETS --seed "$SEED" --device "$DEVICE" --output_folder "$OUTPUT_FOLDER" --load_saved_labels | tee "${OUTPUT_FOLDER}result.txt"
