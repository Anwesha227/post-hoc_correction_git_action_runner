#!/bin/bash

datasets=(
    "semi-aves"
    # "fgvc-aircraft"
    # "stanford_cars"
    # "eurosat"
    # "dtd"
)

for dataset in "${datasets[@]}"; do
    echo ""
    echo "fewshot finetuning on $dataset"
    bash scripts/run_dataset_seed_fewshot_finetune.sh $dataset 1
done