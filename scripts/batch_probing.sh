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
    echo "Few-shot linear probing on $dataset"
    bash scripts/run_dataset_seed_probing.sh $dataset 1
    bash scripts/run_dataset_seed_probing.sh $dataset 2
    bash scripts/run_dataset_seed_probing.sh $dataset 3

done