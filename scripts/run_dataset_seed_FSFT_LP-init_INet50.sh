#!/bin/bash

# Define arrays of values for each parameter

# prefix="ablate_temp_FSFT_resnet50_clip"
# prefix="batch_FSFT_ImageNet-RN50"
prefix="test_batch_FSFT_ImageNet-RN50"


# methods=("mixup" "saliencymix" "CMO" "cutmix-fs" "resizemix" "CMLP" "probing" "finetune" "FLYP" "cutmix" "fixmatch")
methods=("finetune")

# data_sources=("fewshot" "retrieved" "fewshot+retrieved" "fewshot+unlabeled" "fewshot+retrieved+unlabeled")
data_sources=("fewshot")

unlabeled_data_source=("fewshot+unlabeled") # not adding retrieved OOD data to the unlabeled pool

# shot_values=(4 8 16)
shot_values=(16)

retrieval_splits=("T2T500")

unlabeled_splits=("u_train_in.txt")

batch_size=32

loss="CE"

epochs=1
# epochs=50


model_cfgs=(
    # "resnet50_scratch" \
    "resnet50_imagenet_pretrained" \
    # "resnet50_inat_pretrained" \
    # "vitb32_openclip_laion400m" \
    # "vitb16_openclip_laion400m" \
    # "resnet50_clip" \
    # "vitb32_clip_inat" \
    # "vitb32_clip_nabirds" \
    # "vitb32_clip_cub" \
    # "dinov2_vits14_reg" \
    # "dinov2_vitb14_reg" \
    # "dinov2_vitl14_reg" \
    # "dinov2_vitg14_reg" \
    )

# temp_scheme_list=('fewshot+retrieved+unlabeled' 'none')
# temp_scheme_list=('fewshot+retrieved+unlabeled')
temp_scheme_list=('none')


# temperature_list=(0.07)
# temperature_list=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 2.0 3.0)
# temperature_list=(0.01 0.07 0.1 1.0)
# temperature_list=(0.1 1.0)
temperature_list=(1.0)


log_mode="both"


#------------------------------
# DO NOT MODIFY BELOW THIS LINE !!!
#------------------------------

for model_cfg in "${model_cfgs[@]}"; do

    # update learning rate based on the first item of the model_cfg
    first_item=$(echo $model_cfg | cut -d'_' -f1)
    second_item=$(echo $model_cfg | cut -d'_' -f2)
    # echo "Model: $model_cfg"
    # echo "First item: $first_item"
    # echo "Second item: $second_item"

    # resnet50_imagenet_pretrained
    if [ "$first_item" = "resnet50" ] && [ "$second_item" = "imagenet" ]; then
        lr_classifier=1e-3
        lr_backbone=1e-3
        # lr_backbone=1e-4
        wd=1e-4
        cls_inits=("random")
        optim="SGD"
        mu=5
        threshold=0.8
        lambda_u=1.0
        T=1.0

    # resnet50_inat_pretrained
    elif [ "$first_item" = "resnet50" ] && [ "$second_item" = "inat" ]; then
        lr_classifier=1e-3
        lr_backbone=1e-3
        # lr_backbone=1e-4
        wd=1e-4
        cls_inits=("random")
        optim="SGD"
        mu=5
        threshold=0.85
        lambda_u=1.0
        T=1.0

    # resnet50_clip
    elif [ "$first_item" = "resnet50" ] && [ "$second_item" = "clip" ]; then
        lr_classifier=1e-4
        lr_backbone=1e-6
        wd=1e-2
        cls_inits=("REAL-Prompt")
        optim="AdamW"
        mu=5
        threshold=0.8
        lambda_u=1.0
        T=0.01
        # T=1.0

    # # resnet50_clip
    # elif [ "$first_item" = "resnet50" ] && [ "$second_item" = "clip" ]; then
    #     lr_classifier=1e-3
    #     lr_backbone=1e-3
    #     wd=1e-4
    #     cls_inits=("REAL-Prompt")
    #     optim="SGD"
    #     mu=5
    #     threshold=0.8
    #     lambda_u=1.0
    #     T=0.01 # 1.0 does not work, gives mask of 0

    # openlcip or clip
    elif [ "$first_item" = "vitb32" ] || [ "$first_item" = "vitb16" ]; then
        lr_classifier=1e-4
        lr_backbone=1e-6
        wd=1e-2
        cls_inits=("REAL-Prompt")
        optim="AdamW"
        mu=5
        threshold=0.8
        lambda_u=1.0
        T=0.01
        # T=1.0

    # DINOv2
    elif [ "$first_item" = "dinov2" ]; then
        lr_classifier=1e-4
        lr_backbone=1e-6
        wd=1e-2
        cls_inits=("REAL-Prompt")
        optim="AdamW"
        mu=2 # larger would run OOM
        threshold=0.85
        lambda_u=1.0
        T=0.1

    else
        echo "Model not found"
        exit 1
    fi


    # update folder by adding the model_cfg
    folder="${prefix}_${model_cfg}_${epochs}epochs"
    echo "Folder: $folder"

    # Split the model_cfg by underscore and get the second item
    second_item=$(echo $model_cfg | cut -d'_' -f2)

    if [ "$second_item" = "openclip" ] || [ "$second_item" = "clip" ]; then
        script="main.py"
    else
        script="main_ssl.py"
    fi


    # Check if command-line arguments were provided
    if [ "$#" -ge 2 ]; then
        datasets=("$1")  # Use the provided command-line argument for the dataset
        seeds=("$2")
    else
        echo "Usage: $0 <dataset> [seed]"
    fi


    # Check if the results folder exists, if not create it
    if [ ! -d "results/$folder" ]; then
        mkdir -p "results/$folder"
    fi

    output_folder="output/$folder"
    if [ ! -d "$output_folder" ]; then
        mkdir -p "$output_folder"
    fi


    # Dynamically set the filename based on the dataset
    output_file="results/${folder}/${datasets[0]}.csv"

    # Create or clear the output file
    echo "Dataset,Method,Loss,Model,DataSource,Init,Shots,Seed,Retrieve,Stage1Acc,Stage2Acc" > "$output_file"

    # Loop through all combinations and run the script
    for dataset in "${datasets[@]}"; do
        for method in "${methods[@]}"; do
            for data_source in "${data_sources[@]}"; do
                for shots in "${shot_values[@]}"; do
                    for init in "${cls_inits[@]}"; do
                        for seed in "${seeds[@]}"; do
                            for retrieval_split in "${retrieval_splits[@]}"; do
                                for unlabeled_split in "${unlabeled_splits[@]}"; do
                                    for temp_scheme in "${temp_scheme_list[@]}"; do
                                        for temperature in "${temperature_list[@]}"; do

                                            echo "Running: $script $dataset $method $loss $model_cfg $data_source $init $shots $seed $retrieval_split $unlabeled_in_split"

                                            # set the cls_path based on linear probing learned weights
                                            cls_path="output/LinearProbing_${model_cfg}_50epochs/output_${dataset}/${dataset}_probing_fewshot_${init}_${shots}shots_seed${seed}/stage1_model_best.pth"

                                            # imagenet pretrained RN50
                                            # cls_path="output_backup/CE_probing_fewshot_resnet50_imagenet_pretrained/output_semi-aves/semi-aves_probing_fewshot_random_16shots_seed1_50eps/stage1_model_best.pth"


                                            # Run the script and capture the output
                                            output=$(python -W ignore "$script" --prefix "$prefix" --dataset "$dataset" --method "$method" \
                                                    --data_source "$data_source" --unlabeled_data_source "$unlabeled_data_source" --cls_init "$init" \
                                                    --shots "$shots" --seed "$seed" --epochs "$epochs" --bsz "$batch_size" \
                                                    --lr_classifier "$lr_classifier"  --lr_backbone "$lr_backbone" --wd "$wd" \
                                                    --optim "$optim" --loss_name "$loss"\
                                                    --log_mode "$log_mode" --retrieval_split "${retrieval_split}.txt" \
                                                    --unlabeled_split "$unlabeled_split" \
                                                    --mu "$mu" --threshold "$threshold" --lambda_u "$lambda_u" --T "$T" \
                                                    --temp_scheme "$temp_scheme" --temperature "$temperature" \
                                                    --model_cfg "$model_cfg" --folder "$output_folder" \
                                                    --check_zeroshot \
                                                    --cls_path "$cls_path" \
                                                    # --skip_stage2 \
                                                    )

                                            # Print the output to the console
                                            echo "$output"

                                            # Append the results to the CSV file
                                            echo "$output" >> "$output_file"

                                            echo ""

                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done