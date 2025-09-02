# Post-hoc Correction



### Create environment

Create conda environment and install dependencies.

```bash
# lab server has CUDA version 12.8, thus using pytorch-cuda=12.1 for compatibility
# DINOv3 requries python=3.10

conda create -n posthoc_py310 python=3.10 -y
conda activate posthoc_py310
conda install pytorch torchvision torchaudio torchmetrics pytorch-cuda=12.1 -c pytorch -c nvidia

# install openclip and clip
pip install open_clip_torch
pip install git+https://github.com/openai/CLIP.git

pip install pandas scikit-learn 

# clone dinov3
git clone https://github.com/facebookresearch/dinov3.git

# install gdown for downloading datasets
pip install gdown

```

### Dataset Prepraration

Prepare the datasets following the instructions in [DATASETS.md](./DATASETS.md).

### Usage

```bash
# activate conda environment on HPRC
. env.sh

# few-shot linear probing
bash scripts/run_dataset_seed_probing.sh semi-aves 1

# few-shot finetuning
bash scripts/run_dataset_seed_fewshot_finetune.sh semi-aves 1

# obtain top-k predictions on test set for a pretrained model
bash scripts/run_dataset_seed_topk.sh semi-aves 1

# query MLLM for posthoc correction


```