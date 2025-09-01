# Post-hoc Correction



### Create environment

Create conda environment and install dependencies.

```bash
conda create -n posthoc python=3.9

conda activate posthoc
conda install -y pytorch torchvision torchaudio torchmetrics -c pytorch

# install openclip and clip
pip install open_clip_torch
pip install git+https://github.com/openai/CLIP.git

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

# obtain predictions on test set for a pretrained model


# query MLLM for posthoc correction


```