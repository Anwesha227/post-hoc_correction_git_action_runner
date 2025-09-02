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

# For MLLM inference (for instance Qwen) using API, you need to add your API key to a .env file.
Create a .env file in your main project directory.
Add the API key (NEBIUS_API_KEY or OPENROUTER_API_KEY) to the file as follows (this is an example):
NEBIUS_API_KEY = "your API key here"
Save the file.

# For MLLM inference using Ollama (local runtime + model manager)
Download Ollama here: https://ollama.com/download
Pull the model as shown in this example: ollama pull qwen2.5vl:7b
Run sanity check: ollama run qwen2.5vl:7b "Explain the difference between hawks and falcons."
Within your environment, run: pip install ollama

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
