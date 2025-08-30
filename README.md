# Post-hoc Correction

### Prepraration
Create conda environment and install dependencies following the instructions in [ENV.md](./ENV.md).

Prepare the datasets following the instructions in [DATASETS.md](./DATASETS.md).

Retrieve relevant pretraining data following the instructions in [RETRIEVAL.md](./retrieval/RETRIEVAL.md).

```bash
conda create -n posthoc python=3.9

conda install -y pytorch torchvision torchaudio torchmetrics -c pytorch


```


### Usage

```bash
# activate conda environment on HPRC
. env.sh

# few-shot linear probing
bash scripts/run_dataset_seed_probing.sh semi-aves 1

# few-shot finetuning
bash scripts/run_dataset_seed_fewshot_finetune.sh semi-aves 1

```