#!/bin/bash

## ENVIRONMENT SETTINGS
#SBATCH --get-user-env=L          # Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=probing

#SBATCH --account=142713661449
#SBATCH --time=1:00:00
#SBATCH --ntasks=9
#SBATCH --ntasks-per-node=9
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --output=slurm_log/%j_%x.log
##SBATCH --gres=gpu:a100:1             # Request 1 GPUs per node
##SBATCH --gres=gpu:a30:1             # Request 1 GPUs per node
##SBATCH --gres=gpu:rtx:1
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu          # Submit job to the gpu queue
#SBATCH --mail-type=END
#SBATCH --mail-user=tian.liu@tamu.edu

##----------- execute python script
bash scripts/run_dataset_seed_probing.sh semi-aves 1

##----------- Job Environment variables
echo "Job ID: $SLURM_JOBID"
echo "Job submission directory: $SLURM_SUBMIT_DIR"
echo "Temp directory: $TMPDIR"
echo "Scratch directory: $SCRATCH"