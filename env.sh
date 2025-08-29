ml purge && ml load GCCcore/13.2.0 git/2.42.0 pigz/2.8 Anaconda3/2024.02-1

export PS1=" \[\033[34m\]\u@\h \[\033[33m\]\w\[\033[31m\]\[\033[00m\] $ "

# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1

conda activate swat