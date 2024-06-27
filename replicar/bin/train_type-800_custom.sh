#!/bin/bash
target="type"
size="800_custom"
epochs=50
batch_size=20
n_workers=8
lr="1e-4"
weight_decay="1e-5"
variance=150

model_dir="$HOME/tfm/replicar/models/$target-$size"
replicar="$HOME/tfm/replicar/train.py \
    --target $target \
    --size $size \
    --batch_size $batch_size \
    --n_workers $n_workers \
    --model_path $model_dir \
    --epochs $epochs \
    --lr $lr \
    --weight_decay $weight_decay \
    --variance_threshold $variance \
    $@"
datetime=$(date +%Y%m%d-%H%M%S)

mkdir -p $model_dir
echo $replicar
($replicar | tee $model_dir/${datetime}_stdout.log) 3>&1 1>&2 2>&3 | tee $model_dir/${datetime}_stderr.log