#!/bin/bash
target="type"
size="7000_224"
epochs=50
model_dir="$HOME/tfm/replicar/models/$target-$size"
replicar="$HOME/tfm/replicar/train.py --target type --size $size --batch_size 20 --n_workers 8 --model_path $model_dir --epochs $epochs $@"
datetime=$(date +%Y%m%d-%H%M%S)

mkdir -p $model_dir
($replicar | tee $model_dir/${datetime}_stdout.log) 3>&1 1>&2 2>&3 | tee $model_dir/${datetime}_stderr.log

