#!/bin/bash
model_dir="$HOME/tfm/replicar/models/type-800-2"
replicar="$HOME/tfm/replicar/train.py type-800-2 --target type --size 800 --batch_size 20 --n_workers 8 --model_path $model_dir"
datetime=$(date +%Y%m%d-%H%M%S)

mkdir -p $model_dir
($replicar | tee $model_dir/${datetime}_stdout.log) 3>&1 1>&2 2>&3 | tee $model_dir/${datetime}_stderr.log

