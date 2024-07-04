#!/bin/bash
model_dir="$HOME/tfm/replicar/models/top_label-800"
model_path="$model_dir/model.pt"

replicar="$HOME/tfm/replicar/train.py top_label-800 --target top_label --size 800 --batch_size 20 --n_workers 8 --model_path $model_path"
datetime=$(date +%Y%m%d-%H%M%S)

($replicar | tee $model_dir/${datetime}_stdout.log) 3>&1 1>&2 2>&3 | tee $model_dir/${datetime}_stderr.log

