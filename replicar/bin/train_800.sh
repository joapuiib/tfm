#!/bin/bash
model_dir="$HOME/tfm/models/type-800"
replicar="$HOME/tfm/replicar/train.py type-800 --target type --size 800 --batch_size 24 --n_workers 8"
datetime=$(date +%Y%m%d-%H%M%S)

($replicar | tee $model_dir/${datetime}_stdout.log) 3>&1 1>&2 2>&3 | tee $model_dir/${datetime}_stderr.log

