#!/bin/bash
model_dir="$HOME/tfm/models/type-800"
replicar="$HOME/tfm/replicar/train.py type-800 --target type --size 800 --batch_size 10"

($replicar | tee $model_dir/stdout.log) 3>&1 1>&2 2>&3 | tee $model_dir/stderr.log

