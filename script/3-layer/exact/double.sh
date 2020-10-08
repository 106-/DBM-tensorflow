#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/"`date +%Y-%m-%d_%H-%M-%S`"_double_3-layer_exact/"
mkdir -p $DIR

for i in `seq $1`
do
    ./train_main.py ./config/3-layer/exact/double.json 5000 -d $DIR
done