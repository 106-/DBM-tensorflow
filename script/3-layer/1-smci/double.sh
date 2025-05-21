#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/"`date +%Y-%m-%d_%H-%M-%S`"_double_3-layer_1-smci/"
echo "Results directory: $DIR"
mkdir -p $DIR
echo "Contents of results directory after creation:"
ls -la $DIR

for i in `seq $1`
do
    echo "Starting iteration $i"
    ../../../train_main.py ../../../config/3-layer/1-smci/double.json 1 -d $DIR > "$DIR/train_main_iteration_$i.log" 2>&1
    echo "Finished iteration $i"
    echo "Contents of results directory after iteration $i:"
    ls -la $DIR
done