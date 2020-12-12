#!/bin/bash

for i in $(seq 1 74);
do
    echo running job $i
    sbatch data_gen.sh
done