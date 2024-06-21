#!/bin/bash

function doit()
{
    python start_file.py --file $1 --dim 4
}

for f in {MSE_230703_RelaxoAI_01,MSE_230804_RelaxoAI_01CPZ,MSE_230811_RelaxoAI_02CPZ,MSE_703_RelaxoAI_02}
do
    doit $f &
done