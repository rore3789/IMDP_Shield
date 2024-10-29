#!/bin/bash

THREADS=8
REFINEMENTS=0
YAMLNAME="Setup/example.yaml"
python3 Code/setup_dkl.py $YAMLNAME
export JULIA_NUM_THREADS=$THREADS
julia Code/RKHS_Shield_Synthesis.jl $THREADS $REFINEMENTS $YAMLNAME
chmod -R 777 Systems
