#!/bin/bash

accelerate launch src/evaluate.py \
    --test_file /work/ogalolu/datasets/mldoc1000/en/test \
    --finetuned_dir /home/ogalolu/thesis/finetuned_models/mldoc1000/labse__2022_03_18-00_55_28/  \
    --per_device_eval_batch_size 128 \
    --output_dir work-ceph/ogalolu/results/mldoc \
    --seed 42 \
    --max_seq_length 128 \
    --preprocessing_num_workers 8 \
    --max_document_length 32
