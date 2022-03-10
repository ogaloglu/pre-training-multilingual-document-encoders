#!/bin/bash

accelerate launch evaluate.py \
    --test_file /work/ogalolu/datasets/mldoc/en/test \
    --finetuned_dir /home/ogalolu/thesis/trained_models/mbert_2_frozen_4__2022_03_10-15_38_36 \
    --per_device_eval_batch_size 32 \
    --output_dir ../results/mldoc \
    --seed 42 \
    --max_seq_length 128 \
    --preprocessing_num_workers 8 \
    --max_document_length 32
