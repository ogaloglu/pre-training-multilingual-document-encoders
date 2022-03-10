#!/bin/bash

accelerate launch finetuning.py \
    --train_file /work/ogalolu/datasets/mldoc/en/train \
    --validation_file /work/ogalolu/datasets/mldoc/en/validation \
    --pretrained_dir /home/ogalolu/thesis/trained_models/mbert_2_frozen_4__2022_03_10-15_38_36 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 3\
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 1000 \
    --output_dir ../finetuned_models/mldoc \
    --seed 42 \
    --max_seq_length 128 \
    --preprocessing_num_workers 8 \
    --max_document_length 32 \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --dropout 0.1
