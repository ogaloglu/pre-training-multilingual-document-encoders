#!/bin/bash

accelerate launch train.py \
    --train_file /work/ogalolu/datasets/ #Add \
    #--validation_file #Add \
    --pretrained_dir /home/ogalolu/thesis/trained_models/2022_03_08-01_59_07 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3\
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 1000 \
    --output_dir ../finetuned_models \
    --seed 42 \
    --max_seq_length 128 \
    --preprocessing_num_workers 8 \
    --max_document_length 32 \
    #--frozen \ # check
    --dropout 0.1
