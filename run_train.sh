#!/bin/bash

accelerate launch train.py \
    --train_file /work/ogalolu/datasets/medium_final_small_en_0.3 \
    --model_name_or_path bert-base-multilingual-cased \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 4\
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 50 \
    --output_dir ../trained_models \
    --seed 42 \
    --max_seq_length 128 \
    --preprocessing_num_workers 16 \
    --max_document_length 32 \
    --upper_nhead 8 \
    --upper_num_layers 2 \
    --frozen \
    --upper_positional \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --logging_steps 50 \
    --lower_dropout 0.2 \
    --is_contrastive


# accelerate launch train.py \
#     --train_file /work/ogalolu/datasets/final_small_en_0.3 \
#     --model_name_or_path bert-base-multilingual-cased \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --num_train_epochs 4\
#     --gradient_accumulation_steps 1 \
#     --num_warmup_steps 3000 \
#     --output_dir ../trained_models \
#     --seed 42 \
#     --max_seq_length 128 \
#     --preprocessing_num_workers 16 \
#     --max_document_length 32 \
#     --upper_nhead 8 \
#     --upper_num_layers 2 \
#     --frozen \
#     --upper_positional \
#     --learning_rate 5e-5 \
#     --weight_decay 0.0 \
#     --is_contrastive