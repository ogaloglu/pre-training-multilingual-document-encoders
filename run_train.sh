#!/bin/bash

accelerate launch train.py \
    --train_file /work/ogalolu/datasets/final_small_en_0.3\
    --output_dir /work-ceph/ogalolu/models/trained_models \
    --seed 42 \
    --model_name_or_path sentence-transformers/LaBSE\
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 16 \
    --max_seq_length 128 \
    --max_document_length 32 \
    --num_train_epochs 4\
    --num_warmup_steps 1000 \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear\
    --weight_decay 0.0 \
    --logging_steps 500 \
    --frozen \
    --lower_dropout 0.1 \
    --upper_nhead 8 \
    --upper_num_layers 2 \
    --upper_activation gelu \
    --upper_pooling mean \
    --upper_dim_feedforward 2048\
    --scale 20 \
    --use_hard_negatives \
    --upper-positional \
    --is_contrastive

#  inspect
# upper-positional
# use_hard_negatives


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