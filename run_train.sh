#!/bin/bash

accelerate launch train.py \
    --train_file /work/ogalolu/datasets/final_small_en_0.3\
    --output_dir /work-ceph/ogalolu/models/trained_models \
    --is_contrastive
    --seed 42 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 128 \
    --num_train_epochs 5\
    --num_warmup_steps 1000 \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear\
    --weight_decay 1e-5 \
    --logging_steps 100 \
    --lower_dropout 0.1 \
    --upper_nhead 8 \
    --upper_num_layers 2 \
    --upper_activation gelu \
    --upper_dim_feedforward 2048\
    --scale 20 \
    --use_hard_negatives \
    --upper_positional \
    --max_seq_length 128 \
    --max_document_length 32 \
    --stride 0 \
    --upper_pooling mean \
    --model_name_or_path sentence-transformers/LaBSE\
    --per_device_train_batch_size 128
    
# --use_sliding_window_tokenization \
# --frozen \
# --upper_positional

# --max_seq_length 128 \
# --max_document_length 32 \
# --upper_dim_feedforward 2048\
# --upper_num_layers 2 \


