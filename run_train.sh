#!/bin/bash

python train.py \
    --train_file C:\\Users\\onurg\\.cache\\huggingface\\datasets\\updated_wiki40b\\tiny_dataset \
    #--validation_split_percentage \
    --model_name_or_path bert-base-multilingual-cased \
    #--config_name \
    #--tokenizer_name \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3\
    #--max_train_steps 100000\
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 3000 \
    --output_dir ../trained_models\
    --seed 42 \
    #--model_type \
    --max_seq_length 128 \
    --preprocessing_num_workers 4 \
    #--scale \
    #--similarity_fct \
    #--tokenizer_file \
    --max_document_length 32 \
    --upper_hidden_dimension 768 \
    --upper_nhead 8 \
    #--upper_dim_feedforward \
    #--upper_dropout \
    #--upper_activation \
    #--upper_layer_norm_eps \
    --upper_num_layers 2 \
    --frozen True \
    --upper_positional True
