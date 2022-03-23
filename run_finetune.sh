#!/bin/bash

accelerate launch finetuning.py \
    --train_file /work/ogalolu/datasets/mldoc1000/en \
    --pretrained_dir /work-ceph/ogalolu/models/trained_models/labse_2_frozen_hard_128_mean_20__2022_03_23-00_34_36/model.pth \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --num_train_epochs 3\
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 500 \
    --output_dir /work-ceph/ogalolu/models/finetuned_models/mldoc1000 \
    --seed 42 \
    --max_seq_length 128 \
    --preprocessing_num_workers 8 \
    --max_document_length 32 \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --logging_steps 50 \
    --max_train_steps 10000 \
    --dropout 0.1 \
    --custom_model hierarchical

    # --frozen \
    # --custom_from_scratch
    # --custom_model hierarchical\
    # --custom_model sliding_window \
    # /home/ogalolu/thesis/trained_models/labse_2_frozen_hard_3__2022_03_16-22_59_21
    #/home/ogalolu/thesis/trained_models/labse_2_frozen_3__2022_03_16-19_00_46