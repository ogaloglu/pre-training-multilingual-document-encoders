PROJECT_HOME=/home/ogalolu/thesis/pre-training-multilingual-document-encoders/

# MAX_SEQ_LENGTH=4096
# CUSTOM_MODEL=longformer
# PRETRAINED_DIR=/work-ceph/ogalolu/models/long_models/labse-4096

MAX_SEQ_LENGTH=128
CUSTOM_MODEL=hierarchical
PRETRAINED_DIR=/work-ceph/ogalolu/models/trained_models/labse_2_frozen_hard_128_cls_mean_0.0005_2022_04_19-12_31_23

accelerate launch $PROJECT_HOME/retrieval_finetuning/dual_encoder_finetuning.py \
    --output_dir /work-ceph/ogalolu/models/finetuned_models/clef \
    --pretrained_dir $PRETRAINED_DIR \
    --max_seq_length $MAX_SEQ_LENGTH \
    --train_file /work/ogalolu/datasets/clef_2002_16\
    --per_device_train_batch_size 8\
    --per_device_eval_batch_size 8\
    --gradient_accumulation_steps 1\
    --learning_rate 2e-5 \
    --logging_steps 100 \
    --num_warmup_steps 1000 \
    --weight_decay 1e-4 \
    --seed 42 \
    --preprocessing_num_workers 32 \
    --max_patience 7\
    --unfreeze \
    --num_train_epochs 50\
    --pretrained_epoch 9\
    --article_numbers 8\
    --custom_model $CUSTOM_MODEL
