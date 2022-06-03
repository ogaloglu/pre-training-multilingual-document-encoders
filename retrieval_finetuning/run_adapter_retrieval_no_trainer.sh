PROJECT_HOME=/home/ogalolu/thesis/pre-training-multilingual-document-encoders/

# output directory of prepare_data.sh
DATA_DIR=/work/ogalolu/data/msmarco/

# MAX_SEQ_LENGTH=4096
# CUSTOM_MODEL=longformer
# PRETRAINED_DIR=/work-ceph/ogalolu/models/long_models/labse-4096

MAX_SEQ_LENGTH=128
CUSTOM_MODEL=hierarchical
PRETRAINED_DIR=/work-ceph/ogalolu/models/trained_models/labse_2_frozen_hard_128_mean_dcls_0.0005_2022_04_30-11_31_22

accelerate launch $PROJECT_HOME/retrieval_finetuning/adapter_retrieval_no_trainer.py \
    --output_dir /work-ceph/ogalolu/models/finetuned_models/ms_marco \
    --pretrained_dir $PRETRAINED_DIR \
    --max_seq_length $MAX_SEQ_LENGTH \
    --train_file /work/ogalolu/datasets/msmarco\
    --per_device_train_batch_size 8\
    --per_device_eval_batch_size 64\
    --gradient_accumulation_stesps 4\
    --learning_rate 2e-5 \
    --logging_steps 5000 \
    --saving_steps 20000 \
    --num_train_epochs 1 \
    --num_warmup_steps 10000 \
    --weight_decay 1e-4 \
    --seed 42 \
    --preprocessing_num_workers 32 \
    --max_patience 30 \
    --unfreeze \
    --max_train_steps 200000 \
    --pretrained_epoch 4 \
    --custom_from_scratch \
    --custom_model $CUSTOM_MODEL
# --num_warmup_steps 5000 \
# --unfreeze
# --train_file $DATA_DIR/train_sbert.jsonl \
# --validation_file $DATA_DIR/dev_sbert.jsonl \