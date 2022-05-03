PROJECT_HOME=/home/ogalolu/thesis/pre-training-multilingual-document-encoders/

# output directory of prepare_data.sh
DATA_DIR=/work/ogalolu/data/msmarco/

MAX_SEQ_LENGTH=4096
CUSTOM_MODEL=longformer
PRETRAINED_DIR=/work-ceph/ogalolu/models/long_models/labse-4096

python $PROJECT_HOME/ms_marco/adapter_retrieval_no_trainer.py \
    --output_dir /work-ceph/ogalolu/models/finetuned_models/ms_marco \
    --pretrained_dir $PRETRAINED_DIR \
    --max_seq_length $MAX_SEQ_LENGTH \
    --train_file $DATA_DIR/train_sbert.jsonl \
    --validation_file $DATA_DIR/dev_sbert.jsonl \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1\
    --learning_rate 2e-5 \
    --logging_steps 1000 \
    --num_train_epochs 1 \
    --warmup_steps 5000 \
    --weight_decay 1e-5 \
    --seed 42 \
    --preprocessing_num_workers 32 \
    --max_patience 7 \
    --custom_model $CUSTOM_MODEL

# ----unfreeze