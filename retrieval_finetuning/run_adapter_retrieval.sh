PROJECT_HOME=/home/ogalolu/thesis/pre-training-multilingual-document-encoders/

# output directory of prepare_data.sh
DATA_DIR=/work/ogalolu/data/msmarco/

MAX_SEQ_LENGTH=4096
CUSTOM_MODEL=longformer
PRETRAINED_DIR=/work-ceph/ogalolu/models/long_models/labse-4096

python $PROJECT_HOME/retrieval_finetuning/adapter_retrieval.py \
    --overwrite_output_dir \
    --output_dir /work-ceph/ogalolu/models/finetuned_models/ms_marco \
    --cache_dir /work-ceph/ogalolu/models/finetuned_models/ms_marco/.cache/ \
    --pretrained_dir $PRETRAINED_DIR \
    --max_seq_length $MAX_SEQ_LENGTH \
    --custom_model $CUSTOM_MODEL \
    --train_file $DATA_DIR/train_sbert.jsonl \
    --validation_file $DATA_DIR/dev_sbert.jsonl \
    --do_train \
    --do_eval \
    --learning_rate 3e-5 \
    --eval_steps 5000 \
    --save_steps 25000 \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --num_train_epochs 1 \
    --warmup_steps 2500 \
    --log_level info \
    --weight_decay 1e-2 \
    --per_device_eval_batch_size 64 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --max_eval_samples 20000 \
    --resume_from_checkpoint /work-ceph/ogalolu/models/finetuned_models/ms_marco/labse-4096_2022_05_30-18_57_13/checkpoint-50000

