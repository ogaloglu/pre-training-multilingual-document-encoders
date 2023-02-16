# Preranking files
PRERANKING_DIR=/work/ogalolu/data/preranking
# GPU
GPU=3

MODEL_DIR=/work-ceph/ogalolu/models/long_models/labse-4096
CUSTOM_MODEL=longformer
PRETRAINED_EPOCH=/checkpoint-5000

python clef/monobert_eval.py --model_dir $MODEL_DIR --prerank_dir $PRERANKING_DIR --mode clir --gpu $GPU --custom_model $CUSTOM_MODEL --pretrained_epoch $PRETRAINED_EPOCH