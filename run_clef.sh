# Preranking files
PRERANKING_DIR=/work/ogalolu/data/preranking
# GPU
GPU=3

# Directory containing a trained monoBERT model
# MODEL_DIR=/work-ceph/ogalolu/models/monobert/checkpoint-25000

# MODEL_DIR=/work-ceph/ogalolu/models/trained_models/labse_2_frozen_hard_128_cls_mean_0.0005_2022_04_19-12_31_23
# CUSTOM_MODEL=hierarchical

MODEL_DIR=/work-ceph/ogalolu/models/long_models/labse-4096
CUSTOM_MODEL=longformer
PRETRAINED_EPOCH=2

python clef/monobert_eval.py --model_dir $MODEL_DIR --prerank_dir $PRERANKING_DIR --mode clir --gpu $GPU --custom_model $CUSTOM_MODEL --pretrained_epoch $PRETRAINED_EPOCH