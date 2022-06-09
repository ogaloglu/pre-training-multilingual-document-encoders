# Preranking files
PRERANKING_DIR=/work/ogalolu/data/preranking
# GPU
GPU=7

# Directory containing a trained monoBERT model

# MODEL_DIR=/work-ceph/ogalolu/models/finetuned_models/ms_marco/labse_contrastive2022_05_22-12_35_31/
# CUSTOM_MODEL=hierarchical
# PRETRAINED_EPOCH=5000
# python clef/monobert_eval.py --model_dir $MODEL_DIR --prerank_dir $PRERANKING_DIR --mode clir --gpu $GPU --custom_model $CUSTOM_MODEL --pretrained_epoch $PRETRAINED_EPOCH

MODEL_DIR=/work-ceph/ogalolu/models/finetuned_models/clef/labse_contrastive2022_06_05-17_56_41/
CUSTOM_MODEL=hierarchical
python clef/monobert_eval.py --model_dir $MODEL_DIR --prerank_dir $PRERANKING_DIR --mode clir --gpu $GPU --custom_model $CUSTOM_MODEL --dual_encoder