PROJECT_HOME=/home/ogalolu/thesis/pre-training-multilingual-document-encoders/

# Specify output directory for downloading and storing prepared data files.
OUTPUT_DIR=/work/ogalolu/data/msmarco/
mkdir -p $OUTPUT_DIR

python $PROJECT_HOME/ms_marco/prepare_msmarco.py --output_dir $OUTPUT_DIR