# A General-Purpose Multilingual Document Encoder

This project is the codebase for the paper titled "A General-Purpose Multilingual Document Encoder".

Preprint: https://arxiv.org/abs/2305.07016

## Abstract
Massively multilingual pretrained transformers
(MMTs) have tremendously pushed the state of
the art on multilingual NLP and cross-lingual
transfer of NLP models in particular. While a
large body of work leveraged MMTs to mine
parallel data and induce bilingual document
embeddings, much less effort has been devoted
to training general-purpose (massively) multi-
lingual document encoder that can be used for
both supervised and unsupervised document-
level tasks. In this work, we pretrain a massively multilingual document encoder as a hierarchical transformer model (HMDE) in which
a shallow document transformer contextualizes
sentence representations produced by a state-
of-the-art pretrained multilingual sentence encoder. We leverage Wikipedia as a readily
available source of comparable documents for
creating training data, and train HMDE by
means of a cross-lingual contrastive objective,
further exploiting the category hierarchy of
Wikipedia for creation of difficult negatives.
We evaluate the effectiveness of HMDE in two
arguably most common and prominent cross-
lingual document-level tasks: (1) cross-lingual
transfer for topical document classification and
(2) cross-lingual document retrieval. HMDE
is significantly more effective than (i) aggregations of segment-based representations and
(ii) multilingual Longformer. Crucially, owing
to its massively multilingual lower transformer,
HMDE successfully generalizes to languages
unseen in document-level pretraining. We publicly release our code and models
## Model Overview
 <img src="https://user-images.githubusercontent.com/33498883/241818654-ee3289db-17d9-4476-a4d7-928f001870ed.png" alt= "" width="400" height="600">

## Installation Instructions
* **Project setup**:
```bash
git clone https://github.com/ogaloglu/pre-training-multilingual-document-encoders.git
cd pre-training-multilingual-document-encoders
conda create --name mhm
conda activate mhm
pip install .
```
Note: requirements.txt will be added

* **Accelerate configuration**:
    * ```bash
        accelerate config
        ```
    * Then, make configurations regarding the number of processes and mixed-precision. 4 GPUs are utilized during the unfrozen contrastive pre-training, whereas 3 are utilized during the frozen one.



## Data
The links for the datasets will be shared
* **Pre-training data**
* **Evaluation data**

## Models
The links for the pre-trained models will be shared

```bash
models
├── trained_models
├── finetuned_models
│   ├── mldoc
│   └── clef
└── long_models
```

## Pre-Training

* **Multilingual Hierarchical Model (MHM)** \
To pre-train MHM, the following script is used:
```bash
bash scripts/run_train.sh
```
*Key parameters*:
```bash
    --output_dir                        # Path of the resulting model
    --is_contrastive                    # Either the pretraining mode is contrastive or not
    --per_device_train_batch_size       # Batch size per device
    --gradient_accumulation_steps       # Number of updates steps to accumulate
    --num_train_epochs                  # Total number of training epochs
    --learning_rate                     # Learning rate
    --upper_nhead                       # Number of heads in the multiheadattention models of the upper-level encoder
    --upper_num_layers                  # Number of layer in the upper-level encoder
    --upper_activation                  # One of: relu, gelu
    --upper_dim_feedforward             # Dimension of the feedforward network model of the upper-level encoder
    --scale                             # Value to be multiplied with the output of similarity function
    --use_hard_negatives                # Either include hard negatives or not
    --upper_positional                  # Either positional embeddings are used for the upper encoder or not
    --max_seq_length                    # Maximum total input seq. length after tokenization
    --max_document_length               # Maximum number of sentences each document can have
    --lower_pooling                     # One of: mean, cls
    --upper_pooling                     # One of: mean, dcls
    --model_name_or_path                # One of: xlm-roberta-base, sentence-transformers/LaBSE
    --use_sliding_window_tokenization   # Either use sliding window segmentation or not
    --stride                            # Length of the stride, when sliding window approach is used
    --frozen                            # Either the lower-level encoder is frozen or not
```

* **Multilingual Longformer** \
To train multilungual Longfromer: 
```bash
python src/run_longformer.py
```
*Key parameters*:
```bash
--output_dir                            # Path of the resulting model
--per_device_train_batch_size           # Batch size per device
--gradient_accumulation_steps           # Number of updates steps to accumulate
--learning_rate                         # Learning rate
--seed_model                            # One of: xlm-roberta-base, sentence-transformers/LaBSE
--data_path                             # Path of X-WIKI dataset
```

## Fine-Tuning
* **Multilingual Document Classification Dataset (MLDOC)** \
Suggested approach: notebooks/finetuning.ipynb \
Alternative approach:
```bash
bash scripts/run_finetune.sh
```
*Key parameters*:
```bash
--custom_model                          # One of: longformer, hierarchical
--pretrained_dir                        # Path of the pre-trained model
--output_dir                            # Path of the resulting model
--learning_rate                         # Learning rate
```

* **Cross-lingual Evaluation Forum (CLEF) 2003**
    * Bi-encoder: 
    ```bash
    bash src/retrieval_finetuning/run_dual_encoder_finetuning.sh
    ```
    *Key parameters*:
    ```bash
    --max_seq_length                    # Maximum sequence length
    --learning_rate                     # Learning rate
    --pretrained_dir                    # Path of model to be evaluated
    --output_dir                        # Path of the resulting model
    --pretrained_epoch                  # Checkpoint of model to be evaluated
    --article_numbers   	            # Maximum number or negative articles that will be within a forward-pass (To fit in a GPU)
    ```

    * Cross-encoder:
    ```bash
    bash src/retrieval_finetuning/run_adapter_retrieval_no_trainer.sh
    ```
    *Key parameters*:
    ```bash
    --max_seq_length                    # Maximum sequence length, e.g. 128 for hierarchical model or 4096 for Longformer
    --custom_model                      # One of: longformer, hierarchical
    --pretrained_dir                    # Path of the pre-trained model
    --output_dir                        # Path of the resulting model
    --learning_rate                     # Learning rate
    --custom_model                      # One of: longformer, hierarchical
    ```


## Evaluation
* **MLDOC** \
Suggested approach: notebooks/evaluate.ipynb \
Alternative approach: 
```bash
bash scripts/run_evaluate.sh
```
*Key parameters*:
```bash
--finetuned_dir                         # Path of fine-tuned model
--output_dir                            # Path of log file
```
* **CLEF  2003**
```bash
bash scripts/run_clef_dual_encoder.sh
```
*Key parameters*:
```bash
--custom_model                          # One of: longformer, hierarchical
--pretrained_dir                        # Path of model to be evaluated
--pretrained_epoch                      # Checkpoint of model to be evaluated
--dual_encoder                          # To be used for a bi-encoder 
```

## Citation
If you use this repository, please consider citing our paper:
```bibtex
@misc{galoğlu2023generalpurpose,
      title={A General-Purpose Multilingual Document Encoder}, 
      author={Onur Galoğlu and Robert Litschko and Goran Glavaš},
      year={2023},
      eprint={2305.07016},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
