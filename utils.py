""" Utility functions to be for tokenization and handling arguments."""
import argparse
import os
import json
from collections import namedtuple

from nltk import sent_tokenize
from datasets import arrow_dataset
from transformers.modeling_outputs import SequenceClassifierOutput


MODEL_MAPPING = {
    "bert-base-multilingual-cased": "mbert",
    "xlm-roberta-base": "xlmr",
    "sentence-transformers/LaBSE": "labse"
}


def custom_tokenize(example: arrow_dataset.Example, tokenizer,
                    args: argparse.Namespace, article_numbers: int) -> arrow_dataset.Example:

    def tokenize_helper(article: str, tokenizer, args: argparse.Namespace) -> tuple:
        sentences = [tokenizer.encode(sentence, add_special_tokens=False) for sentence in sent_tokenize(article)]
        sentences = [sentence[:args.max_seq_length - 2] for sentence in sentences]
        sentences = [[tokenizer.convert_tokens_to_ids(tokenizer.cls_token)] + sentence +
                     [tokenizer.convert_tokens_to_ids(tokenizer.sep_token)] for sentence in sentences]

        sentence_lengths = [len(sentence) for sentence in sentences]
        mask = [[1]*sen_len for sen_len in sentence_lengths]

        return sentences, mask

    for i in range(1, article_numbers + 1):
        example[f"article_{i}"], example[f"mask_{i}"] = tokenize_helper(example[f"article_{i}"], tokenizer, args)

    return example


def save_args(args: argparse.Namespace, args_path: str = None, pretrained: bool = False):
    """Saves command line arguments to a json file.

    Args:
        args (argparse.Namespace): Arguments to be saved.
    """
    # TODO: has to be refactored.
    if not pretrained:
        if args_path is None:
            args_path = args.output_dir
        path = os.path.join(args_path, "args.json")
        with open(path, 'w') as f:
            json.dump(args.__dict__, f, indent=2)
            print("args")
    else:
        path = os.path.join(args_path, "pretrained_args.json")
        with open(path, 'w') as f:
            args = args._asdict()
            json.dump(args, f, indent=2)
            print("p_args")


def load_args(args_path: str) -> namedtuple:
    """Loads arguments of the pretrained model from the given json file.

    Args:
        args_path (str): Path of the arguments that are used for the pretrained.

    Returns:
        namedtuple: argparse.Namespace like object to store arguments.
    """
    with open(args_path) as f:
        args = json.load(f)

    args = namedtuple("Args", args.keys())(*args.values())
    return args


def path_adder(args: argparse.Namespace, finetuning: bool = False, custom_model: str = None, c_args: argparse.Namespace = None) -> str:
    # TODO: has to be refactored.
    if not finetuning:
        i_path = f"{MODEL_MAPPING[args.model_name_or_path]}_{args.upper_num_layers}{'_frozen' if args.frozen else ''}{'_hard' if args.use_hard_negatives else ''}_{args.num_train_epochs}__"
    elif finetuning and custom_model == "hierarchical":
        i_path = f"{MODEL_MAPPING[args.model_name_or_path]}{'_contrastive' if args.is_contrastive else ''}__"
        # i_path = f"{MODEL_MAPPING[args.model_name_or_path]}{'_contrastive' if args.is_contrastive else ''}{'_init' if c_args.custom_from_scratch else ''}__"
    else:
        i_path = f"{MODEL_MAPPING[args.pretrained_dir]}{'_sliding_window' if args.custom_model ==  'sliding_window' else ''}__"
    return i_path


def preprocess_function(examples: arrow_dataset.Batch, tokenizer):
    # Tokenization function for the AutoModels
    # https://huggingface.co/docs/transformers/preprocessing
    # TODO: make if statement
    #result = tokenizer(examples["text"], padding=True, truncation=True)
    result = tokenizer(examples["text"], padding=True, truncation=True, max_length=128)
    result["labels"] = examples["labels"]
    return result


def sliding_tokenize(example: arrow_dataset.Example, args: argparse.Namespace, tokenizer):
    # Tokenization function for sliding window models
    sentences = tokenizer(example["text"],
                          max_length=args.max_seq_length,
                          truncation=True,
                          stride=34,  # TODO: add to args
                          return_overflowing_tokens=True,
                          padding=True)
    return {
        "article_1": sentences["input_ids"],
        "mask_1": sentences["attention_mask"],
        "labels": example["labels"]
    }
