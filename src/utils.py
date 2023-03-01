""" Utility functions to be for tokenization and handling arguments."""
import argparse
import json
import os
from collections import namedtuple
from typing import Union

from nltk import sent_tokenize

from datasets import arrow_dataset

MODEL_MAPPING = {
    "bert-base-multilingual-cased": "mbert",
    "xlm-roberta-base": "xlmr",
    "xlm-roberta-large": "xlmr-large",
    "sentence-transformers/LaBSE": "labse",
    "markussagen/xlm-roberta-longformer-base-4096": "markussagen-longformer",
    "roberta-base": "roberta",
}


def sliding_tokenize(
    article: str, tokenizer, args: argparse.Namespace, task=None
) -> tuple:
    """Tokenization function for sliding window approach."""
    sentences = tokenizer(
        article,
        max_length=args.max_seq_length,
        truncation=True,
        stride=args.stride,
        return_overflowing_tokens=True,
        padding=True,
    )

    return sentences["input_ids"], sentences["attention_mask"]


def tokenize_helper(
    article: str, tokenizer, args: argparse.Namespace, task: str = None
) -> tuple:
    """Tokenization function for sentence splitting approach."""
    if task is None:
        sentences = [
            tokenizer.encode(sentence, add_special_tokens=False)
            for sentence in sent_tokenize(article)
        ]
    # elif getattr(args, "task") == "retrieval":
    elif task == "retrieval":
        query, document = article.split("[SEP]")
        sentences = [query] + sent_tokenize(document)
        sentences = [
            tokenizer.encode(sentence, add_special_tokens=False)
            for sentence in sentences
        ]
    sentences = [sentence[: args.max_seq_length - 2] for sentence in sentences]
    sentences = [
        [tokenizer.convert_tokens_to_ids(tokenizer.cls_token)]
        + sentence
        + [tokenizer.convert_tokens_to_ids(tokenizer.sep_token)]
        for sentence in sentences
    ]

    sentence_lengths = [len(sentence) for sentence in sentences]
    mask = [[1] * sen_len for sen_len in sentence_lengths]

    return sentences, mask


def custom_tokenize(
    example: Union[arrow_dataset.Example, dict],
    tokenizer,
    args: argparse.Namespace,
    article_numbers: int,
    task: str = None,
    dual_encoder: bool = False,
) -> arrow_dataset.Example:
    """Controller function for tokenization."""
    if args.use_sliding_window_tokenization:
        func = sliding_tokenize
    else:
        func = tokenize_helper

    start = 1
    if dual_encoder:
        result = tokenizer(
            example["article_1"],
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
            return_token_type_ids=False,
        )
        example["article_1"] = result["input_ids"]
        example["mask_1"] = result["attention_mask"]
        start += 1

    for i in range(start, article_numbers + 1):
        example[f"article_{i}"], example[f"mask_{i}"] = func(
            example[f"article_{i}"], tokenizer, args, task
        )

    return example


def retrieval_preprocess(
    example: Union[arrow_dataset.Example, dict], tokenizer, args: argparse.Namespace
) -> dict:
    """Tokenization for retrieval tasks."""
    tmp = "[SEP]".join([example["query"].strip(), example["passage"].strip()])
    tmp_dict = {"article_1": tmp}
    tokenized = custom_tokenize(
        tmp_dict, tokenizer=tokenizer, args=args, article_numbers=1, task="retrieval"
    )
    return tokenized


def save_args(
    args: argparse.Namespace, args_path: str = None, pretrained: bool = False
):
    """Saves command line arguments to a json file.

    Args:
        args (argparse.Namespace): Arguments to be saved.
    """
    # TODO: has to be refactored.
    if not pretrained:
        if args_path is None:
            args_path = args.output_dir
        path = os.path.join(args_path, "args.json")
        with open(path, "w") as f:
            json.dump(args.__dict__, f, indent=2)
            print("args")
    else:
        path = os.path.join(args_path, "pretrained_args.json")
        with open(path, "w") as f:
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


def path_adder(
    args: argparse.Namespace,
    finetuning: bool = False,
    custom_model: str = None,
    c_args: argparse.Namespace = None,
) -> str:
    # TODO: has to be refactored.
    if not finetuning:
        i_path = (
            f"{MODEL_MAPPING[args.model_name_or_path]}_{'dummy_' if 'large' in args.train_file else ''}{args.upper_num_layers}"
            f"{'_frozen' if args.frozen else ''}{'_hard' if args.use_hard_negatives else ''}_"
            f"{args.per_device_train_batch_size}{'_sliding_window' if args.use_sliding_window_tokenization else ''}"
            f"_{args.lower_pooling}_{args.upper_pooling}_{args.learning_rate}_"
        )
    elif finetuning and custom_model == "hierarchical":
        i_path = (
            f"{MODEL_MAPPING[args.model_name_or_path]}_{'contrastive' if args.is_contrastive else ''}"
            # f"{'_init' if c_args.custom_from_scratch else ''}__"
        )
    elif finetuning and custom_model == "longformer":
        i_path = (
            f"{args.pretrained_dir.split('/')[-1]}_"  # {'_contrastive' if args.is_contrastive else ''}"
            # f"{'_init' if c_args.custom_from_scratch else ''}__"
        )
    else:
        i_path = (
            f"{MODEL_MAPPING[args.pretrained_dir]}_{args.max_seq_length}{'_frozen' if args.frozen else ''}"
            f"{'_sliding_window' if args.custom_model ==  'sliding_window' else ''}__"
        )
    return i_path


def preprocess_function(examples: arrow_dataset.Batch, tokenizer, max_seq_length: int):
    """Tokenization function for the AutoModels."""
    # result = tokenizer(examples["text"], padding=True, truncation=True)
    result = tokenizer(
        examples["text"], padding=True, truncation=True, max_length=max_seq_length
    )
    result["labels"] = examples["labels"]
    return result


def select_base(pretrained_path: str) -> str:
    """Returns the base model type for the given path."""
    for key, value in MODEL_MAPPING.items():
        if value in pretrained_path:
            return key
    raise ValueError("Respective path does not contain any supported model.")
