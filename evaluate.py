#!/usr/bin/env python3
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import torch
import datasets
from datasets import load_from_disk, load_metric
from torch.utils.data import DataLoader

import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from huggingface_hub import Repository
from transformers import (
    AutoTokenizer,
    set_seed,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)

from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

from utils import custom_tokenize, load_args, path_adder, preprocess_function, sliding_tokenize
from data_collator import CustomDataCollator
from models import HierarchicalClassificationModel

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune the hierarchical model on a text classification task")
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        # Modified
        "--max_seq_length",
        type=int,
        default=None,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        # Modified
        "--finetuned_dir",
        type=str,
        help="Path to the output directory of finetuning.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    # Modified:
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--max_document_length",
        type=int,
        default=None,
        required=True,
        help="The maximum number of sentences each document can have. Documents are either truncated or"
             "padded if their length is different.",
    )
    parser.add_argument(
        "--custom_model",
        type=str,
        help="If a custom model is to be used, the model type has to be specified.",
        default=None,
        choices=["hierarchical", "sliding_window"]
    )
    args = parser.parse_args()

    # Sanity checks
    if args.test_file is None:
        raise ValueError("Need testing file.")

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    # Modified: classification arguments
    args = parse_args()

    # TODO: change the logic
    # Argments from pretraining
    if args.custom_model == "hierarchical":
        pretrained_args = load_args(os.path.join(args.finetuned_dir, "pretrained_args.json"))
    finetuned_args = load_args(os.path.join(args.finetuned_dir, "args.json"))

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            # Modified: output_dir is concatanated with datetime and command line arguments are also saved
            # TODO: refactor
            if args.custom_model == "hierarchical":
                inter_path = path_adder(pretrained_args, finetuning=True, custom_model=args.custom_model)
            else:
                inter_path = path_adder(finetuned_args, finetuning=True)
            inter_path += datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            args.output_dir = os.path.join(args.output_dir, inter_path)
            os.makedirs(args.output_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        # Modified
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "loginfo.log")),
            logging.StreamHandler()
        ]
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    # Modified:
    test_dataset = load_from_disk(args.test_file)
    # Labels
    label_list = test_dataset.unique("labels")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.finetuned_dir, use_fast=not args.use_slow_tokenizer)

    if args.custom_model in ("hierarchical", "sliding_window"):
        model = HierarchicalClassificationModel(c_args=finetuned_args,
                                                args=None if args.custom_model == "sliding_window" else pretrained_args,
                                                tokenizer=tokenizer,
                                                num_labels=num_labels)
        model.load_state_dict(torch.load(os.path.join(args.finetuned_dir, "model.pth")))
    else:
        config = AutoConfig.from_pretrained(args.finetuned_dir, num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.finetuned_dir,
            config=config,
        )

    if args.custom_model == "hierarchical":
        with accelerator.main_process_first():
            # Modified
            test_dataset = test_dataset.rename_column("text", "article_1")
            ARTICLE_NUMBERS = 1
            test_dataset = test_dataset.map(
                custom_tokenize,
                fn_kwargs={"tokenizer": tokenizer, "args": args, "article_numbers": ARTICLE_NUMBERS},
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
    elif args.custom_model == "sliding_window":
         with accelerator.main_process_first():
            test_dataset = test_dataset.map(
                sliding_tokenize,
                fn_kwargs={"tokenizer": tokenizer, "args": args},
                num_proc=args.preprocessing_num_workers,
                remove_columns=test_dataset.column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )       
    else:
        with accelerator.main_process_first():
            test_dataset = test_dataset.map(
                preprocess_function,
                fn_kwargs={"tokenizer": tokenizer},
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=test_dataset.column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )

    # Modified

    # Log a few random samples from the training set:
    for index in random.sample(range(len(test_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {test_dataset[index]}.")

    if args.custom_model in ("hierarchical", "sliding_window"):
        ARTICLE_NUMBERS = 1
        data_collator = CustomDataCollator(tokenizer=tokenizer,
                                           max_sentence_len=pretrained_args.max_seq_length if args.max_seq_length is None else args.max_seq_length,
                                           max_document_len=pretrained_args.max_document_length if args.max_document_length is None else args.max_document_length,
                                           article_numbers=ARTICLE_NUMBERS,
                                           consider_dcls=True if args.custom_model == "hierarchical" else False)
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    # Modified: only accuracy.
    # Get the metric function
    metric = load_metric("accuracy")

    model.eval()
    for batch in test_dataloader:
        # Modified for Hierarchical Classification Model
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch["labels"]),
        )

    eval_metric = metric.compute()
    logger.info(f"final accuracy: {eval_metric}")


if __name__ == "__main__":
    main()
