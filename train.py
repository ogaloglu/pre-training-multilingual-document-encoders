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
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
from random import randint
from datetime import datetime
from pathlib import Path

import datasets
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from huggingface_hub import Repository
from tokenizers import Tokenizer
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    BertTokenizerFast,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

from models import ContrastiveModel
from data_collator import CustomDataCollator
from utils import custom_tokenize, save_args, path_adder

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Pretrain multilingual document encoder")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=10,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        # Modified for saving arguments
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        required=True,
        help="The maximum total input seq. length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")

    # Modified: Additional parameters for ContrastiveModel
    parser.add_argument(
        "--scale",
        type=int,
        default=20,
        help="Output of similarity function is multiplied by scale value.",
    )
    parser.add_argument(
        "--similarity_fct",
        type=str,
        default="cos_sim",
        # TODO: add choices for similarity function
        help="Similarity function between sentence embeddings. By default, cos_sim."
             "Can also be set to dot product (and then set scale to 1).",
    )
    parser.add_argument(
        "--tokenizer_file",
        type=str,
        default=None,
        help="Path for the trained tokenizer file.",
    )
    parser.add_argument(
        "--max_document_length",
        type=int,
        default=None,
        required=True,
        help="The maximum number of sentences each document can have. Documents are either truncated"
             "or padded if their length is different.",
    )
    # parser.add_argument(
    #     "--upper_hidden_dimension",
    #     type=int,
    #     default=None,
    #     required=True,
    #     help="The number of expected features in the input of the upper level encoder.",
    # )
    parser.add_argument(
        "--upper_nhead",
        type=int,
        default=None,
        required=True,
        help="The number of heads in the multiheadattention models of the upper level encoder.",
    )
    parser.add_argument(
        "--upper_dim_feedforward",
        type=int,
        default=2048,
        help="The dimension of the feedforward network model of the upper level encoder.",
    )
    parser.add_argument(
        "--lower_dropout",
        type=float,
        default=0.1,
        help="The dropout value of lower level encoder.",
    )
    parser.add_argument(
        "--upper_dropout",
        type=float,
        default=0.1,
        help="The dropout value of upper level encoder.",
    )
    parser.add_argument(
        "--upper_activation",
        type=str,
        default="relu",
        choices=["relu", "gelu"],
        help="The the activation function of the intermediate layer of upper level encoder.",
    )
    parser.add_argument(
        "--upper_layer_norm_eps",
        type=float,
        default=1e-5,
        help="The eps value in layer normalization components of upper level encoder.",
    )
    parser.add_argument(
        "--upper_num_layers",
        type=int,
        default=None,
        required=True,
        help="The number of sub-encoder-layers in the encoder of the upper level encoder.",
    )
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="Either the lower level encoder is frozen or not."
    )
    parser.add_argument(
        "--upper_positional",
        action="store_true",
        help="Either positional embeddings are used for the upper encoder or not."
    )
    parser.add_argument(
        "--use_hard_negatives",
        action="store_true",
        help="Either include hard negatives or not."
    )
    parser.add_argument(
        "--is_contrastive",
        action="store_true",
        help="Either the pretraining mode is contrastive or not."
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Frequency of logging mini-batch loss.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="If True, validation loss is checked in each logging step"
             "(rather than only after each epoch)."
    )
    parser.add_argument(
        "--upper_pooling",
        type=str,
        required=True,
        choices=["mean", "dcls"],
        help="Determines to pooling method of the upper encoder."
    )
    args = parser.parse_args()
    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_arguments()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # Modified: for handling unsued parameters
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Modified: change the order
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
            inter_path = path_adder(args)
            inter_path += datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            args.output_dir = os.path.join(args.output_dir, inter_path)
            os.makedirs(args.output_dir, exist_ok=True)
            save_args(args)

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

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        # Modified for loading dataset from disk
        raw_datasets = load_from_disk(args.train_file)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if args.validation_file is None:
            raw_datasets = raw_datasets.train_test_split(test_size=args.validation_split_percentage, seed=args.seed)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # TODO: move lower init here
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # Modified for custom tokenizer file
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    elif args.tokenizer_file:
        base_tokenizer = Tokenizer.from_file(args.tokenizer_file)
        tokenizer = BertTokenizerFast(tokenizer_object=base_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Modified: Model is set to be ContrastiveModel
    # Modified: Add token as document level [CLS]
    tokenizer.add_tokens(["[DCLS]"])
    model = ContrastiveModel(args, tokenizer)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # Modified: Tokenization pipeline

    if not args.is_contrastive:
        article_numbers = 1
    elif args.use_hard_negatives:
        article_numbers = 4
    else:
        article_numbers = 2
        # remove hard negatives
        raw_datasets = raw_datasets.remove_columns(['article_3', 'article_4'])
    logger.info("article number is: %s ", article_numbers)

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            custom_tokenize,
            fn_kwargs={"tokenizer": tokenizer, "args": args, "article_numbers": article_numbers},
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator
    # Modified: CustomDataCollator for documents
    data_collator = CustomDataCollator(tokenizer=tokenizer,
                                       max_sentence_len=args.max_seq_length,
                                       max_document_len=args.max_document_length,
                                       article_numbers=article_numbers)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    # Modified for checkpoint saving:
    min_loss = float("inf")

    for epoch in range(args.num_train_epochs):
        model.train()
        # Modified for running_loss
        running_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Modified for ContrastiveModel
            loss = model(**batch)
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                # Modified
                running_loss += loss.item()
            if step % args.logging_steps == args.logging_steps - 1:
                # TODO change
                if args.inspect:
                    model.eval()
                    losses = []
                    for _, batch in enumerate(eval_dataloader):
                        with torch.no_grad():
                            loss = model(**batch)
                        losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))

                    losses = torch.cat(losses)
                    losses = losses[: len(eval_dataset)]
                    total_loss = torch.mean(losses)
                    logger.info(f"epoch: {epoch}, step {step+1}:, train_loss: {running_loss/args.logging_steps}, val_loss: {total_loss}")
                    model.train()
                else:
                    logger.info(f"epoch: {epoch}, step {step+1}:, loss: {running_loss/args.logging_steps}")
                running_loss = 0.0
            if completed_steps >= args.max_train_steps:
                break
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                # Modified for Contrastive Model
                loss = model(**batch)
            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            total_loss = torch.mean(losses)
        except OverflowError:
            total_loss = float("inf")

        logger.info(f"epoch {epoch}: loss: {total_loss}")

        # Modified: changed for checkpoint saving
        if total_loss < min_loss:
            min_loss = total_loss
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            # Modified
            accelerator.save(obj=unwrapped_model.hierarchical_model.state_dict(),
                             f=args.output_dir + "/model.pth")
            logger.info(f"model after spoch {epoch} is saved")
            # TODO: change later to save only one time
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                # repo.push_to_hub(
                #     commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                # )

    # if args.output_dir is not None:
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     # Modified
    #     accelerator.save(obj=unwrapped_model.hierarchical_model.state_dict(),
    #                      f=args.output_dir + "/model.pth")
    #     if accelerator.is_main_process:
    #         tokenizer.save_pretrained(args.output_dir)
    #         if args.push_to_hub:
    #             repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)


if __name__ == "__main__":
    main()
