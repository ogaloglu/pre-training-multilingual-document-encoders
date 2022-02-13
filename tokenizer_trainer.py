""" A script to train tokenizer from scratch on the given settings and languages."""
import argparse
import logging
import sys
import os

import datasets

from datasets import load_dataset, concatenate_datasets
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

logging.root.handlers = []
logging.basicConfig(level="INFO", format = '%(asctime)s:%(levelname)s: %(message)s' ,stream = sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a tokenizer on the given languages from scratch"
        )
    parser.add_argument(
        "--languages",
        default=["en", "de", "fr", "it"],
        nargs="+",
        type=str,
        required=True,
        help="The languages (of respective Wiki Corpora) to be included in the training of tokenizer.",
    )
    parser.add_argument(
        "--out",
        default="./",
        type=str,
        help="Path to the output directory, where the files will be saved.",
    )
    parser.add_argument(
        "--name", 
        default="wordpiece_tokenizer", 
        type=str, 
        help="The name of the output vocab files."
    )
    parser.add_argument(
        "--vocab_size",
        default=35000,
        type=int,
        help="Vocabulary size of the tokenizer.",
    )
    parser.add_argument(
        "--min_frequency",
        default=2,
        type=int,
        help="Minimum frequency a pair should have in order to be merged.",
    )
    parser.add_argument(
        "--limit_alphabet",
        default=1000,
        type=int,
        help="Maximum different characters to keep in the alphabet.",
    )

    args = parser.parse_args()
    return args


def get_dataset(langs: list(str)) -> datasets.Dataset:
    """Loads the Wikipedia datasets of the given languages and returns a concataned version of them.

    Args:
        langs (list): List of language codes

    Returns:
        datasets.Dataset: Concatanated dataset
    """
    wiki_datasets = [load_dataset("wikipedia", f'20200501.{i}', split="train") for i in langs]
    wiki_datasets = [i.remove_columns('title') for i in wiki_datasets]
    logger.info(wiki_datasets)

    dataset = concatenate_datasets(wiki_datasets)
    logger.info(dataset)

    return dataset


def main():
    args = parse_args()

    dataset = get_dataset(args.languages)

    # Build an iterator over the dataset
    def get_training_corpus(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]

    # Initialize an empty tokenizer
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

    # Set normalzier and pre_tokenizer
    tokenizer.normalizer = normalizers.BertNormalizer(
        lowercase=False, strip_accents=False, clean_text=True
        )
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    # 
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[DCLS]"]
    trainer = trainers.WordPieceTrainer(vocab_size=args.vocab_size, 
                                        special_tokens=special_tokens, 
                                        show_progress=True,
                                        min_frequency=args.min_frequency,
                                        limit_alphabet=args.limit_alphabet)

    # Train
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer, length=len(dataset))

    #  Set decoder and enable padding
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer.enable_padding(pad_token="[PAD]")

    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    # Set post_processer
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id),
        ],
    )

    file_name = os.path.join(args.out, f"{args.name}.json")
    tokenizer.save(file_name)

if __name__ == "__main__":
    main()