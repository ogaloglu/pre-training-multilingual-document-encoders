""" A script to train a multilingual longformer."""
import logging
import math
import os
from dataclasses import dataclass, field

from datasets import load_from_disk
from longformer import get_attention_injected_model
from model_utils import (
    copy_proj_layers,
    create_long_model,
    pretrained_masked_model_selector,
    pretrained_model_selector,
)
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TextDataset,
    Trainer,
    TrainingArguments,
)
from utils import MODEL_MAPPING

logger = logging.getLogger(__name__)


def tokenize_function(examples, max_seq_length, tokenizer):
    # Remove empty lines
    examples["article"] = [
        line for line in examples["article"] if len(line) > 0 and not line.isspace()
    ]
    return tokenizer(
        examples["article"],
        truncation=True,
        max_length=max_seq_length,
    )


def pretrain_and_evaluate(
    args,
    model,
    tokenizer,
    eval_only,
    model_path,
    max_seq_length,
    num_proc,
    val_raw_dataset,
    train_raw_dataset,
):

    val_dataset = val_raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=val_raw_dataset.column_names,
        fn_kwargs={"max_seq_length": max_seq_length, "tokenizer": tokenizer},
    )

    if eval_only:
        train_dataset = val_dataset
    else:
        train_dataset = train_raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=train_raw_dataset.column_names,
            fn_kwargs={"max_seq_length": max_seq_length, "tokenizer": tokenizer},
        )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15, pad_to_multiple_of=512
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    eval_loss = trainer.evaluate()
    eval_loss = eval_loss["eval_loss"]
    logger.info(f"Initial eval bpc: {eval_loss/math.log(2)}")

    if not eval_only:
        trainer.train(model_path=model_path)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss["eval_loss"]
        logger.info(f"Eval bpc after pretraining: {eval_loss/math.log(2)}")


def main():
    @dataclass
    class ModelArgs:
        attention_window: int = field(
            default=512, metadata={"help": "Size of attention window"}
        )
        max_pos: int = field(default=4096, metadata={"help": "Maximum position"})
        seed_model: str = field(
            default="xlm-roberta-base", metadata={"help": "Seed model to convert long"}
        )

    parser = HfArgumentParser(
        (
            TrainingArguments,
            ModelArgs,
        )
    )

    # TODO: change
    dataset_name = "extended"
    output_dir = (
        "/ceph/ogalolu/models/long_models_extended"
        if dataset_name == "extended"
        else "ceph/ogalolu/models/long_models"
    )
    s_m = "xlm-roberta-base"  # sentence-transformers/LaBSE, xlm-roberta-base
    m_p = 4096
    data_path = f"/work/ogalolu/datasets/longformer_{dataset_name}_updated"
    training_args, model_args = parser.parse_args_into_dataclasses(
        look_for_args_file=False,
        args=[
            "--output_dir",
            f"{output_dir}/{MODEL_MAPPING[s_m]}-{m_p}",
            "--warmup_steps",
            "500",
            "--learning_rate",
            "0.00003",
            "--weight_decay",
            "0.01",
            "--adam_epsilon",
            "1e-6",
            #'--max_steps', '3000',
            "--num_train_epochs",
            "1",
            "--logging_steps",
            "10000",
            "--prediction_loss_only",
            "True",
            "--save_steps",
            "10000",
            "--max_grad_norm",
            "5.0",
            "--per_device_eval_batch_size",
            "1",  # 8
            "--per_device_train_batch_size",
            "1",  # total: 0.064
            "--gradient_accumulation_steps",
            "32",  # 64
            "--evaluation_strategy",
            "steps",
            "--do_train",
            "--do_eval",
            "--fp16",
            "True",
            "--dataloader_num_workers",
            "4",
            "--dataloader_pin_memory",
            "True",
            "--seed",
            "42",
        ],
    )
    dataset = load_from_disk(data_path)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        # Modified
        handlers=[
            logging.FileHandler(os.path.join(training_args.output_dir, "loginfo.log")),
            logging.StreamHandler(),
        ],
    )

    logger.info(f"Loading the model from {training_args.output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)
    PRETRAINED_MODEL = pretrained_masked_model_selector(model_args.seed_model)
    model = get_attention_injected_model(PRETRAINED_MODEL)
    model = model.from_pretrained(training_args.output_dir)

    logger.info(
        f"Pretraining {MODEL_MAPPING[model_args.seed_model]}-{model_args.max_pos} ... "
    )
    pretrain_and_evaluate(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        eval_only=False,
        model_path=training_args.output_dir,
        max_seq_length=model_args.max_pos,
        num_proc=32,
        val_raw_dataset=dataset["test"],
        train_raw_dataset=dataset["train"],
    )

    logger.info(f"Copying local projection layers into global projection layers ... ")
    model = copy_proj_layers(model)
    logger.info(f"Saving model to {training_args.output_dir}")
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
