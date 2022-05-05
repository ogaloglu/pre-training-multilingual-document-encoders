from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import logging
import os
import sys
from typing import Dict, Type, Callable, List
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from sentence_transformers import CrossEncoder
from dataclasses import dataclass

# TODO: CHANGE!!!!!! (...)
# import util
# from ...utils import custom_tokenize, load_args, save_args, path_adder, preprocess_function
# from ...model_utils import freeze_base
# from ...data_collator import CustomDataCollator
# from ...models import HierarchicalClassificationModel

sys.path.insert(0, '/home/ogalolu/thesis/pre-training-multilingual-document-encoders/clef/cross_encoder')
import util
sys.path.insert(0, '/home/ogalolu/thesis/pre-training-multilingual-document-encoders')
from utils import custom_tokenize, load_args, save_args, path_adder, preprocess_function, MODEL_MAPPING, select_base, retrieval_preprocess
from model_utils import freeze_base, copy_proj_layers, pretrained_masked_model_selector, pretrained_model_selector, pretrained_sequence_model_selector
from data_collator import CustomDataCollator
from models import HierarchicalClassificationModel
from longformer import get_attention_injected_model


logger = logging.getLogger(__name__)


@dataclass
class CustomConfig:
    num_labels: int = None


class CrossEncoder(CrossEncoder):
    """
    __init__ and predict methods are overwritten to be able to use hierarchical models, which are different than Huggingface AutoModels
    and which have different tokenization + data collation scheme.    
    """
    # Modified: Added "hiearchical_args" for hierarchical model related arguments. 
    def __init__(self, model_name:str, num_labels:int = None, max_length:int = 512, device:str = None, tokenizer_args:Dict = {},
                  automodel_args:Dict = {}, default_activation_function = None, hierarchical_args = None):
        
        # Modified: "classifier_trained" is removed and additional model loading options are added. 
        if num_labels is None:
            num_labels = 1
           
        # Argments from pretraining
        if hierarchical_args is not None:
            self.hierarchical_args = hierarchical_args     
            self.custom_model = hierarchical_args.custom_model
            self.hierarchical_args.pretrained_dir = self.hierarchical_args.model_dir
        else:
            self.custom_model = None
        
        if self.custom_model == "hierarchical":
            self.pretrained_args = load_args(os.path.join(model_name, "args.json"))
            # self.hierarchical_args.use_sliding_window_tokenization = getattr(self.pretrained_args , "use_sliding_window_tokenization", False)
        elif self.custom_model == "sliding_window":
            # self.hierarchical_args.use_sliding_window_tokenization = True
            self.pretrained_args.use_sliding_window_tokenization = True

        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)

        if self.custom_model == "longformer":
            self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            max_length=4096,
            padding=True,
            # padding="max_length",
            truncation=True,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.custom_model in ("hierarchical", "sliding_window"):
            self.model = HierarchicalClassificationModel(c_args=self.hierarchical_args,
                                                    args=None if self.custom_model == "sliding_window" else self.pretrained_args,
                                                    tokenizer=self.tokenizer,
                                                    num_labels=num_labels)
            self.config = CustomConfig(num_labels=num_labels)
        elif self.custom_model == "longformer":

            self.config = AutoConfig.from_pretrained(model_name)
            self.config.num_labels = num_labels
 
            psm = pretrained_sequence_model_selector(select_base(model_name))
            self.model = get_attention_injected_model(psm)
            self.model = self.model.from_pretrained(
                model_name,  # /checkpoint-14500
                max_length=4096,
                num_labels=num_labels
            )
        else:
            self.config = AutoConfig.from_pretrained(model_name)
            self.config.num_labels = num_labels
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=self.config,
            )
            
        # End of modification
        self.max_length = max_length
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

        if default_activation_function is not None:
            self.default_activation_function = default_activation_function
            try:
                self.config.sbert_ce_default_activation_function = util.fullname(self.default_activation_function)
            except Exception as e:
                logger.warning("Was not able to update config about the default_activation_function: {}".format(str(e)) )
        # TODO: change
        elif self.config is None:
            self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()
        elif hasattr(self.config, 'sbert_ce_default_activation_function') and self.config.sbert_ce_default_activation_function is not None:
            self.default_activation_function = util.import_from_string(self.config.sbert_ce_default_activation_function)()
        else:
            self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()
        
        # Modified:
        if self.custom_model in ("hierarchical", "sliding_window"):

            self.data_collator = CustomDataCollator(tokenizer=self.tokenizer,
                                               max_sentence_len=self.pretrained_args.max_seq_length,
                                               max_document_len=self.pretrained_args.max_document_length,
                                               article_numbers=1,
                                               consider_dcls=True if self.custom_model == "hierarchical" else False,
                                               target_device=self._target_device)

    def predict(self, sentences: List[List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               num_workers: int = 0,
               activation_fct = None,
               apply_softmax = False,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False
               ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.
        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor:  Conver the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        input_was_string = False
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if self.custom_model not in ("hierarchical", "sliding_window"):
            inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, num_workers=num_workers, shuffle=False)
        else:
            custom_batched_sentences = self.custom_batching(sentences)
            inp_dataloader = DataLoader(custom_batched_sentences, batch_size=32, collate_fn=self.data_collator, num_workers=num_workers, shuffle=False, )

        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        self.model.eval()
        self.model.to(self._target_device)
        with torch.no_grad():
            for features in iterator:
                # Modified: remove return_dict=True
                model_predictions = self.model(**features)
                logits = activation_fct(model_predictions.logits)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores

    def custom_batching(self, batch: list[list[str]]):
        """Shapes the input into the format so that the tokenizers of hierarchical models can process it."""
        processed_batch = list()
        for example in batch:
            tmp = "[SEP]".join([text.strip() for text in example])
            tmp_dict = {"article_1": tmp}
            # self.pretrained_args.task = "retrieval"
            tokenized = custom_tokenize(tmp_dict, self.tokenizer, self.pretrained_args, article_numbers=1, task="retrieval")
            processed_batch.append(tokenized)
        return processed_batch

    def smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        # tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        tokenized = self.tokenizer(*texts, padding=True, truncation=True, pad_to_multiple_of=512, return_tensors="pt", max_length=self.max_length)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized
