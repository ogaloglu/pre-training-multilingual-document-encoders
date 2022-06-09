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
from utils import custom_tokenize, load_args, save_args, path_adder, preprocess_function, MODEL_MAPPING, select_base, retrieval_preprocess, tokenize_helper
from model_utils import freeze_base, copy_proj_layers, pretrained_masked_model_selector, pretrained_model_selector, pretrained_sequence_model_selector, cos_sim, ContrastiveModelOutput
from data_collator import CustomDataCollator
from models import HierarchicalClassificationModel, HiearchicalModel, HiearchicalBaseModel
from longformer import get_attention_injected_model


logger = logging.getLogger(__name__)


class DualModelExtended(nn.Module):
    def __init__(self, c_args, args, tokenizer, article_numbers=2, **kwargs):
        super().__init__()
        if c_args.custom_model == "hierarchical":

            if getattr(c_args, "upper_pooling", None) is not None:
                args = args._replace(upper_pooling=c_args.upper_pooling)
            if getattr(c_args, "lower_pooling", None) is not None:       
                args = args._replace(lower_pooling=c_args.lower_pooling) 
            self.hierarchical_model = HiearchicalModel(args, tokenizer)

            cpt = torch.load(os.path.join(c_args.pretrained_dir, "model.pth"))
            # Modified: If the model is saved differently, the following hack will be used
            if "hierarchical_model" in "".join(cpt.keys()):
                cpt = {k[19:]: v for k, v in cpt.items() if "hierarchical_model" in k}                                               
                self.hierarchical_model.load_state_dict(cpt)
                
        elif c_args.custom_model == "sliding_window":
            self.hierarchical_model = HiearchicalBaseModel(c_args, tokenizer)
        else:
            raise NotImplementedError("Respective model type is not supported.")        
        
        self.lower_model = self.hierarchical_model.lower_model
        self.lower_pooling = getattr(args, "lower_pooling", "cls")
        self.article_numbers = article_numbers

        if args.similarity_fct == "cos_sim":
            self.similarity_fct = cos_sim
        else:
            raise NotImplementedError("Respective similarity function is not implemented.")        


    # Modified: Remove document related variables from article_1
    def forward(self, article_1, mask_1,  
                article_2, mask_2, dcls_2, document_mask_2,
                **kwargs):

        inter_output = self.lower_model(article_1[0].unsqueeze(0), mask_1[0].unsqueeze(0))

        if self.lower_pooling == "mean":
            output_1 = get_mean(inter_output, a_m)  # (batch_size=1, hidden_size)
        elif self.lower_pooling == "cls":
            output_1 = inter_output[:, 0]  # (batch_size=1, hidden_size)

        output_2 = self.hierarchical_model(input_ids=article_2,
                                           attention_mask=mask_2,
                                           dcls=dcls_2,
                                           document_mask=document_mask_2
                                           )  # (batch_size, hidden_size)

        scores_1 = self.similarity_fct(output_1, output_2)
  
        return ContrastiveModelOutput(
            scores_1=scores_1[0],
            dist_1=torch.argmax(scores_1[0]),
        )


class DualModelEvaluator():
    def __init__(self, model_name:str, device:str = None, hierarchical_args = None, article_numbers:int = 2):
        super().__init__()
           
        # Argments from pretraining
        if hierarchical_args is not None:
            self.hierarchical_args = hierarchical_args     
            self.custom_model = hierarchical_args.custom_model
            self.hierarchical_args.pretrained_dir = self.hierarchical_args.model_dir
        else:
            self.custom_model = None
        
        if self.custom_model == "hierarchical":
            self.pretrained_args = load_args(os.path.join(model_name, "pretrained_args.json"))
        elif self.custom_model == "sliding_window":
            self.pretrained_args.use_sliding_window_tokenization = True

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
            self.model = DualModelExtended(c_args=self.hierarchical_args,
                                           args=None if self.custom_model == "sliding_window" else self.pretrained_args,
                                           tokenizer=self.tokenizer)
        else:
            pass
                    
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)
       
        # Modified:
        if self.custom_model in ("hierarchical", "sliding_window"):

            self.data_collator = CustomDataCollator(tokenizer=self.tokenizer,
                                               dual_encoder=True,
                                               max_sentence_len=self.pretrained_args.max_seq_length,
                                               max_document_len=self.pretrained_args.max_document_length,
                                               article_numbers=2,
                                               consider_dcls=True if hierarchical_args.custom_model == "hierarchical" else False,
                                               target_device=self._target_device)
        else:
            raise NotImplementedError("Respective models are not supported")   

    def predict(self, sentences: List[List[str]],
                batch_size: int = 128,
                show_progress_bar: bool = None,
                num_workers: int = 0,
                convert_to_numpy: bool = True,
                convert_to_tensor: bool = False
                ):
            """
            Performs predicts with the CrossEncoder on the given sentence pairs.
            :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
            :param batch_size: Batch size for encoding
            :param show_progress_bar: Output progress bar
            :param num_workers: Number of workers for tokenization
            :param convert_to_numpy: Convert the output to a numpy matrix.
            :param convert_to_tensor:  Conver the output to a tensor.
            :return: Predictions for the passed sentence pairs
            """
            input_was_string = False
            if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
                sentences = [sentences]
                input_was_string = True

            if self.custom_model not in ("hierarchical", "sliding_window"):
                pass
                # inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, num_workers=num_workers, shuffle=False)
            else:
                custom_batched_sentences = self.dual_tokenize(sentences)
                inp_dataloader = DataLoader(custom_batched_sentences, batch_size=batch_size, collate_fn=self.data_collator, num_workers=num_workers, shuffle=False, )

            if show_progress_bar is None:
                show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

            iterator = inp_dataloader
            if show_progress_bar:
                iterator = tqdm(inp_dataloader, desc="Batches")

            pred_scores = []
            self.model.eval()
            self.model.to(self._target_device)
            with torch.no_grad():
                for features in iterator:
                    # Modified: remove return_dict=True
                    output = self.model(**features)
                    pred_scores.extend(output.scores_1)

            if convert_to_tensor:
                pred_scores = torch.stack(pred_scores)
            elif convert_to_numpy:
                pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

            if input_was_string:
                pred_scores = pred_scores[0]

            return pred_scores

    def dual_tokenize(self, examples: list[list[str]]) -> list:
        """."""
        batch = []
        if self.pretrained_args.use_sliding_window_tokenization:
            func = sliding_tokenize
        else:
            func = tokenize_helper
            
        for i, example in enumerate(examples):
            tmp_dict = {}
            result = self.tokenizer(example[0], padding=True, truncation=True, max_length=self.pretrained_args.max_seq_length,
                                    return_token_type_ids=False)    
            tmp_dict["article_1"] = result["input_ids"]
            tmp_dict["mask_1"] = result["attention_mask"]
            
            tmp_dict[f"article_2"], tmp_dict[f"mask_2"] = func(example[1], self.tokenizer, self.pretrained_args)
            batch.append(tmp_dict)

        return batch


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
            self.pretrained_args = load_args(os.path.join(model_name, "pretrained_args.json"))
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

            self.model.load_state_dict(torch.load(os.path.join(self.hierarchical_args.pretrained_dir, 
                                                               f"model_{self.hierarchical_args.pretrained_epoch}.pth")))
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
               batch_size: int = 128,
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
            inp_dataloader = DataLoader(custom_batched_sentences, batch_size=batch_size, collate_fn=self.data_collator, num_workers=num_workers, shuffle=False, )

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
