""" Utility functions to be used for modeling."""
import copy
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertModel,
    PreTrainedModel,
    RobertaForMaskedLM,
    RobertaForSequenceClassification,
    RobertaModel,
    XLMRobertaForMaskedLM,
    XLMRobertaForSequenceClassification,
    XLMRobertaModel,
)
from transformers.file_utils import ModelOutput
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention


def cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])

    Taken from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def freeze_base(model):
    """For freezing base of the auto_models."""
    for param in model.base_model.parameters():
        param.requires_grad = False


def get_extended_attention_mask(attention_mask: Tensor) -> Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = extended_attention_mask.to(
        dtype=extended_attention_mask.dtype
    )  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def get_mean(upper_output: Tensor, document_mask: Tensor) -> Tensor:
    """For freezing base of the auto_models."""
    # document_mask has to be expanded to the shape of upper_output
    input_mask_expanded = (
        document_mask.unsqueeze(-1).expand(upper_output.size()).float()
    )

    # Rather than taking simply mean, we have to consider the padded documents.
    # Therefore, effective length of each document will also change accordingly.
    sum_embeddings = torch.sum(upper_output * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    output = sum_embeddings / sum_mask
    return output


@dataclass
class ContrastiveModelOutput(ModelOutput):
    """
    Class for ContrastiveModel's outputs, with loss and the result of used similarity function.
    Args:
        loss (torch.Tensor):

        scores_1 (torch.Tensor):

        dist_1 (torch.Tensor):

        scores_2 (torch.Tensor):

        dist_2 (torch.Tensor):

    """

    loss: torch.Tensor = None
    scores_1: torch.Tensor = None
    dist_1: torch.Tensor = None
    scores_2: torch.Tensor = None
    dist_2: torch.Tensor = None


@dataclass
class ContrastiveModelRepresentationOutput(ModelOutput):
    """
    Class for ContrastiveModel's outputs for document and query representations.
    Args:
        output_1 (torch.Tensor):

        output_2 (torch.Tensor):

    """

    output_1: torch.Tensor = None
    output_2: torch.Tensor = None


def pretrained_masked_model_selector(seed_model: str) -> PreTrainedModel:
    if seed_model == "xlm-roberta-base":
        PRETRAINED_MODEL = XLMRobertaForMaskedLM
    elif seed_model == "roberta-base":
        PRETRAINED_MODEL = RobertaForMaskedLM
    elif seed_model in ("sentence-transformers/LaBSE", "bert-base-multilingual-cased"):
        PRETRAINED_MODEL = BertForMaskedLM
    else:
        raise NotImplementedError("Other models are not supported")
    return PRETRAINED_MODEL


def pretrained_model_selector(seed_model: str) -> PreTrainedModel:
    if seed_model == "xlm-roberta-base":
        PRETRAINED_MODEL = XLMRobertaModel
    elif seed_model == "roberta-base":
        PRETRAINED_MODEL = RobertaModel
    elif seed_model in ("sentence-transformers/LaBSE", "bert-base-multilingual-cased"):
        PRETRAINED_MODEL = BertModel
    else:
        raise NotImplementedError("Other models are not supported")
    return PRETRAINED_MODEL


def pretrained_sequence_model_selector(seed_model: str) -> PreTrainedModel:
    if seed_model == "xlm-roberta-base":
        PRETRAINED_MODEL = XLMRobertaForSequenceClassification
    elif seed_model == "roberta-base":
        PRETRAINED_MODEL = RobertaForSequenceClassification
    elif seed_model in ("sentence-transformers/LaBSE", "bert-base-multilingual-cased"):
        PRETRAINED_MODEL = BertForSequenceClassification
    else:
        raise NotImplementedError("Other models are not supported")
    return PRETRAINED_MODEL


def create_long_model(seed_model, save_model_to, attention_window, max_pos):
    """Modified: all roberta -> base_model."""

    # MODIFIED:
    model = AutoModelForMaskedLM.from_pretrained(seed_model)
    tokenizer = AutoTokenizer.from_pretrained(
        seed_model, model_max_length=max_pos, use_fast=True
    )
    config = model.config

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs["model_max_length"] = max_pos
    (
        current_max_pos,
        embed_size,
    ) = model.base_model.embeddings.position_embeddings.weight.shape
    if seed_model in ("xlm-roberta-base", "roberta-base"):
        max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
        assert max_pos > current_max_pos
    config.max_position_embeddings = max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.base_model.embeddings.position_embeddings.weight.new_empty(
        max_pos, embed_size
    )
    # copy position embeddings over and over to initialize the new position embeddings

    # MODIFIED:
    if seed_model in ("xlm-roberta-base", "roberta-base"):
        k = 2
    else:
        k = 0
    step = current_max_pos - k
    while k < max_pos - 1:
        if seed_model in ("xlm-roberta-base", "roberta-base"):
            new_pos_embed[
                k : (k + step)
            ] = model.base_model.embeddings.position_embeddings.weight[k:]
        else:
            new_pos_embed[
                k : (k + step)
            ] = model.base_model.embeddings.position_embeddings.weight[:]
        # END OF MODIFICATION
        k += step
    model.base_model.embeddings.position_embeddings.weight.data = new_pos_embed
    model.base_model.embeddings.position_ids.data = torch.tensor(
        [i for i in range(max_pos)]
    ).reshape(1, max_pos)

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.base_model.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn

    print(f"saving model to {save_model_to}")
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer


def copy_proj_layers(model):
    for i, layer in enumerate(model.base_model.encoder.layer):
        layer.attention.self.query_global = copy.deepcopy(layer.attention.self.query)
        layer.attention.self.key_global = copy.deepcopy(layer.attention.self.key)
        layer.attention.self.value_global = copy.deepcopy(layer.attention.self.value)
    return model
