""" Utility functions to be used for modeling."""
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers.file_utils import ModelOutput


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
    # For freezing base of the auto_models.
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
    extended_attention_mask = extended_attention_mask.to(dtype=extended_attention_mask.dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def get_mean(upper_output: Tensor, document_mask: Tensor) -> Tensor:

    # print(upper_output)
    # document_mask has to be expanded to the shape of upper_output
    input_mask_expanded = document_mask.unsqueeze(-1).expand(upper_output.size()).float()

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

    loss: torch.Tensor
    scores_1: OptionL[torch.Tensor] = None
    dist_1: Optional[torch.Tensor] = None
    scores_2: Optional[torch.Tensor] = None
    dist_2: Optional[torch.Tensor] = None
