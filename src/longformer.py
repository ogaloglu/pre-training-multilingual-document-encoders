""" A script that contains model classes to be used for multilingual Longformer."""
import math

import torch
from torch.nn import functional as F
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention


class LongModelSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        """
        :class:`LongformerSelfAttention` expects `len(hidden_states)` to be multiple of `attention_window`. Padding to
        `attention_window` happens in :meth:`LongformerModel.forward` to avoid redoing the padding on each layer.

        Modified LongformerSelfAttention layer to work with RoBERTa models. They have different set of arguments
        compared to original Longformer attention: RoBERTa is passing some additional arguments and arguments required
        by LongformerSelfAttention are missing so we have to add it manually.

        The `attention_mask` is changed in :meth:`LongformerModel.forward` from 0, 1, 2 to:

            * -10000: no attention
            * 0: local attention
            * +10000: global attention
        Taken from https://github.com/allenai/longformer/blob/2fb7b11df5319801742b23a407b8abb302d15750/scripts/convert_model_to_long.ipynb
        """
        hidden_states = hidden_states.transpose(0, 1)

        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        ## Lines below has been changed

        attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)

        # is index masked or global attention
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = any(is_index_global_attn.flatten())

        ## End of modification

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(
            seq_len, batch_size, self.num_heads, self.head_dim
        ).transpose(0, 1)
        key_vectors = key_vectors.view(
            seq_len, batch_size, self.num_heads, self.head_dim
        ).transpose(0, 1)

        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )

        # values to pad for attention probs

        ## Lines below has been changed

        remove_from_windowed_attention_mask = (
            (attention_mask != 0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        )

        ## End of modification

        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = remove_from_windowed_attention_mask.type_as(
            query_vectors
        ).masked_fill(remove_from_windowed_attention_mask, -10000.0)
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_mask.new_ones(size=float_mask.size()),
            float_mask,
            self.one_sided_attn_window_size,
        )

        # pad local attention probs
        attn_scores += diagonal_mask

        assert list(attn_scores.size()) == [
            batch_size,
            seq_len,
            self.num_heads,
            self.one_sided_attn_window_size * 2 + 1,
        ], f"local_attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"

        # compute local attention probs from global attention keys and contact over window dim
        if is_global_attn:
            # compute global attn indices required through out forward fn
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self._get_global_attn_indices(is_index_global_attn)
            # calculate global attn probs from global key

            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )
            # concat to local_attn_probs
            # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)

            # free memory
            del global_key_attn_scores

        attn_probs = F.softmax(
            attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        attn_probs = torch.masked_fill(
            attn_probs, is_index_masked[:, :, None, None], 0.0
        )
        attn_probs = attn_probs.type_as(attn_scores)

        # free memory
        del attn_scores

        # apply dropout
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        value_vectors = value_vectors.view(
            seq_len, batch_size, self.num_heads, self.head_dim
        ).transpose(0, 1)

        # compute local attention output with global attention value and add
        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        assert attn_output.size() == (
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ), "Unexpected size"
        attn_output = (
            attn_output.transpose(0, 1)
            .reshape(seq_len, batch_size, embed_dim)
            .contiguous()
        )

        # compute value for global attention and overwrite to attention output
        # TODO: remove the redundant computation
        if is_global_attn:
            (
                global_attn_output,
                global_attn_probs,
            ) = self._compute_global_attn_output_from_hidden(
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
            )

            # get only non zero global attn output
            nonzero_global_attn_output = global_attn_output[
                is_local_index_global_attn_nonzero[0],
                :,
                is_local_index_global_attn_nonzero[1],
            ]

            # overwrite values with global attention
            attn_output[
                is_index_global_attn_nonzero[::-1]
            ] = nonzero_global_attn_output.view(
                len(is_local_index_global_attn_nonzero[0]), -1
            )
            # The attention weights for tokens with global attention are
            # just filler values, they were never used to compute the output.
            # Fill with 0 now, the correct values are in 'global_attn_probs'.
            attn_probs[is_index_global_attn_nonzero] = 0

        outputs = (attn_output.transpose(0, 1),)

        if output_attentions:
            outputs += (attn_probs,)

        return (
            outputs + (global_attn_probs,)
            if (is_global_attn and output_attentions)
            else outputs
        )


def get_attention_injected_model(
    pretrained_model, attention_model=LongModelSelfAttention
):
    class LongModelForMaskedLM(pretrained_model):
        def __init__(self, config):
            super().__init__(config)
            for i, layer in enumerate(self.base_model.encoder.layer):
                layer.attention.self = attention_model(config, layer_id=i)

    return LongModelForMaskedLM
