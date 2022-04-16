""" A script that contains model classes to be used in training."""
import os

import torch
from torch import nn
from transformers import BertPreTrainedModel, AutoConfig, RobertaPreTrainedModel, BertModel, XLMRobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder

from model_utils import cos_sim, get_extended_attention_mask, get_mean, ContrastiveModelOutput


class LowerXLMREncoder(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Modified
        config.model_type = "xlm-roberta"
        self.roberta = XLMRobertaModel(config)
        # TODO: change to post_init()
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        model_output = self.base_model(input_ids, attention_mask=attention_mask, token_type_ids=None) 
        output = model_output['last_hidden_state']  # (batch_size, words, hidden_size)
        return output


class LowerBertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        # TODO: change to post_init()
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        model_output = self.base_model(input_ids, attention_mask=attention_mask, token_type_ids=None)
        output = model_output['last_hidden_state']  # (batch_size, words, hidden_size)
        return output


class HiearchicalBaseModel(nn.Module):
    # To be used for sliding window approaches
    def __init__(self, args, tokenizer, **kwargs):
        super().__init__()
        self.lower_config = AutoConfig.from_pretrained(args.pretrained_dir)
        self.lower_model = self.lower_selector(args.pretrained_dir)
        self.lower_dropout = nn.Dropout(args.lower_dropout)

        # If True, freeze the lower encoder
        if args.frozen:
            self._freeze_lower()

        # Setting the pooling method of the upper encoder
        self.lower_pooling = getattr(args, "lower_pooling", "cls")

    def forward(self, input_ids, attention_mask=None, dcls=None, document_mask=None):
        input_ids = input_ids.permute(1, 0, 2)  # (sentences, batch_size, words)
        attention_mask = attention_mask.permute(1, 0, 2)
        lower_encoded = []

        for i_i, a_m in zip(input_ids, attention_mask):
            inter_output = self.lower_model(i_i, a_m)

            if self.lower_pooling == "mean":
                inter_output = get_mean(inter_output, a_m)  # (batch_size, hidden_size)
            elif self.lower_pooling == "cls":
                inter_output = inter_output[:, 0]  # (batch_size, hidden_size)

            inter_output = self.lower_dropout(inter_output)
            lower_encoded.append(inter_output)

        lower_output = torch.stack(lower_encoded)  # (sentences, batch_size, hidden_size)
        lower_output = lower_output.permute(1, 0, 2)  # (batch_size, sentences, hidden_size)

        # Mean Pooling
        # final_output = torch.mean(lower_output, 1)  # (batch_size, hidden_size)
        final_output = get_mean(lower_output, document_mask)  # (batch_size, hidden_size)
        
        return final_output

    def _freeze_lower(self):
        for param in self.lower_model.base_model.parameters():
            param.requires_grad = False

    def lower_selector(self, model_name):
        if self.lower_config.model_type == "xlm-roberta":
            lower_model = LowerXLMREncoder.from_pretrained(model_name)
        elif self.lower_config.model_type == "bert":
            lower_model = LowerBertEncoder.from_pretrained(model_name)
        else:
            raise NotImplementedError("Respective model type is not supported.")
        return lower_model


# TODO: Inherit from HiearchicalBaseModel, memory issues
class HiearchicalModel(nn.Module):
    # self.lower_model.base_model is a reference to self.lower_model.bert
    def __init__(self, args, tokenizer, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer

        self.lower_config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.lower_model = self.lower_selector(args.model_name_or_path)
        self.lower_dropout = nn.Dropout(args.lower_dropout)

        self.lower_model.resize_token_embeddings(len(self.tokenizer))

        self.upper_config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.update_config(args)

        # Initiliaze custom Bert model with updated config
        self.upper_embeddings = BertEmbeddings(self.upper_config)
        self.upper_encoder = BertEncoder(self.upper_config)

        # If True, freeze the lower encoder
        if args.frozen:
            self._freeze_lower()

        # If positional encoding will be used in upper encoder or not
        self.upper_positional = getattr(args, "upper_positional", True)

        # Setting the pooling method of the upper encoder
        self.upper_pooling = getattr(args, "upper_pooling")

        # Setting the pooling method of the upper encoder
        self.lower_pooling = getattr(args, "lower_pooling", "cls")

    def forward(self, input_ids, attention_mask=None, dcls=None, document_mask=None):
        input_ids = input_ids.permute(1, 0, 2)  # (sentences, batch_size, words)
        attention_mask = attention_mask.permute(1, 0, 2)
        lower_encoded = []

        for i_i, a_m in zip(input_ids, attention_mask):
            inter_output = self.lower_model(i_i, a_m)

            if self.lower_pooling == "mean":
                inter_output = get_mean(inter_output, a_m)  # (batch_size, hidden_size)
            elif self.lower_pooling == "cls":
                inter_output = inter_output[:, 0]  # (batch_size, hidden_size)

            inter_output = self.lower_dropout(inter_output)
            lower_encoded.append(inter_output)

        lower_output = torch.stack(lower_encoded)  # (sentences, batch_size, hidden_size)
        lower_output = lower_output.permute(1, 0, 2)  # (batch_size, sentences, hidden_size)

        dcls_out = self.upper_embeddings.word_embeddings(dcls)
        lower_output = torch.cat([dcls_out, lower_output], dim=1)

        if self.upper_positional:
            lower_output = self.upper_embeddings(inputs_embeds=lower_output)

        # Added upper encoder level attention mask
        extended_document_mask = get_extended_attention_mask(document_mask)
        encoder_output = self.upper_encoder(hidden_states=lower_output,
                                            attention_mask=extended_document_mask)  # (batch_size, sentences, hidden_size)

        upper_output = encoder_output["last_hidden_state"]
        if self.upper_pooling == "mean":
            final_output = get_mean(upper_output, document_mask)  # (batch_size, hidden_size)
        elif self.upper_pooling == "dcls":
            final_output = upper_output[:, 0]  # (batch_size, hidden_size)
        else:
            raise NotImplementedError("Respective pooling type is not supported.")

        return final_output

    def _freeze_lower(self):
        for param in self.lower_model.base_model.parameters():
            param.requires_grad = False

    def lower_selector(self, model_name):
        if self.lower_config.model_type == "xlm-roberta":
            lower_model = LowerXLMREncoder.from_pretrained(model_name)
        elif self.lower_config.model_type == "bert":
            lower_model = LowerBertEncoder.from_pretrained(model_name)
        else:
            raise NotImplementedError("Respective model type is not supported.")
        return lower_model

    def update_config(self, args):
        # self.config.hidden_size
        self.upper_config.num_attention_heads = args.upper_nhead
        self.upper_config.intermediate_size = args.upper_dim_feedforward
        self.upper_config.hidden_dropout_prob = args.upper_dropout
        self.upper_config.hidden_act = args.upper_activation
        self.upper_config.layer_norm_eps = args.upper_layer_norm_eps
        self.upper_config.num_hidden_layers = args.upper_num_layers
        self.upper_config.vocab_size = len(self.tokenizer)


class ContrastiveModel(nn.Module):
    def __init__(self, args, tokenizer, **kwargs):
        super().__init__()
        self.hierarchical_model = HiearchicalModel(args, tokenizer)
        self.scale = args.scale
        self.use_hard_negatives = args.use_hard_negatives
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        if args.similarity_fct == "cos_sim":
            self.similarity_fct = cos_sim
        else:
            raise NotImplementedError("Respective similarity function is not implemented.")

    def forward(self, article_1, mask_1, dcls_1, document_mask_1,
                article_2, mask_2, dcls_2, document_mask_2,
                article_3=None, mask_3=None, dcls_3=None, document_mask_3=None,
                article_4=None, mask_4=None, dcls_4=None, document_mask_4=None):
        output_1 = self.hierarchical_model(input_ids=article_1,
                                           attention_mask=mask_1,
                                           dcls=dcls_1,
                                           document_mask=document_mask_1
                                           )  # (batch_size, hidden_size)
        output_2 = self.hierarchical_model(input_ids=article_2,
                                           attention_mask=mask_2,
                                           dcls=dcls_2,
                                           document_mask=document_mask_2
                                           )  # (batch_size, hidden_size)
        if self.use_hard_negatives:
            output_3 = self.hierarchical_model(input_ids=article_3,
                                               attention_mask=mask_3,
                                               dcls=dcls_3,
                                               document_mask=document_mask_3
                                               )  # (batch_size, hidden_size)
            output_4 = self.hierarchical_model(input_ids=article_4,
                                               attention_mask=mask_4,
                                               dcls=dcls_4,
                                               document_mask=document_mask_4
                                               )  # (batch_size, hidden_size)

            scores_1 = self.similarity_fct(output_1, torch.cat([output_2, output_3])) * self.scale
            scores_2 = self.similarity_fct(output_2, torch.cat([output_1, output_4])) * self.scale
        else:
            scores_1 = self.similarity_fct(output_1, output_2) * self.scale
            scores_2 = self.similarity_fct(output_2, output_1) * self.scale

        labels = torch.tensor(range(len(scores_1)), dtype=torch.long, device=scores_1.device)

        return ContrastiveModelOutput(
            loss=self.cross_entropy_loss(scores_1, labels) + self.cross_entropy_loss(scores_2, labels),
            scores_1=scores_1,
            dist_1=torch.argmax(scores_1, dim=1),
            scores_2=scores_2,
            dist_2=torch.argmax(scores_2, dim=1),
        )

class HierarchicalClassificationModel(nn.Module):
    def __init__(self, c_args, args, tokenizer, num_labels, **kwargs):
        super().__init__()
        if c_args.custom_model == "hierarchical":
            self.hierarchical_model = HiearchicalModel(args, tokenizer)
            if not c_args.custom_from_scratch:
                self.hierarchical_model.load_state_dict(torch.load(os.path.join(c_args.pretrained_dir, 
                                                                                f"model_{c_args.pretrained_epoch}.pth"
                                                                                )))
        elif c_args.custom_model == "sliding_window":
            self.hierarchical_model = HiearchicalBaseModel(c_args, tokenizer)
        else:
            raise NotImplementedError("Respective model type is not supported.")

        self.num_labels = num_labels

        # TODO: change
        if c_args.dropout is not None:
            self.dropout = nn.Dropout(c_args.dropout)

        # For freezing/unfreezing the whole HierarchicalModel
        if c_args.unfreeze:
            self._unfreeze_model()
        elif c_args.freeze:
            self._freeze_model()

        self.classifier = nn.Linear(self.hierarchical_model.lower_config.hidden_size, self.num_labels)

    def forward(self, article_1, mask_1, dcls_1, document_mask_1, labels):
        output = self.hierarchical_model(input_ids=article_1,
                                           attention_mask=mask_1,
                                           dcls=dcls_1,
                                           document_mask=document_mask_1
                                           )  # (batch_size, hidden_size)

        output = self.dropout(output)
        logits = self.classifier(output)

        if self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

    def _unfreeze_model(self):
        for param in self.hierarchical_model.parameters():
            param.requires_grad = True

    def _freeze_model(self):
        for param in self.hierarchical_model.parameters():
            param.requires_grad = False
    