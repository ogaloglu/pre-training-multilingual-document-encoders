""" A script that contains model classes to be used in training."""
import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel, AutoConfig

from utils import cos_sim


class LowerEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.post_init()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        model_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = model_output['last_hidden_state'][:, 0, :]  # (batch_size, hidden_size)
        return output


class HiearchicalModel(nn.Module):
    def __init__(self, args, tokenizer, **kwargs):
        super().__init__()
        # TODO: from pretrained or config
        self.lower_config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.lower_model = LowerEncoder.from_pretrained(args.model_name_or_path)

        self.tokenizer = tokenizer
        self.lower_model.resize_token_embeddings(len(self.tokenizer))
        # TODO: Add positional embeddings
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.lower_config.hidden_dimension,
                                                        nhead=args.upper_nhead,
                                                        dim_feedforward=args.upper_dim_feedforward,
                                                        dropout=args.upper_dropout,
                                                        activation=args.upper_activation,
                                                        layer_norm_eps=args.upper_layer_norm_eps,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                                         num_layers=args.upper_num_layers)

        if args.frozen:
            self._freeze_lower()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        input_ids = input_ids.permute(1, 0, 2)  # (sentences, batch_size, words)
        attention_mask = attention_mask.permute(1, 0, 2)
        lower_encoded = []

        for i_i, a_m in zip(input_ids, attention_mask):
            lower_encoded.append(self.lower_model(i_i, a_m))

        lower_output = torch.stack(lower_encoded)  # (sentences, batch_size, hidden_size)
        lower_output = lower_output.permute(1, 0, 2)  # (batch_size, sentences, hidden_size)

        # Modified: Document level [CLS] tokens are prepended to the documents
        dcls_tokens = self.tokenizer(["[DCLS]"] * lower_output.shape[0],
                                     add_special_tokens=False,
                                     return_tensors="pt",
                                     return_attention_mask=False,
                                     return_token_type_ids=False)
        # TODO: Maybe create random tensors instead?
        dcls_tokens.to(lower_output.device)
        dcls_out = self.lower_model.bert.embeddings(dcls_tokens["input_ids"])
        lower_output = torch.cat([dcls_out, lower_output], dim=1)

        upper_output = self.transformer_encoder(lower_output)  # (batch_size, sentences, hidden_size)
        upper_output = upper_output[:, 0, :]  # (batch_size, hidden_size)

        return upper_output

    def _freeze_lower(self):
        for param in self.lower_model.bert.parameters():
            param.requires_grad = False


class ContrastiveModel(nn.Module):
    def __init__(self, args, tokenizer, **kwargs):
        super().__init__()
        self.hierarchical_model = HiearchicalModel(args, tokenizer)
        self.scale = args.scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        if args.similarity_fct == "cos_sim":
            self.similarity_fct = cos_sim
        else:
            raise NotImplementedError("Respective similarity function is not implemented.")

    def forward(self, article_1, mask_1, article_2, mask_2, article_3, mask_3, article_4, mask_4):
        # output_1 = self.hierarchical_model(input_ids=article_1,
        #                                    attention_mask=mask_1)  # (batch_size, hidden_size)
        # output_2 = self.hierarchical_model(input_ids=article_2,
        #                                    attention_mask=mask_2)  # (batch_size, hidden_size)

        # scores = self.similarity_fct(output_1, output_2) * self.scale
        # labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)

        output_1 = self.hierarchical_model(input_ids=article_1,
                                           attention_mask=mask_1)  # (batch_size, hidden_size)
        output_2 = self.hierarchical_model(input_ids=article_2,
                                           attention_mask=mask_2)  # (batch_size, hidden_size)
        output_3 = self.hierarchical_model(input_ids=article_3,
                                           attention_mask=mask_3)  # (batch_size, hidden_size)
        output_4 = self.hierarchical_model(input_ids=article_4,
                                           attention_mask=mask_4)  # (batch_size, hidden_size)

        scores_1 = self.similarity_fct(output_1, output_2, output_3) * self.scale
        scores_2 = self.similarity_fct(output_2, output_1, output_4) * self.scale

        labels = torch.tensor(range(len(scores_1)), dtype=torch.long, device=scores_1.device)
        return self.cross_entropy_loss(scores_1, labels) + self.cross_entropy_loss(scores_2, labels)
