""" A script that contains model classes to be used in training."""
import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel

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
    def __init__(self, args, **kwargs):
        super().__init__()
        # TODO: from pretrained or config
        self.lower_model = LowerEncoder.from_pretrained(args.pretrained_model_path)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.upper_hidden_dimension,
                                                        nhead=args.upper_nhead,
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

        # TODO: add document level [CLS]

        lower_output = torch.stack(lower_encoded)  # (sentences, batch_size, hidden_size)
        lower_output = lower_output.permute(1, 0, 2)  # (batch_size, sentences, hidden_size)
        upper_output = self.transformer_encoder(lower_output)  # (batch_size, sentences, hidden_size)
        upper_output = upper_output[:, 0, :]  # (batch_size, hidden_size)

        return upper_output

    def _freeze_lower(self):
        for param in self.lower_model.bert.parameters():
            param.requires_grad = False


class ContrastiveModel(nn.Module):
    def __init__(self, args, scale: float = 20.0, similarity_fct=cos_sim, **kwargs):
        super().__init__()
        self.hierarchical_model = HiearchicalModel(args)
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        article_1, mask_1, article_2, mask_2 = batch

        output_1 = self.hierarchical_model(input_ids=article_1,
                                           attention_mask=mask_1)  # (batch_size, hidden_size)
        output_2 = self.hierarchical_model(input_ids=article_2,
                                           attention_mask=mask_2)  # (batch_size, hidden_size)

        scores = self.similarity_fct(output_1, output_2) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        return self.cross_entropy_loss(scores, labels)
