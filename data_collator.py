
from dataclasses import dataclass

import torch


@dataclass
class CustomDataCollator:
    """ A data collator which can be used for dynamic padding, when each instance of a batch is a 
    list of lists. Each sentence is a list and each document (instance of a batch) contains multiple 
    sentences.
    """
    tokenizer: None
    max_sentence_len: int = 128
    max_document_len: int = 32
    return_tensors: str = "pt"
    

    def __call__(self, features: list) -> dict:
        batch = {}

        # TODO: make article number dynamic
        for article_number in range(1, 3):
            batch_sentences = list()
            batch_masks = list()
            
            sen_len_article = [len(sentence) for instance in features for sentence in instance[f"article_{article_number}"]]
            sen_len_mask = [len(sentence) for instance in features for sentence in instance[f"mask_{article_number}"]]
            
            assert sen_len_article == sen_len_mask, (
                f"There is a mismatch for article_{article_number} and mask_{article_number}."
                )
            
            sen_len = min(self.max_sentence_len, max(sen_len_article))
            
            doc_len_article = [len(instance[f"mask_{article_number}"]) for instance in features]
            doc_len = min(self.max_document_len, max(doc_len_article))

            for id in range(len(features)):
                sentences, masks = self.pad_sentence(sen_len, features[id], article_number)
                self.pad_document(sentences, masks, doc_len)

                batch_sentences.append(sentences)
                batch_masks.append(masks)
      
            # TODO: decide on dtype for tensor, torch.int/torch.long?
            batch[f"article_1": torch.tensor(batch_sentences, dtype=torch.int64)]
            batch[f"mask_1": torch.tensor(batch_masks, dtype=torch.int64)]
        
        return batch

    def pad_sentence(self, sen_len: int, feature: dict, i: int) -> tuple():
        sentences = [sentence + [self.tokenizer.convert_tokens_to_ids("[PAD]")] * (sen_len - len(sentence))  for sentence in feature[f"article_{i}"]]
        # TODO: check for attention_mask ID
        masks = [sentence + [0] * (sen_len - len(sentence))  for sentence in feature[f"mask_{i}"]]

        return sentences, masks

    def pad_document(self, sentences: list, masks: list, doc_len: int):
        mask_padding_array = [0 for i0 in range(len(masks[0]))]
        sentence_padding_array = [self.tokenizer.convert_tokens_to_ids("[PAD]") for i0 in range(len(sentences[0]))]

        if len(sentences) < doc_len:
            sentences += [sentence_padding_array for difference in range(doc_len - len(sentences))]
            masks += [mask_padding_array for difference in range(doc_len - len(masks))]
        elif len(sentences) > doc_len:
            sentences = sentences[: doc_len]
            masks = masks[: doc_len]