""" A data collator which enables dynamic padding for jagged lists."""
from dataclasses import dataclass

import torch


@dataclass
class CustomDataCollator:
    """ A data collator which can be used for dynamic padding, when each instance of a batch is a
    list of lists. Each sentence is a list and each document (instance of a batch) contains multiple
    sentences.
    """
    tokenizer: None
    article_numbers: int = None
    max_sentence_len: int = 128
    max_document_len: int = 32
    return_tensors: str = "pt"
    consider_dcls: bool = True
    target_device: str = None

    def __call__(self, features: list) -> dict:
        batch = {}

        for article_number in range(1, self.article_numbers + 1):
            batch_sentences = list()
            # For sentence level masking
            batch_masks = list()
            # For document level masking
            batch_document_masks = list()
            # For early initiliazed "DCLS" (Document level CLS)
            if self.consider_dcls:
                batch_dcls = list()

            sen_len_article = [len(sentence) for instance in features
                               for sentence in instance[f"article_{article_number}"]]
            sen_len_mask = [len(sentence) for instance in features for sentence in instance[f"mask_{article_number}"]]

            assert sen_len_article == sen_len_mask, (
                f"There is a mismatch for article_{article_number} and mask_{article_number}."
                )
            sen_len = min(self.max_sentence_len, max(sen_len_article))

            doc_len_article = [len(instance[f"mask_{article_number}"]) for instance in features]
            doc_len = min(self.max_document_len, max(doc_len_article))

            for feature in features:
                sentences, masks = self.pad_sentence(sen_len, feature, article_number)
                document_mask = [1] * len(masks)
                self.pad_document(sentences, masks, document_mask, doc_len)
                # Modified: For DCLS token
                if self.consider_dcls:
                    document_mask = [1] + document_mask

                batch_sentences.append(sentences)
                batch_masks.append(masks)
                batch_document_masks.append(document_mask)
                if self.consider_dcls:
                    # For each 
                    batch_dcls.append(self.tokenizer.encode("[DCLS]", add_special_tokens=False))

            batch[f"article_{article_number}"] = torch.tensor(batch_sentences, dtype=torch.int64)
            batch[f"mask_{article_number}"] = torch.tensor(batch_masks, dtype=torch.int64)            
            batch[f"document_mask_{article_number}"] = torch.tensor(batch_document_masks, dtype=torch.int64)
            if self.consider_dcls:
                batch[f"dcls_{article_number}"] = torch.tensor(batch_dcls, dtype=torch.int64)

            # Modified for classification task
            if "labels" in features[0]:
                batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.int64)
            # TODO: can be written better
            elif "label" in features[0]:
                batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.int64)

            # Modified for compatibility with CrossEncoder batching scheme
            if self.target_device is not None:
                batch = {k: t.to(self.target_device) for k, t in batch.items()}

        return batch

    def pad_sentence(self, sen_len: int, feature: dict, article_number: int) -> tuple():
        """Returns padded sentences so that within the batch, each sentence has the same number of words.

        Args:
            sen_len (list): Number of words that each sentence should have.
            feature (dict): Respective training instance of the batch.
            article_number (int): Article number.

        Returns:
           (tuple): Sentences and attention masks of the respective document after sentence-level padding.
        """
        sentences = [sentence + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)] * (sen_len - len(sentence))
                     for sentence in feature[f"article_{article_number}"]]
        masks = [sentence + [0] * (sen_len - len(sentence)) for sentence in feature[f"mask_{article_number}"]]
        return sentences, masks

    def pad_document(self, sentences: list, masks: list, document_mask: list, doc_len: int):
        """ Does document level padding so that within the batch, each document has the same
        number of sentences.

        Args:
            sentences (list): Sentences of the respective document.
            masks (list): Sentence level attention masks of the respective document.
            document_mask (list): Document level attention mask of the respective document
            doc_len (int): Number of sentences that each document of the batch should have.
        """
        # Pad documents while considering [DCLS] (document-level CLS) that will be preprended later
        if self.consider_dcls:
            doc_len -= 1

        mask_padding_array = [0 for i0 in range(len(masks[0]))]
        sentence_padding_array = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token) for i0 in range(len(sentences[0]))]

        if len(sentences) < doc_len:
            sentences += [sentence_padding_array for difference in range(doc_len - len(sentences))]
            masks += [mask_padding_array for difference in range(doc_len - len(masks))]
            # TODO: check
            document_mask.extend([0] * (doc_len - len(document_mask)))
        elif len(sentences) > doc_len:
            sentences[:] = sentences[: doc_len]
            masks[:] = masks[: doc_len]
            document_mask[:] = document_mask[: doc_len]
