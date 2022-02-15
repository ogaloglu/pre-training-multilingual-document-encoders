# Examples of tokenizer training

All of the tokenizers are trained on Wikipedia editions of Italian, German, English and French. The number of vocabulary is set to be 35.000 for each cases.

There are two parameters namely `min_frequency` and `limit_alphabet` that have substantial affect on the tokenizer's vocabulary [WordPieceTokenizer](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizers.trainers.WordPieceTrainer).

Different configurations are saved as "tokenizer_[`min_frequency`]_[`limit_alphabet`]".