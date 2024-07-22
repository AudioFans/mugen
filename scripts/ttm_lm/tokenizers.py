from typing import Any


from audidata.tokenizers import (BaseTokenizer, ConcatTokenizer,
                                 SpecialTokenizer)


class GTZANLabelTokenizer(BaseTokenizer):
    def __init__(self, words: int):
        super().__init__(words=words)


class CodecTokenizer(BaseTokenizer):
    def __init__(self, words: int):
        super().__init__(words=words)


def get_gtzan_tokenizer(gtzan_labels: list[Any], codec_labels: list[Any]) -> BaseTokenizer:
    
    tokenizer = ConcatTokenizer([
        SpecialTokenizer(),
        GTZANLabelTokenizer(words=gtzan_labels),
        CodecTokenizer(words=codec_labels)
    ])

    return tokenizer