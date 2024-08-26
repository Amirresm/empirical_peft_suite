# A list of all multilingual tokenizer which require lang attribute.
from transformers import (
    MBartTokenizer,
    MBartTokenizerFast,
    # MBart50Tokenizer,
    MBart50TokenizerFast,
)


MULTILINGUAL_TOKENIZERS = [
    MBartTokenizer,
    MBartTokenizerFast,
    # MBart50Tokenizer,
    MBart50TokenizerFast,
]


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}
