import os
from logging_utils import logger

import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def get_training_corpus(raw_datasets, cols, step):
    dataset = raw_datasets
    for start_idx in range(0, len(dataset), step):
        samples = dataset[start_idx : start_idx + step]
        yield "\n".join([samples[col] for col in cols])


def init_config(config_name, model_name_or_path, cache_dir=None):
    config = AutoConfig.from_pretrained(
        (config_name if config_name else model_name_or_path),
        cache_dir=cache_dir,
        # revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
    )
    return config


def init_tokenizer(
    tokenizer_name_or_path,
    model_name_or_path=None,
    use_fast_tokenizer=False,
    cache_dir=None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        (
            tokenizer_name_or_path
            if tokenizer_name_or_path and os.path.isdir(tokenizer_name_or_path)
            else model_name_or_path
        ),
        cache_dir=cache_dir,
        use_fast=use_fast_tokenizer,
        # revision=model_revision,
        # use_auth_token=True if use_auth_token else None,
    )

    return tokenizer


def train_tokenizer(tokenizer, training_corpus):
    if hasattr(tokenizer, "train_new_from_iterator") and callable(
        tokenizer.train_new_from_iterator
    ):
        tokenizer = tokenizer.train_new_from_iterator(training_corpus)
    else:
        logger.warn("Failed to train tokenizer: Tokenizer is not trainable")

    return tokenizer


def init_model(
    model_name_or_path,
    model_config,
    cache_dir,
    quantization_mode=None,
    is_decoder_only=False,
):
    bnb_config = None
    model_dtype = None
    if quantization_mode == "4bit":
        logger.info("Quantizing model to 4-bit")
        model_dtype = torch.bfloat16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=model_dtype,
        )
    elif quantization_mode == "8bit":
        logger.info("Quantizing model to 8-bit")
        model_dtype = None
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    ModelClass = AutoModelForSeq2SeqLM
    # TODO: uncomment
    # if is_decoder_only:
    #     ModelClass = AutoModelForCausalLM

    model = ModelClass.from_pretrained(
        model_name_or_path,
        config=model_config,
        cache_dir=cache_dir,
        # revision=model_revision,
        # use_auth_token=True if use_auth_token else None,
        quantization_config=bnb_config,
        # device_map="auto" if is_decoder_only else None,
        device_map="auto",
        torch_dtype=model_dtype,
    )
    if is_decoder_only:
        model.config.use_cache = False

    # TODO: make sure
    if model_dtype is None:
        model_dtype = model.dtype

    return model, model_dtype
