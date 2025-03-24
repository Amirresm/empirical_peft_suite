import os

from adapters import AutoAdapterModel

from constants import MULTILINGUAL_TOKENIZERS
from src.logging_utils import logger

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    MBartTokenizerFast,
    MBartTokenizer,
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
        logger.warning("Failed to train tokenizer: Tokenizer is not trainable")

    return tokenizer


def init_model(
    model_name_or_path,
    model_config,
    cache_dir,
    use_adapterhub,
    adapter_config,
    quantization_mode=None,
    is_decoder_only=False,
    ):
    bnb_config = None
    model_dtype = None
    if quantization_mode == "4bit":
        logger.info("Quantizing model to 4-bit")
        model_dtype = torch.bfloat16
        if not use_adapterhub and adapter_config == "ia3":
            model_dtype = torch.float
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
    if is_decoder_only:
        ModelClass = AutoModelForCausalLM

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


def ensure_decoder_only_padding_token(model, tokenizer):
    if not tokenizer.pad_token:
        tokenizer.pad_token = (
            tokenizer.bos_token
        )  # https://github.com/huggingface/transformers/issues/22794
        tokenizer.padding_side = "left"
        model.config.pad_token_id = tokenizer.pad_token_id
        assert (
            model.config.pad_token_id == tokenizer.pad_token_id
        ), "The model's pad token ID does not match the tokenizer's pad token ID!"


def ensure_decoder_start_token(model, tokenizer, target_lang):
    if model.config.decoder_start_token_id is None and isinstance(
        tokenizer, (MBartTokenizer, MBartTokenizerFast)
    ):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(
                target_lang
            )

    if model.config.decoder_start_token_id is None:
        # raise ValueError(
        #     "Make sure that `config.decoder_start_token_id` is correctly defined"
        # )
        logger.warning("No decoder_start_token_id found in config")
        # model.config.decoder_start_token_id = model.config.eos_token_id


def ensure_embedding_size(model, tokenizer, max_source_length):
    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < max_source_length
    ):
        logger.warning(
            "Increasing the model's number of position embedding vectors from"
            f" {model.config.max_position_embeddings} to {max_source_length}."
        )
        logger.warning(
            f"`--max_source_length` is set to {max_source_length}, but the model only has"
            f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
            f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
            " model's position encodings by passing `--resize_position_embeddings`."
        )
        model.resize_position_embeddings(max_source_length)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))


def ensure_multilingual_tokenizer(model, tokenizer, target_lang, forced_bos_token=None):
    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            target_lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = target_lang
        tokenizer.tgt_lang = target_lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[forced_bos_token]
            if forced_bos_token is not None
            else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id
