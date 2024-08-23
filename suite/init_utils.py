
from logging_utils import logger


def init_config():
    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
def init_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name_or_path
            if model_args.tokenizer_name_or_path
            and os.path.isdir(model_args.tokenizer_name_or_path)
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if (
        model_args.train_tokenizer
        and model_args.use_fast_tokenizer
        and hasattr(tokenizer, "train_new_from_iterator")
        and callable(tokenizer.train_new_from_iterator)
    ):
        logger.info("Training tokenizer...")
        training_corpus = get_training_corpus(
            raw_datasets["train"],
            [data_args.text_column, data_args.summary_column],
            1000,
        )
        tokenizer = tokenizer.train_new_from_iterator(training_corpus)

def init_model():
    bnb_config = None
    model_dtype = None
    if model_args.quantization_mode == "4bit":
        logger.info("Quantizing model to 4-bit")
        model_dtype = torch.bfloat16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=model_dtype,
        )
    elif model_args.quantization_mode == "8bit":
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
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        quantization_config=bnb_config,
        # device_map="auto" if is_decoder_only else None,
        device_map="auto",
        torch_dtype=model_dtype,
    )
    if is_decoder_only:
        model.config.use_cache = False
