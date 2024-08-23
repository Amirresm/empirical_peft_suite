import os
import pathlib

from advf_utils import (
    freeze_adapter,
    unfreeze_reload_adapter,
    zero_freeze_adapter,
)
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np

from suite.adapter_utils import init_ah_adapter, init_ah_advfusion, load_ah_adapter
from suite.init_utils import (
    get_training_corpus,
    init_config,
    init_model,
    init_tokenizer,
    train_tokenizer,
)
import torch
import adapters
import evaluate
import transformers
from adapters import (
    Seq2SeqAdapterTrainer,
    AdapterTrainer,
    SeqBnConfig,
    setup_adapter_training,
    AdapterConfig,
    LoRAConfig,
    CompacterConfig,
    IA3Config,
)
from adapters.composition import Fuse
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    Trainer,
    set_seed,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode
from transformers.utils.versions import require_version

from bleu2.calc_bleu2 import calculate_bleu2

from arg_utils import parse_arguments
from logging_utils import logger

has_codebleu = False
# try:
#     from codebleu import calc_codebleu
#     has_codebleu = True
# except ImportError:
#     print("CodeBLEU not found", flush=True)

import adapters
import accelerate
import torch
import transformers
import bitsandbytes

print("PYTHON HEALTH CHECK (adp)", flush=True)
print("torch: ", torch.__version__)
print("transformers: ", transformers.__version__)
print("bitsandbytes: ", bitsandbytes.__version__)
print("adapters: ", adapters.__version__)
print("accelerate: ", accelerate.__version__)

check_min_version("4.26.0")
require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/summarization/requirements.txt",
)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
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


def ensure_path_exists(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def create_llama_prompt(
    input_text, target_text=None, is_training=False, eos_token="</s>"
):
    if target_text is None:
        return f"{input_text}{eos_token if is_training else ''}"
    else:
        return f"[INST] Do not define a function. Do not import anything. Do not write any comments. Generate one line of Python code snippet to satisfy the following description: {input_text}. [/INST] CODE: {target_text}</s>"


def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")


@torch.inference_mode()
def generate_batch_completion(
    model, tokenizer, prompt, batch_size
) -> list[str]:
    prompt_input = create_llama_prompt(prompt)
    input_batch = [prompt_input for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **inputs,
        # use_cache=True,
        max_new_tokens=200,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    batch_completions = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    # res = [filter_code(fix_indents(extract_code(completion))) for completion in batch_completions]
    res = [fix_indents(completion) for completion in batch_completions]
    res = batch_completions
    logger.info(f"Generated completions prompt:\n {prompt}")
    # logger.info(f"Generated completions raw:\n {batch_completions[0]}")
    logger.info(f"Generated completions example:\n {res[0]}")
    return res


def clean_whitespaces_generations(text):
    trim_list = [" ", "\n"]
    trim_map = {" ": "sp", "\n": "nl"}
    new_text = text[0]
    last_ch = text[0]
    occ = 0
    for ch in text[1:]:
        if last_ch in trim_list and ch != last_ch:
            if occ > 20:
                new_text += f"<{trim_map[last_ch]}{occ}>"
                occ = 0
            else:
                new_text += last_ch * (occ + 1)
                occ = 0
        if ch not in trim_list:
            new_text += ch
        else:
            if ch == last_ch:
                occ += 1
        last_ch = ch

    if last_ch in trim_list:
        if occ > 1:
            new_text += f"<{trim_map[last_ch]}{occ}>"
            occ = 0
        else:
            new_text += last_ch

    return new_text


def handle_last_checkpoint(
    output_dir, overwrite_output_dir, resume_from_checkpoint
):
    last_checkpoint = None
    if os.path.isdir(output_dir) and not overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is None and len(os.listdir(output_dir)) > 3:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


def main():
    print("Python script starting...")
    logger.info("Checking logger.info")
    logger.warning("Checking logger.warn")
    logger.error("Checking logger.error")

    torch.cuda.empty_cache()

    model_args, data_args, training_args, adapter_args, advfusion_args = (
        parse_arguments()
    )

    advadp_path_list: list[str] = advfusion_args.advadp_path_list

    setup_logging(logger, training_args.get_process_log_level)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}\n")
    logger.info(f"Adapter parameters {adapter_args}\n")
    logger.info(f"Data parameters {data_args}\n")
    logger.info(f"Model parameters {model_args}\n")

    if data_args.source_prefix is None and (
        model_args.model_name_or_path.startswith("t5")
        or model_args.model_name_or_path.startswith("codet5")
    ):
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    last_checkpoint = None
    if training_args.do_train:
        last_checkpoint = handle_last_checkpoint(
            training_args.output_dir,
            training_args.overwrite_output_dir,
            training_args.resume_from_checkpoint,
        )

    set_seed(training_args.seed)
    is_decoder_only = "llama" in model_args.model_name_or_path.lower()

    raw_datasets = load_raw_datasets(
        data_args.dataset_name,
        data_args.dataset_config_name,
        data_args.train_file,
        data_args.validation_file,
        data_args.test_file,
        training_args.seed,
    )
    logger.info(f"raw_datasets: {raw_datasets}")
    # if training_args.do_train:
    #     logger.info(f"First training sample: {raw_datasets['train'][0]}")
    # if training_args.do_eval:
    #     logger.info(f"First eval sample: {raw_datasets['validation'][0]}")
    # if training_args.do_predict:
    #     logger.info(f"First test sample: {raw_datasets['test'][0]}")

    [h.flush() for h in logger.handlers]

    config = init_config(
        config_name=model_args.config_name,
        model_name_or_path=model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = init_tokenizer(
        tokenizer_name_or_path=model_args.tokenizer_name_or_path,
        model_name_or_path=model_args.model_name_or_path,
        use_fast_tokenizer=model_args.use_fast_tokenizer,
        cache_dir=model_args.cache_dir,
    )

    if model_args.train_tokenizer and model_args.use_fast_tokenizer:
        logger.info("Training tokenizer...")
        training_corpus = get_training_corpus(
            raw_datasets["train"],
            [data_args.text_column, data_args.summary_column],
            1000,
        )
        tokenizer = train_tokenizer(tokenizer, training_corpus)

    model, model_dtype = init_model(
        model_name_or_path=model_args.model_name_or_path,
        model_config=config,
        cache_dir=model_args.cache_dir,
        quantization_mode=model_args.quantization_mode,
        is_decoder_only=is_decoder_only,
    )

    # Convert the model into an adapter model
    if adapter_args.train_adapter:
        match adapter_args.adapter_config:
            case "advfusion":
                (target_adapter_path, target_adapter_name, fusion_name) = (
                    init_ah_advfusion(
                        advadp_path_list=advadp_path_list,
                        advfusion_target=advfusion_args.advfusion_target,
                        model=model,
                        model_dtype=model_dtype,
                    )
                )
                advfusion_args.target_adapter_path = target_adapter_path
                advfusion_args.target_adapter_name = target_adapter_name
                advfusion_args.fusion_name = fusion_name

            case _:
                adapter_name = init_ah_adapter(
                        adapter_config=adapter_args.adapter_config,
                        config_title=model_args.config_title,
                        model=model,
                        model_dtype=model_dtype
                        )
                if model_args.preload_adapter:
                    load_ah_adapter(
                            adapter_path=model_args.adapter_path,
                            adapter_name=adapter_name,
                            set_active=True
                            )
        # logger.info(f"Active heads: {model.active_head()}")
        logger.info(f"Adapter Summary:\n{model.adapter_summary()}")


    logger.info(f"Model architucture:\n{model}")

    if is_decoder_only and not tokenizer.pad_token:
        tokenizer.pad_token = (
            tokenizer.bos_token
        )  # https://github.com/huggingface/transformers/issues/22794
        tokenizer.padding_side = "left"
        model.config.pad_token_id = tokenizer.pad_token_id
        assert (
            model.config.pad_token_id == tokenizer.pad_token_id
        ), "The model's pad token ID does not match the tokenizer's pad token ID!"

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None and isinstance(
        tokenizer, (MBartTokenizer, MBartTokenizerFast)
    ):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[
                data_args.lang
            ]
        else:
            model.config.decoder_start_token_id = (
                tokenizer.convert_tokens_to_ids(data_args.lang)
            )

    if model.config.decoder_start_token_id is None:
        # raise ValueError(
        #     "Make sure that `config.decoder_start_token_id` is correctly defined"
        # )
        logger.info("No decoder_start_token_id found in config")
        # model.config.decoder_start_token_id = model.config.eos_token_id

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = (
        data_args.source_prefix if data_args.source_prefix is not None else ""
    )

    # Preprocessing the datasets.
    # for split in ["train", "validation", "test"]:
    #     if split in raw_datasets:
    #         logger.info(f"Filtering {split} dataset for None records.")
    #         raw_datasets[split] = raw_datasets[split].filter(
    #             lambda x: x[data_args.text_column] is not None
    #             and (
    #                 data_args.summary_column == "NONE"
    #                 or x[data_args.summary_column] is not None
    #             ),
    #             num_proc=data_args.preprocessing_num_workers,
    #         )
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token]
            if data_args.forced_bos_token is not None
            else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(
        data_args.dataset_name, None
    )
    if data_args.text_column is None:
        text_column = (
            dataset_columns[0]
            if dataset_columns is not None
            else column_names[0]
        )
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = (
            dataset_columns[1]
            if dataset_columns is not None
            else column_names[1]
        )
    else:
        summary_column = data_args.summary_column
        # if summary_column not in column_names:
        #     raise ValueError(
        #         f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
        #     )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(
        model, "prepare_decoder_input_ids_from_labels"
    ):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    logger.info(f"Tokenizer pad token: {tokenizer.pad_token}")

    def preprocess_encoder_decoder_function(examples):
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                input = examples[text_column][i]
                target = examples[summary_column][i]
                if data_args.text_tokenized:
                    input = " ".join(input)
                if data_args.summary_tokenized:
                    target = " ".join(target)
                inputs.append(input)
                targets.append(target)
        inputs = [prefix + inp for inp in inputs]
        # logger.info(f"preprocess inp:\n{inputs[0]}")
        # logger.info(f"preprocess tgt:\n{targets[0]}")
        model_inputs = tokenizer(
            inputs,
            max_length=data_args.max_source_length,
            padding=padding,
            truncation=True,
        )

        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [
                    (char if char != tokenizer.pad_token_id else -100)
                    for char in label
                ]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_decoder_only_function(examples):
        samples = []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and (
                summary_column == "NONE" or examples[summary_column][i]
            ):
                input = examples[text_column][i]
                target = (
                    None
                    if summary_column == "NONE"
                    else examples[summary_column][i]
                )
                if data_args.text_tokenized:
                    input = " ".join(input)
                if target and data_args.summary_tokenized:
                    target = " ".join(target)

                # input = 'def '.join(input.split('def ')[:2])
                sample = create_llama_prompt(
                    input, is_training=True, eos_token=tokenizer.eos_token
                )
                samples.append(sample)

        tokenized_samples = tokenizer(
            samples,
            max_length=data_args.max_source_length,
            padding=padding,
            truncation=True,
        )

        labels = tokenized_samples["input_ids"]
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels
            ]

        tokenized_samples["labels"] = labels
        return tokenized_samples

    preprocess_function = (
        preprocess_decoder_only_function
        if is_decoder_only
        else preprocess_encoder_decoder_function
    )

    def generation_preprocess_function(examples):
        samples = []
        targets = []
        for i in range(len(examples[text_column])):
            if examples[text_column][i]:
                input = examples[text_column][i]
                if data_args.text_tokenized:
                    input = " ".join(input)

                targets.append(
                    f"{create_llama_prompt(input, is_training=True, eos_token=tokenizer.eos_token)}"
                )
                input = '"""'.join(input.split('"""')[:2]) + '"""\n'
                sample = create_llama_prompt(
                    input, is_training=False, eos_token=tokenizer.eos_token
                )
                samples.append(sample)

        tokenized_samples = tokenizer(
            samples,
            max_length=data_args.max_source_length,
            padding=padding,
            truncation=True,
        )

        tokenized_targets = tokenizer(
            targets,
            max_length=data_args.max_source_length,
            padding=padding,
            truncation=True,
        )

        labels = tokenized_targets["input_ids"]
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels
            ]

        tokenized_samples["labels"] = labels
        return tokenized_samples

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples
            )
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(
            desc="train dataset map pre-processing"
        ):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            logger.info(f"train_dataset:\n{train_dataset}")

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples
            )
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            logger.info(f"eval_dataset:\n{eval_dataset}")

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
            logger.info(f"predict_dataset:\n{predict_dataset}")

    if training_args.predict_with_generate:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("Generation requires a test dataset")
        generation_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(generation_dataset), data_args.max_predict_samples
            )
            generation_dataset = generation_dataset.select(
                range(max_predict_samples)
            )
        with training_args.main_process_first(
            desc="generation dataset map pre-processing"
        ):
            generation_dataset = generation_dataset.map(
                generation_preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on generation dataset",
            )
            logger.info(f"generation_dataset:\n{generation_dataset}")
    # Data collator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    # data_collator_class = DataCollator if is_decoder_only and False else DataCollatorForSeq2Seq
    data_collator_class = (
        DataCollatorForLanguageModeling
        if is_decoder_only
        else DataCollatorForSeq2Seq
    )
    data_collator = (
        data_collator_class(
            tokenizer,
            mlm=False,
        )
        if is_decoder_only
        else data_collator_class(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    )

    if data_args.metric_path is not None:
        metric_bleu = evaluate.load(path=data_args.metric_path)
    if data_args.metric_path_alt is not None:
        metric_rouge = evaluate.load(path=data_args.metric_path_alt)

    performance_metrics = {}

    def postprocess_text(preds, labels):
        if is_decoder_only:
            new_preds = []
            new_labels = []
            for pred in preds:
                splits = pred.split('"""')
                if len(splits) == 3:
                    new_preds.append(splits[2].strip())
                else:
                    new_preds.append(pred)

            for label in labels:
                splits = label.split('"""')
                if len(splits) == 3:
                    new_labels.append(splits[2].strip())
                else:
                    new_labels.append(label)

            return new_preds, new_labels
        else:
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            # rougeLSum expects newline after each sentence
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
            return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        inspect_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
        inspect_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=False
        )
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        logger.info(
            f"\nCompute_metrics Preds example:\n{clean_whitespaces_generations(inspect_preds[0])}\n"
        )
        logger.info(
            f"\nCompute_metrics Labels example:\n{clean_whitespaces_generations(inspect_labels[0])}\n\n"
        )
        decoded_preds, decoded_labels = postprocess_text(
            decoded_preds, decoded_labels
        )

        result = metric_rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        result = {f"ROUGE_{k}": round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)

        # CodeBERT bleu metric
        bleu2, b2args = calculate_bleu2(
            decoded_preds, decoded_labels, smooth=True
        )
        bleu2 = {
            f"BLEU2_{k}": str(v) if isinstance(v, list) else v
            for k, v in bleu2.items()
        }
        result = {**result, **bleu2}
        if data_args.metric_path is not None:
            if any(
                [len(decoded_pred) > 0 for decoded_pred in decoded_preds]
            ) and any(
                [len(decoded_label) > 0 for decoded_label in decoded_labels]
            ):
                result_bleu = metric_bleu.compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                    smooth=True,
                )
                result_bleu["bleuP"] = round(result_bleu["bleu"] * 100, 4)
                result_bleu = {
                    f"BLEU_{k}": str(v) if isinstance(v, list) else v
                    for k, v in result_bleu.items()
                }
            else:
                logger.info(
                    f"Skipping BLEU computation as decoded_preds is empty: \n {decoded_preds[:20]} \n decoded_labels: \n {decoded_labels[:20]}"
                )
                result_bleu = {
                    "BLEU_bleu": -1.0,
                    "BLEU_bleuP": -1.0,
                    "BLEU_brevity_penalty": -1.0,
                    "BLEU_length_ratio": -1.0,
                    "BLEU_precisions": -1.0,
                    "BLEU_reference_length": -1.0,
                    "BLEU_translation_length": -1.0,
                }
        if data_args.metric_path is not None:
            result = {**result, **result_bleu}

        if has_codebleu:
            cb_results = calc_codebleu(
                [[l] for l in decoded_labels], decoded_preds, lang="python"
            )
            cb_results["codebleuP"] = results["codebleu"] * 100
            result = {**result, **cb_results}

        return result

    def preprocess_logits_for_metrics(logits, labels):
        # logger.info(f"preprocess logits:\n{logits}\nlabels:\n{labels}\nend preprocess")
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    # Early stopping
    if data_args.patience and data_args.patience > 0:
        training_args.load_best_model_at_end = True

    logger.info(f"Model memory footprint:\n{model.get_memory_footprint()}")

    tokenizer.save_pretrained(model_args.tokenizer_name_or_path)
    logger.info(f"Tokenizer saved to {model_args.tokenizer_name_or_path}")

    # Initialize our Trainer
    trainer: Trainer | None = None
    if training_args.do_train:  # or training_args.do_eval:
        if adapter_args.train_adapter:
            trainer_class = (
                AdapterTrainer if is_decoder_only else Seq2SeqAdapterTrainer
            )
        else:
            trainer_class = Trainer if is_decoder_only else Seq2SeqTrainer

        logger.info(
            f"metric for choosing best model is {training_args.metric_for_best_model}"
        )
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        logger.info(
            f"PREDICT WITH GENERATE {training_args.predict_with_generate}"
        )
        if data_args.patience and data_args.patience > 0:
            callback = EarlyStoppingCallback(
                early_stopping_patience=data_args.patience
            )
            trainer.add_callback(callback)

    # Training
    max_train_samples = (
        data_args.max_train_samples
        if data_args.max_train_samples is not None
        else len(train_dataset)
    )
    if training_args.do_train and max_train_samples > 0:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        if adapter_args.adapter_config == "advfusion":
            zero_freeze_adapter(
                model, advfusion_args.target_adapter_name, model_dtype
            )

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        if adapter_args.adapter_config == "advfusion":
            metrics = train_result.metrics
            metrics["train_samples"] = min(
                max_train_samples, len(train_dataset)
            )
            trainer.log_metrics("train_before", metrics)
            trainer.save_metrics("train_before", metrics)
            trainer.save_state()
            unfreeze_reload_adapter(
                model,
                advfusion_args.target_adapter_path,
                advfusion_args.target_adapter_name,
            )
            train_result = trainer.train(resume_from_checkpoint=checkpoint)

        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            total_time = start.elapsed_time(end) / (1000 * 60)
            performance_metrics.update({"total_gpu_time": total_time})

        trainer.save_model()
        if adapter_args.train_adapter:
            ensure_path_exists(model_args.adapter_path)
            if adapter_args.adapter_config == "advfusion":
                model.save_adapter_fusion(
                    model_args.adapter_path, advfusion_args.fusion_name
                )
                logger.info(
                    f"Fusion {advfusion_args.fusion_name} saved to {model_args.adapter_path}"
                )
            else:
                model.save_adapter(model_args.adapter_path, adapter_name)
                logger.info(
                    f"Adapter {adapter_name} saved to {model_args.adapter_path}"
                )

        metrics = train_result.metrics
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        if torch.cuda.is_available():
            peak_memory = (torch.cuda.max_memory_allocated() / 1024**2) / 1000
            performance_metrics.update({"peak_memory": peak_memory})
            logger.info(f"Performance metrics: {performance_metrics}")
            trainer.save_metrics("performance", performance_metrics)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = (
        data_args.num_beams
        if data_args.num_beams is not None
        else training_args.generation_num_beams
    )
    max_eval_samples = (
        data_args.max_eval_samples
        if data_args.max_eval_samples is not None
        else len(eval_dataset)
    )
    if training_args.do_eval and max_eval_samples > 0:
        logger.info("*** Evaluate ***")
        metrics = (
            trainer.evaluate(metric_key_prefix="eval")
            if is_decoder_only
            else trainer.evaluate(
                max_length=max_length,
                # num_beams=num_beams, metric_key_prefix="eval"
            )
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = (
            trainer.predict(
                predict_dataset,
                metric_key_prefix="predict",
            )
            if is_decoder_only
            else trainer.predict(
                predict_dataset,
                metric_key_prefix="predict",
                # max_length=max_length,
                # num_beams=num_beams,
            )
        )

        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(
            max_predict_samples, len(predict_dataset)
        )

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                source = raw_datasets["test"].select(
                    range(min(max_predict_samples, len(predict_dataset)))
                )
                labels = predict_results.label_ids
                preds = predict_results.predictions
                labels = np.where(
                    labels != -100, labels, tokenizer.pad_token_id
                )
                preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

                raw_inputs = predict_dataset["input_ids"]
                raw_inputs = np.where(
                    raw_inputs != -100,
                    raw_inputs,
                    tokenizer.pad_token_id,
                )
                raw_labels = predict_dataset["labels"]
                raw_labels = [
                    [tokenizer.pad_token_id if t == -100 else t for t in rl]
                    for rl in raw_labels
                ]
                raw_inputs = tokenizer.batch_decode(
                    raw_inputs,
                    skip_special_tokens=True,
                )
                raw_labels = tokenizer.batch_decode(
                    raw_labels,
                    skip_special_tokens=True,
                )
                labels = tokenizer.batch_decode(
                    labels,
                    skip_special_tokens=True,
                )
                preds = tokenizer.batch_decode(
                    preds,
                    skip_special_tokens=True,
                )
                pairs = [
                    f"{index + 1}=========\n->Original Input:\n{or_inp}\n->Original Target:\n{or_tgt}\n->Reconstructed Target:\n{orig}\n->Reconstructed Predication:\n{pred}\n->Raw Input:\n{raw_input}\n->Raw Target:\n{raw_label}\n--\n"
                    for pred, orig, or_inp, or_tgt, raw_input, raw_label, index in zip(
                        preds,
                        labels,
                        source[data_args.text_column],
                        source[data_args.summary_column],
                        raw_inputs,
                        raw_labels,
                        range(len(preds)),
                    )
                ]
                output_prediction_file = os.path.join(
                    (
                        model_args.generation_output_path
                        if model_args.generation_output_path is not None
                        else training_args.output_dir
                    ),
                    "generated_predictions.txt",
                )
                ensure_path_exists(
                    (
                        model_args.generation_output_path
                        if model_args.generation_output_path is not None
                        else training_args.output_dir
                    )
                )
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(pairs))

                logger.info(
                    f"{len(pairs)} predictions saved to {output_prediction_file}"
                )
        # if trainer.is_world_process_zero():
        #     source = raw_datasets["test"].select(
        #         # range(min(max_predict_samples, len(predict_dataset)))
        #         range(10)
        #     )
        #     for i, inputs in enumerate(source):
        #         inputs = "summarize: " + " ".join(inputs["code_tokens"])
        #         logger.info(f"Input: {inputs}")
        #         inputs = tokenizer.encode(inputs, return_tensors="pt").to(model.device)
        #         outputs = model.generate(inputs)
        #         outputs = tokenizer.decode(outputs[0])
        #         logger.info(f"Output: {outputs}")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "summarization",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = (
                f"{data_args.dataset_name} {data_args.dataset_config_name}"
            )
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
