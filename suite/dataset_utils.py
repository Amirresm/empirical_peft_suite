from itertools import chain

from datasets import DatasetDict, load_dataset
from huggingface_hub.constants import typing

from constants import (
    COMPLETION_COL,
    PROMPT_COL,
    DatasetInstances,
    DatasetTypes,
    summarization_name_mapping,
)
from text_utils import (
    csn_create_prompt,
    csn_join,
    join_columns,
    join_prefix_prompt,
    modify_prompt,
    split_column,
    spp_split,
)
from logging_utils import logger


def detect_type(files: list[str | None]):
    type_mapping = {
        "spp": DatasetTypes.ONECOL,
        "CodeSearchNet": DatasetTypes.TWOCOL,
        "csn": DatasetTypes.TWOCOL,
    }
    instance_mapping = {
        "spp": DatasetInstances.SPP,
        "CodeSearchNet": DatasetInstances.CSN,
        "csn": DatasetInstances.CSN,
    }
    ds_type = DatasetTypes.UNKNOWN
    ds_instance = DatasetInstances.UNKNOWN
    for file in files:
        for k, v in type_mapping.items():
            if file and k.lower() in file.lower():
                ds_type = v
        for k, v in instance_mapping.items():
            if file and k.lower() in file.lower():
                ds_instance = v
    return ds_type, ds_instance


def load_raw_datasets(
    dataset_name: str | None,
    dataset_config_name: str | None,
    train_file: str | None,
    validation_file: str | None,
    test_file: str | None,
    seed: int = 42,
) -> tuple[DatasetDict, str, str]:
    extension = ""
    if dataset_name is not None:
        raw_datasets = load_dataset(
            dataset_name,
            dataset_config_name,
            # cache_dir=model_args.cache_dir,
            # token=True if model_args.use_auth_token else None,
        )
        ds_type = DatasetTypes.UNKNOWN
        ds_instance = DatasetInstances.UNKNOWN
    else:
        data_files = {}
        if train_file is not None:
            data_files["train"] = train_file
            extension = train_file.split(".")[-1]
        if validation_file is not None and not validation_file.startswith("SPLIT"):
            data_files["validation"] = validation_file
            extension = validation_file.split(".")[-1]
        if test_file is not None and not test_file.startswith("SPLIT"):
            data_files["test"] = test_file
            extension = test_file.split(".")[-1]
        if extension == "jsonl":
            extension = "json"

        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            # cache_dir=model_args.cache_dir,
            # token=True if model_args.use_auth_token else None,
        )

        ds_type, ds_instance = detect_type([train_file, validation_file, test_file])

        if isinstance(raw_datasets, DatasetDict):
            if (
                validation_file
                and "train" in raw_datasets
                and validation_file.startswith("SPLIT")
            ):
                split = float(validation_file.split("SPLIT")[-1])
                if split > 0:
                    raw_datasets["validation"] = raw_datasets["train"].train_test_split(
                        test_size=split, seed=seed
                    )["test"]
                    raw_datasets["train"] = raw_datasets["train"].train_test_split(
                        test_size=split, seed=seed
                    )["train"]

            if test_file and "train" in raw_datasets and test_file.startswith("SPLIT"):
                split = float(test_file.split("SPLIT")[-1])
                if split > 0:
                    raw_datasets["test"] = raw_datasets["train"].train_test_split(
                        test_size=split, seed=seed
                    )["test"]
                    raw_datasets["train"] = raw_datasets["train"].train_test_split(
                        test_size=split, seed=seed
                    )["train"]

    if isinstance(raw_datasets, DatasetDict):
        logger.error("dataset type is unexpected")

    if ds_type == DatasetTypes.UNKNOWN or ds_instance == DatasetInstances.UNKNOWN:
        raise ValueError(
            "Dataset type is unknown. Please implement 'detect_type' and 'split_column' functions"
        )

    return typing.cast(DatasetDict, raw_datasets), ds_type, ds_instance


def get_dataset_metadata(
    do_train,
    do_eval,
    do_predict,
    raw_datasets,
    dataset_name,
    text_column,
    summary_column,
):
    if do_train:
        column_names = raw_datasets["train"].column_names
    elif do_eval:
        column_names = raw_datasets["validation"].column_names
    elif do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        column_names = []

    dataset_columns = summarization_name_mapping.get(dataset_name, None)
    if text_column is None:
        text_column = (
            dataset_columns[0] if dataset_columns is not None else column_names[0]
        )
    else:
        text_column = text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if summary_column is None:
        summary_column = (
            dataset_columns[1] if dataset_columns is not None else column_names[1]
        )
    else:
        summary_column = summary_column
        # if summary_column not in column_names:
        #     raise ValueError(
        #         f"--summary_column' value '{summary_column}' needs to be one of: {', '.join(column_names)}"
        #     )

    return column_names, text_column, summary_column


def filter_dataset(
    dataset: DatasetDict,
    text_column: str,
    summary_column: str,
    preprocessing_num_workers: int | None = None,
):
    for split in dataset:
        if split in dataset:
            logger.info(f"Filtering {split} dataset for None records.")
            dataset[split] = dataset[split].filter(
                lambda x: x[text_column] is not None
                and (summary_column == "NONE" or x[summary_column] is not None),
                num_proc=preprocessing_num_workers,
            )


def get_column_preprocessor(
    text_column,
    summary_column,
    is_text_tokenized,
    is_summary_tokenized,
    prefix,
    ds_type: str,
    ds_instance: str,
):
    def column_preprocessor(examples):
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if (ds_type == DatasetTypes.ONECOL and examples[text_column][i]) or (
                ds_type == DatasetTypes.TWOCOL
                and examples[text_column][i]
                and examples[summary_column][i]
            ):
                if ds_type == DatasetTypes.ONECOL:
                    input = examples[text_column][i]
                    if is_text_tokenized:
                        input = " ".join(input)
                    input, target = split_column(examples[text_column][i], ds_instance)
                else:
                    input = examples[text_column][i]
                    target = examples[summary_column][i]
                    if is_text_tokenized:
                        input = " ".join(input)
                    if is_summary_tokenized:
                        target = " ".join(target)

                input = modify_prompt(input, ds_instance)
                input = join_prefix_prompt(prefix, input)
                inputs.append(input)
                targets.append(target)

        examples[PROMPT_COL] = inputs
        examples[COMPLETION_COL] = targets
        return examples

    return column_preprocessor


def get_tokenize_preprocessor(
    tokenizer,
    max_source_length,
    max_target_length,
    padding,
    is_decoder_only,
    ds_instance,
    ignore_pad_token_for_loss=False,
):
    def tokenize_preprocessor(examples):
        if is_decoder_only:
            samples = []
            for i in range(len(examples[PROMPT_COL])):
                prompt = examples[PROMPT_COL][i]
                completion = examples[COMPLETION_COL][i]
                sample = join_columns(prompt, completion, ds_instance)
                sample = f"{sample}{tokenizer.eos_token}"
                samples.append(sample)

            model_inputs = tokenizer(
                samples,
                max_length=max_source_length,
                padding=padding,
                truncation=True,
            )
            return model_inputs

        model_inputs = tokenizer(
            examples[PROMPT_COL],
            max_length=max_source_length,
            padding=padding,
            truncation=True,
        )

        labels = tokenizer(
            text_target=examples[COMPLETION_COL],
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(char if char != tokenizer.pad_token_id else -100) for char in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return tokenize_preprocessor


def get_encoder_decoder_preprocessor(
    tokenizer,
    text_column,
    summary_column,
    prefix,
    max_source_length,
    max_target_length,
    padding,
    is_text_tokenized=False,
    is_summary_tokenized=False,
    ignore_pad_token_for_loss=False,
    is_gen_job=False,
):
    def preprocess_encoder_decoder_function(examples):
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and (
                is_gen_job or summary_column == "NONE" or examples[summary_column][i]
            ):
                if is_gen_job or summary_column == "NONE":
                    # Assuming dataset is spp
                    input, target = spp_split(examples[text_column][i])
                else:
                    input = examples[text_column][i]
                    target = examples[summary_column][i]

                # ONLY CSN
                if is_text_tokenized:
                    input = " ".join(input)
                if is_summary_tokenized:
                    target = " ".join(target)

                input = join_prefix_prompt(prefix, input)
                inputs.append(input)
                targets.append(target)

        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
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
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(char if char != tokenizer.pad_token_id else -100) for char in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_encoder_decoder_function


def get_decoder_only_preprocessor(
    tokenizer,
    text_column,
    summary_column,
    max_source_length,
    padding,
    is_text_tokenized=False,
    is_summary_tokenized=False,
    is_gen_job=False,
):
    def preprocess_decoder_only_function(examples):
        samples = []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and (
                is_gen_job or summary_column == "NONE" or examples[summary_column][i]
            ):
                input = examples[text_column][i]
                target = (
                    None
                    if is_gen_job or summary_column == "NONE"
                    else examples[summary_column][i]
                )
                if is_text_tokenized:
                    input = " ".join(input)
                if target and is_summary_tokenized:
                    target = " ".join(target)

                if target is not None:
                    # we are dealing with CSN
                    input = csn_join(csn_create_prompt(input), target)

                # llama tokenzier doesn't and eos token
                sample = f"{input}{tokenizer.eos_token}"
                samples.append(sample)

        tokenized_samples = tokenizer(
            samples,
            max_length=max_source_length,
            padding=padding,
            truncation=True,
        )

        return tokenized_samples

    return preprocess_decoder_only_function


def get_group_text_preprocessor(
    block_size: int,
    # padding,
    # ignore_pad_token_for_loss=False,
):
    def group_text_preprocessor(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        labels = result["input_ids"].copy()

        # if padding == "max_length" and ignore_pad_token_for_loss:
        #     labels = [
        #         [(id if id != tokenizer.pad_token_id else -100) for id in label]
        #         for label in labels
        #     ]

        result["labels"] = labels
        return result

    return group_text_preprocessor


def process_dataset(
    raw_datasets,
    max_sample_count,
    column_names,
    split,
    main_process_first,
    tokenizer,
    text_column,
    summary_column,
    is_text_tokenized,
    is_summary_tokenized,
    prefix,
    ds_type,
    ds_instance,
    max_source_length,
    max_target_length,
    padding,
    is_decoder_only,
    ignore_pad_token_for_loss,
    overwrite_cache=False,
    preprocessing_num_workers=None,
):
    if split not in raw_datasets:
        raise ValueError(f"--do_{split} requires a train dataset")

    column_preprocessor = get_column_preprocessor(
        text_column=text_column,
        summary_column=summary_column,
        is_text_tokenized=is_text_tokenized,
        is_summary_tokenized=is_summary_tokenized,
        prefix=prefix,
        ds_type=ds_type,
        ds_instance=ds_instance,
    )

    tokenize_preprocessor = get_tokenize_preprocessor(
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        padding=padding,
        is_decoder_only=is_decoder_only,
        ignore_pad_token_for_loss=ignore_pad_token_for_loss,
        ds_instance=ds_instance,
    )

    dataset = raw_datasets[split]
    if max_sample_count is not None:
        max_sample_count = min(len(dataset), max_sample_count)
        dataset = dataset.select(range(max_sample_count))

    with main_process_first(desc=f"{split} dataset map pre-processing"):
        dataset = dataset.map(
            column_preprocessor,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=dataset.column_names,
            load_from_cache_file=not overwrite_cache,
            desc=f"Processing columns on {split} dataset",
        )

        raw_datasets[split] = dataset
        dataset = dataset.map(
            tokenize_preprocessor,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=dataset.column_names,
            load_from_cache_file=not overwrite_cache,
            desc=f"Running tokenizer on {split} dataset",
        )
        logger.info(f"{split}_dataset:\n{dataset}")
        return dataset, max_sample_count


def group_dataset(
    tokenizer,
    model,
    dataset,
    max_source_length,
    main_process_first,
    overwrite_cache=False,
    preprocessing_num_workers=None,
):
    # if data_args.block_size is None:
    block_size = tokenizer.model_max_length
    if block_size > model.config.max_position_embeddings:
        logger.warning(
            f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
            f"Using block_size={min(max_source_length, model.config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
        )
        if model.config.max_position_embeddings > 0:
            block_size = min(
                max_source_length,
                model.config.max_position_embeddings,
            )
        else:
            block_size = max_source_length
    # else:
    #     if data_args.block_size > tokenizer.model_max_length:
    #         logger.warning(
    #             f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
    #             f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
    #         )
    #     block_size = min(data_args.block_size, tokenizer.model_max_length)

    group_texts = get_group_text_preprocessor(block_size)
    with main_process_first(desc="grouping texts together"):
        dataset = dataset.map(
            group_texts,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        logger.info(f"grouped dataset:\n{dataset}")
        return dataset
