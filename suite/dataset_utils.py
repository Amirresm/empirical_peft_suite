from itertools import chain

from datasets import DatasetDict, load_dataset
from huggingface_hub.constants import typing

from text_utils import create_llama_prompt, find_nth
from logging_utils import logger


def load_raw_datasets(
    dataset_name: str | None,
    dataset_config_name: str | None,
    train_file: str | None,
    validation_file: str | None,
    test_file: str | None,
    seed: int = 42,
) -> DatasetDict:
    extension = ""
    if dataset_name is not None:
        raw_datasets = load_dataset(
            dataset_name,
            dataset_config_name,
            # cache_dir=model_args.cache_dir,
            # token=True if model_args.use_auth_token else None,
        )
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

    return typing.cast(DatasetDict, raw_datasets)


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
):
    def preprocess_encoder_decoder_function(examples):
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                input = examples[text_column][i]
                target = examples[summary_column][i]
                if is_text_tokenized:
                    input = " ".join(input)
                if is_summary_tokenized:
                    target = " ".join(target)
                inputs.append(input)
                targets.append(target)
        inputs = [prefix + inp for inp in inputs]
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


def get_decoder_only_preprocessor_depr(
    tokenizer,
    text_column,
    summary_column,
    max_source_length,
    padding,
    is_text_tokenized=False,
    is_summary_tokenized=False,
    ignore_pad_token_for_loss=False,
):
    def preprocess_decoder_only_function(examples):
        samples = []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and (
                summary_column == "NONE" or examples[summary_column][i]
            ):
                input = examples[text_column][i]
                target = (
                    None if summary_column == "NONE" else examples[summary_column][i]
                )
                if is_text_tokenized:
                    input = " ".join(input)
                if target and is_summary_tokenized:
                    target = " ".join(target)

                # input = 'def '.join(input.split('def ')[:2])
                sample = create_llama_prompt(
                    input, is_training=True, eos_token=tokenizer.eos_token
                )
                samples.append(sample)

        tokenized_samples = tokenizer(
            samples,
            max_length=max_source_length,
            padding=padding,
            truncation=True,
        )

        labels = tokenized_samples["input_ids"]
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels = [
                [(id if id != tokenizer.pad_token_id else -100) for id in label]
                for label in labels
            ]

        tokenized_samples["labels"] = labels
        return tokenized_samples

    return preprocess_decoder_only_function


def get_decoder_only_preprocessor(
    tokenizer,
    text_column,
    summary_column,
    max_source_length,
    padding,
    is_text_tokenized=False,
    is_summary_tokenized=False,
    ignore_pad_token_for_loss=False,
):
    def preprocess_decoder_only_function(examples):
        samples = []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and (
                summary_column == "NONE" or examples[summary_column][i]
            ):
                input = examples[text_column][i]
                target = (
                    None if summary_column == "NONE" else examples[summary_column][i]
                )
                if is_text_tokenized:
                    input = " ".join(input)
                if target and is_summary_tokenized:
                    target = " ".join(target)

                # input = 'def '.join(input.split('def ')[:2])

                if target is not None:
                    input = f"# code:\n{input}\n# summarize:\n{target}"
                sample = create_llama_prompt(
                    input, is_training=True, eos_token=tokenizer.eos_token
                )
                samples.append(sample)

        tokenized_samples = tokenizer(
            samples,
            max_length=max_source_length,
            padding=padding,
            truncation=True,
        )

        # labels = tokenized_samples["input_ids"].copy()
        # if padding == "max_length" and ignore_pad_token_for_loss:
        #     labels = [
        #         [(id if id != tokenizer.pad_token_id else -100) for id in label]
        #         for label in labels
        #     ]

        # tokenized_samples["labels"] = labels
        return tokenized_samples

    return preprocess_decoder_only_function

def get_generation_preprocessor(
    tokenizer,
    text_column,
    max_source_length,
    padding,
    is_text_tokenized=False,
    ignore_pad_token_for_loss=False,
):
    def generation_preprocess_function(examples):
        samples = []
        targets = []
        for i in range(len(examples[text_column])):
            if examples[text_column][i]:
                input = examples[text_column][i]
                if is_text_tokenized:
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
            max_length=max_source_length,
            padding=padding,
            truncation=True,
        )

        tokenized_targets = tokenizer(
            targets,
            max_length=max_source_length,
            padding=padding,
            truncation=True,
        )

        labels = tokenized_targets["input_ids"]
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels = [
                [(id if id != tokenizer.pad_token_id else -100) for id in label]
                for label in labels
            ]

        tokenized_samples["labels"] = labels
        return tokenized_samples

    return generation_preprocess_function

def get_text_grouper(block_size: int):
    def group_texts(examples):
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
        result["labels"] = result["input_ids"].copy()
        return result

    return group_texts
