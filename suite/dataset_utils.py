from datasets import load_dataset

def load_raw_datasets(dataset_name, dataset_config_name, train_file, validation_file, test_file, seed = 42):
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
        if (
            validation_file is not None
            and not validation_file.startswith("SPLIT")
        ):
            data_files["validation"] = validation_file
            extension = validation_file.split(".")[-1]
        if (
            test_file is not None
            and not test_file.startswith("SPLIT")
        ):
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

        if "train" in raw_datasets and validation_file.startswith(
            "SPLIT"
        ):
            split = float(validation_file.split("SPLIT")[-1])
            if split > 0:
                raw_datasets["validation"] = raw_datasets[
                    "train"
                ].train_test_split(test_size=split, seed=seed)[
                    "test"
                ]
                raw_datasets["train"] = raw_datasets["train"].train_test_split(
                    test_size=split, seed=seed
                )["train"]

        if "train" in raw_datasets and test_file.startswith("SPLIT"):
            split = float(test_file.split("SPLIT")[-1])
            if split > 0:
                raw_datasets["test"] = raw_datasets["train"].train_test_split(
                    test_size=split, seed=seed
                )["test"]
                raw_datasets["train"] = raw_datasets["train"].train_test_split(
                    test_size=split, seed=seed
                )["TRAIN"]

    return raw_datasets

def get_dataset_mappers(
        tokenizer,
        text_column,
        summary_column,
        prefix,
        max_source_length,
        max_target_length,
        padding,
        is_text_tokenized=False,
        is_summary_tokenized=False,
        ):
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
