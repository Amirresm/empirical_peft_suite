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

