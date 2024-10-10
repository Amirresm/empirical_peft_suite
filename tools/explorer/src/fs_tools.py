import os

from src.config import ConfigMeta


def scan_dirtree(dir: str) -> list[ConfigMeta]:
    configs: list[ConfigMeta] = []
    for job_path in os.scandir(dir):
        job_path = job_path.path
        # job_name = job_path.split("/")[-1]

        for model_path in os.scandir(job_path):
            model_path = model_path.path
            # model_name = model_path.split("/")[-1]

            for dataset_path in os.scandir(model_path):
                dataset_path = dataset_path.path
                # dataset_name = dataset_path.split("/")[-1]

                for config_path in os.scandir(dataset_path):
                    config_path = config_path.path
                    config_name = config_path.split("/")[-1]
                    config_meta = ConfigMeta.from_dirname(config_name)
                    if config_meta is not None:
                        configs.append(config_meta)

    return configs
