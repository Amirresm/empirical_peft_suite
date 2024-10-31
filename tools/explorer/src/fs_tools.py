import os
from typing import Any, Dict, Iterable
import json
import gzip

from src.config import ConfigMeta
from src.tui_utils import Options


def scan_dirtree(dir: str) -> list[tuple[str, list[ConfigMeta]]]:
    configs: list[tuple[str, list[ConfigMeta]]] = []
    for job_path in os.scandir(dir):
        job_path = job_path.path
        job_name = job_path.split("/")[-1]

        for model_path in os.scandir(job_path):
            model_path = model_path.path
            model_name = model_path.split("/")[-1]

            for dataset_path in os.scandir(model_path):
                dataset_path = dataset_path.path
                dataset_name = dataset_path.split("/")[-1]
                batch = []
                name = f"{job_name}_{model_name}_{dataset_name}"

                for config_path in os.scandir(dataset_path):
                    config_path = config_path.path
                    config_name = config_path.split("/")[-1]
                    config_meta = ConfigMeta.from_dirname(config_name)
                    if config_meta is not None:
                        batch.append(config_meta)

                configs.append((name, batch))

    return configs


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))


def load_cache_from_disk(
    save_path: str,
) -> tuple[list[tuple[ConfigMeta, Dict[str, Any]]], Options, str]:
    view_name = save_path.split("/")[-1].split(".")[0]
    config_rows_pairs: list[tuple[ConfigMeta, Dict[str, Any]]] = []
    dump = list(stream_jsonl(save_path))
    base_options = Options.from_dict(dump[0]["base_options"])
    for dict in dump:
        config = ConfigMeta.from_dict(dict["config"])
        rows = dict["rows"]
        config_rows_pairs.append((config, rows))

    return config_rows_pairs, base_options, view_name


def save_cache_to_disk(
    config_rows_pairs: list[tuple[ConfigMeta, Dict[str, Any]]],
    base_options: Options,
    save_path: str,
    view_name: str,
):
    dump = []
    base_options_dict = base_options.to_dict()
    for config, rows in config_rows_pairs:
        dump.append({
            "config": config.to_dict(),
            "rows": rows,
            "base_options": base_options_dict,
        })
    write_jsonl(os.path.join(save_path, f"{view_name}.jsonl"), dump)
