import os
import pathlib
import json
import datetime

from filelock import FileLock
import nltk

from logging_utils import logger

import adapters
import accelerate
import torch
import transformers
import bitsandbytes

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode
from transformers.utils.versions import require_version


def check_dependencies():
    print("PYTHON HEALTH CHECK (adp)", flush=True)
    print("torch: ", torch.__version__)
    print("transformers: ", transformers.__version__)
    print("bitsandbytes: ", bitsandbytes.__version__)
    print("adapters: ", adapters.__version__)
    print("accelerate: ", accelerate.__version__)


def check_version():
    check_min_version("4.26.0")
    require_version(
        "datasets>=1.8.0",
        "To fix: pip install -r examples/pytorch/summarization/requirements.txt",
    )


def check_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        if is_offline_mode():
            raise LookupError(
                "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
            )
        with FileLock(".lock") as _:
            nltk.download("punkt", quiet=True)


def ensure_path_exists(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def handle_last_checkpoint(output_dir, overwrite_output_dir, resume_from_checkpoint):
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


class CudaTimer:
    def __init__(self):
        if not torch.cuda.is_available():
            logger.warning("CudaTimer: CUDA is not available")
        else:
            self.timer = torch.cuda.Event(enable_timing=True)
            self.end_timer = torch.cuda.Event(enable_timing=True)

    def start(self):
        if torch.cuda.is_available():
            self.timer.record()

    def stop(self):
        if torch.cuda.is_available():
            self.end_timer.record()
            torch.cuda.synchronize()
            # self.timer.record()
            return self.timer.elapsed_time(self.end_timer) / (1000)
        return None

def _secs2timedelta(secs):
    """
    convert seconds to hh:mm:ss.msec, msecs rounded to 2 decimals
    """

    msec = int(abs(secs - int(secs)) * 100)
    return f"{datetime.timedelta(seconds=int(secs))}.{msec:02d}"

def metrics_format(metrics):
    """
    Reformat Trainer metrics values to a human-readable format

    Args:
        metrics (`Dict[str, float]`):
            The metrics returned from train/evaluate/predict

    Returns:
        metrics (`Dict[str, float]`): The reformatted metrics
    """

    metrics_copy = metrics.copy()
    for k, v in metrics_copy.items():
        if "_mem_" in k:
            metrics_copy[k] = f"{ v >> 20 }MB"
        elif "_runtime" in k:
            metrics_copy[k] = _secs2timedelta(v)
        elif k == "total_flos":
            metrics_copy[k] = f"{ int(v) >> 30 }GF"
        elif isinstance(metrics_copy[k], float):
            metrics_copy[k] = round(v, 4)

    return metrics_copy


def log_metrics(split, metrics):
    print(f"***** {split} metrics *****")
    metrics_formatted = metrics_format(metrics)
    k_width = max(len(str(x)) for x in metrics_formatted.keys())
    v_width = max(len(str(x)) for x in metrics_formatted.values())
    for key in sorted(metrics_formatted.keys()):
        print(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")

def save_metrics(split, metrics, output_dir, combined=True):
    output_path = os.path.join(output_dir, f"{split}_results.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4, sort_keys=True)

    if combined:
        output_dir = os.path.join(output_dir, "all_results.json")
        if os.path.exists(output_dir):
            with open(output_dir, "r") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}

        all_metrics.update(metrics)
        with open(output_dir, "w") as f:
            json.dump(all_metrics, f, indent=4, sort_keys=True)

