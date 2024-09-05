import os
from pathlib import Path
import json
import gzip
from typing import Optional
import sys
import numpy as np
import itertools
import argparse


def gunzip_json(path: Path) -> Optional[dict]:
    """
    Reads a .json.gz file, but produces None if any error occurs.
    """
    try:
        with gzip.open(path, "rt") as f:
            return json.load(f)
    except Exception as e:
        print(e)
        return None


def open_json(path: Path) -> Optional[dict]:
    """
    Reads a .json.gz file, but produces None if any error occurs.
    """
    try:
        with open(path, "rt") as f:
            return json.load(f)
    except Exception as e:
        print(e)
        return None


def gzip_json(path: Path, data: dict) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


"""

This script calculates pass@k. It receives a list of directories as its
argument, and calculates the mean pass@k for the set of problems in each
directory. It checks that all results in a directory were generated at the same
temperature. It calculates pass@1 for temperature 0.2 and both pass@10 and
pass@100 for temperature 0.8.

The output has the following columns:

- Dataset: the name of a directory
- Pass@k: the value of k
- Estimate: the mean pass@k for the problems in the directory
- NumProblems: the number of problems in the directory
- MinCompletions: the minimum number of completions for any problem in the 
  directory
- MaxCompletions: the maximum number of completions for any problem in the
  directory
"""


def estimator(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def for_file(path):
    data = open_json(path)
    if data is None:
        return None
    n = len(data["results"])
    c = len([
        True for r in data["results"] if r["status"] == "OK" and r["exit_code"] == 0
    ])
    return {
        "pass@1": estimator(n, c, 1),
        "pass@10": estimator(n, c, 10),
        "pass@100": estimator(n, c, 100),
        "n": n,
        "c": c,
        "temperature": data["temperature"] if "temperature" in data else 0.2,
    }


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
            metrics_copy[k] = f"{v >> 20}MB"
        elif k == "total_flos":
            metrics_copy[k] = f"{int(v) >> 30}GF"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suppress-header", action="store_true", help="Suppress the header"
    )
    parser.add_argument("-k", type=int, default=None, help="The value of k")
    parser.add_argument("dirs", type=str, help="Directories with results. ", nargs="+")
    args = parser.parse_args()
    if not args.suppress_header:
        print("Dataset,Pass@k,Estimate,NumProblems,MinCompletions,MaxCompletions")
    for d in args.dirs:
        results = [
            for_file(p)
            for p in itertools.chain(
                Path(d).glob("*.results.json"), Path(d).glob("*.results.json.gz")
            )
        ]
        results = [r for r in results if r is not None]
        name = d.split("/")[-1] if d.split("/")[-1] != "" else d.split("/")[-2]
        temperatures = set(r["temperature"] for r in results)
        if len(temperatures) != 1:
            eprint(f"Found multiple temperatures {temperatures} in {d} {results}")
            continue
        temperature = list(temperatures)[0]
        num_problems = len(results)
        min_completions = np.min([r["n"] for r in results])
        max_completions = np.max([r["n"] for r in results])
        if temperature == 0.8:
            pass_10 = np.mean([r["pass@10"] for r in results])
            pass_100 = np.mean([r["pass@100"] for r in results])
            print(
                f"{name},10,{pass_10},{num_problems},{min_completions},{max_completions}"
            )
            print(
                f"{name},100,{pass_100},{num_problems},{min_completions},{max_completions}"
            )
        else:
            pass_1 = np.mean([r["pass@1"] for r in results])
            print(
                f"{name},1,{pass_1},{num_problems},{min_completions},{max_completions}"
            )

            results_out = {
                "humaneval_pass@1": pass_1,
                "humaneval_pass@1_count": pass_1 * num_problems,
            }

            log_metrics(name, results_out)
            parent_dir = os.path.dirname(d)
            parent_dir = os.path.dirname(parent_dir)
            save_metrics(name, results_out, parent_dir)

        if args.k is not None:
            pass_k = np.mean([estimator(r["n"], r["c"], args.k) for r in results])
            print(
                f"{name},{args.k},{pass_k},{num_problems},{min_completions},{max_completions}"
            )


if __name__ == "__main__":
    main()
