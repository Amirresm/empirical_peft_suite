import os
import json
import argparse
from typing import Dict
import tqdm
from codebleu import calc_codebleu

keys = [
    ("->Prompt:", "prompt"),
    ("->Target:", "target"),
    ("->Pred:", "pred"),
    ("->Output:", "output"),
]

measure = "pred"


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
    output_path = os.path.join(output_dir, f"{split}_codebleu_results.json")
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
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    with open(args.input, "r") as file:
        file_name = os.path.basename(args.input)
        split = file_name.split(".")[0]
        parent_dir = os.path.dirname(args.input)
        parent_dir = os.path.dirname(parent_dir)
        preds, targets = read_generations_from_file2(file)

        results = calc_all_metrics(preds, targets, split)

        log_metrics(split, results)

        save_metrics(split, results, parent_dir)
        # save results next to args.input
        # with open(args.input + ".codebleu_results.txt", "w") as file:
        #     for key, value in results.items():
        #         file.write(f"{key}: {value}\n")


def read_generations_from_file2(file, line_limit=1000000):
    bar = tqdm.tqdm()
    lines = file.readlines()
    buffer_dict: Dict | None = None
    cursor = None
    preds = []
    refs = []
    for line in lines:
        line_limit -= 1
        if line_limit == 0:
            break
        if "=========" in line:
            if buffer_dict is None:
                buffer_dict = {}
                continue
            else:
                preds.append(buffer_dict[measure])
                refs.append(buffer_dict["target"])
                bar.update(1)
                bar.set_description(f"Processed {len(preds)}")
                continue
        for key, value in keys:
            if line.startswith(key):
                cursor = value
                continue
        if line.startswith("->"):
            continue
        if buffer_dict is not None and cursor is not None:
            if cursor in buffer_dict:
                buffer_dict[cursor] += line
            else:
                buffer_dict[cursor] = line

    return preds, refs


def calc_all_metrics(preds, labels, split):
    result = {}
    cb_results = calc_codebleu([[label] for label in labels], preds, lang="python")
    cb_results["codebleuP"] = cb_results["codebleu"] * 100
    result = {**result, **cb_results}

    res = {}
    for k, v in result.items():
        res[f"{split}_{k}"] = v

    return res


if __name__ == "__main__":
    main()
