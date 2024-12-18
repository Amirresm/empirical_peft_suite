import re
import os
import json
import argparse
from typing import Dict
import tqdm
from codebleu import calc_codebleu
from evaluate import load

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


def save_metrics(split, metrics, output_dir, task, combined=True):
    task = "codebleu" if task == "codebleu" else "exactmatch"
    output_path = os.path.join(output_dir, f"{split}_{task}_results.json")
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


def custom_exact_match(preds, targets):
    strict_match = 0
    exact_match = 0
    loose_match = 0
    total = 0
    for pred, target in zip(preds, targets):
        total += 1
        if pred == target:
            strict_match += 1
        if pred.strip() == target.strip():
            exact_match += 1
        if pred.lower().strip() == target.lower().strip():
            loose_match += 1

    print(f"Total: {total}")
    print(f"Strict Match: {strict_match}")
    print(f"Exact Match: {exact_match}")
    print(f"Loose Match: {loose_match}")
    return {
        "strict_match": strict_match / total,
        "exact_match": exact_match / total,
        "loose_match": loose_match / total,
    }


def scan_dir(dir: str, task, total=None):
    current = 0
    datasets = set()
    jobs = []
    configs = {}
    gen_job_datasets = ["spp_30k", "sppu_30k", "multiplt-r"]
    # gen_job_datasets = ["spp_30k", "sppu_30k"]
    for model in os.scandir(dir):
        model = model.path
        # model_name = model.split("/")[-1]
        # print(f"Model: {model_name}")

        for dataset in os.scandir(model):
            dataset = dataset.path
            dataset_name = dataset.split("/")[-1]
            datasets.add(dataset_name)
            # print(f"    Dataset: {dataset_name}")

            for config_path in os.scandir(dataset):
                config_path = config_path.path
                if dataset_name in gen_job_datasets:
                    config_name = config_path.split("/")[-1]
                    if re.search("(spp.*k)", config_name):
                        matched = re.search("(spp.*k)", config_name).group(0)
                        config_name = config_name.replace(
                            matched, matched.replace("_", "-")
                        )
                    splits = config_name.split("_")
                    process_path = os.path.join(
                        config_path, "gen_output", "generated_generations.txt"
                    )
                    if (
                        os.path.exists(process_path)
                        and len(splits) == 6
                        and all(splits)
                        and splits[0] != "infermodea"
                    ):
                        current += 1
                        split = "generate"
                        results_path = os.path.join(
                            config_path,
                            f"{split}_{"codebleu" if task == "codebleu" else "exactmatch"}_results.json",
                        )
                        if os.path.exists(results_path):
                            print(f"Omitting {config_path}: already processed")
                            continue
                        if dataset_name == "multiplt-r" and task == "codebleu":
                            print(f"Omitting {config_path}: codebleu is not supported for R")
                            continue
                        else:
                            print(f"Processing {process_path}")
                            jobs.append((process_path, task, config_path))

    for process_path, task, config_path in tqdm.tqdm(jobs):
        print(f"Processing {process_path}")
        do_codebleu(process_path, task)

    return configs, datasets, current


def do_codebleu(dir, task):
    with open(dir, "r") as file:
        file_name = os.path.basename(dir)
        split = file_name.split(".")[0]
        split = split.split("_")[-1]
        if split == "predictions":
            split = "predict"
        elif split == "generations":
            split = "generate"
        else:
            split = f"ukn_{file_name.split('.')[0]}"
        parent_dir = os.path.dirname(dir)
        parent_dir = os.path.dirname(parent_dir)
        preds, targets = read_generations_from_file2(file)

        exact_match_metric = load("exact_match")
        import time

        start = time.time()
        try:
            print("EXAMPLE =============================================================")
            print("PRED: ===============================================================")
            print(preds[33])
            print("TRGT: ===============================================================")
            print(targets[33])
            print("=====================================================================")
            if task == "codebleu":
                results = calc_all_metrics(preds, targets, split)
            elif task == "exact_match":
                results = results = exact_match_metric.compute(
                    predictions=preds, references=targets
                )
                custom_results = custom_exact_match(preds, targets)
                res = {}
                for k, v in results.items():
                    res[f"{split}_EM_{k}"] = v
                for k, v in custom_results.items():
                    res[f"{split}_EM_c_{k}"] = v
                results = res
            else:
                return

            log_metrics(split, results)
            save_metrics(split, results, parent_dir, task)

        except Exception as e:
            print(f"Failed, Error: {e}")

        end = time.time()
        print(f"Time taken: {end - start}")


def read_generations_from_file2(file, line_limit=1000000):
    bar = tqdm.tqdm()
    lines = file.readlines()
    buffer_dict: Dict | None = None
    beginning_regex = re.compile(r"\d+=========")
    cursor = None
    preds = []
    refs = []
    for line in lines:
        line_limit -= 1
        if line_limit == 0:
            break
        if beginning_regex.match(line):
            if buffer_dict is None:
                buffer_dict = {}
                continue
            else:
                preds.append(buffer_dict[measure])
                refs.append(buffer_dict["target"])
                buffer_dict = {}
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
    weights = (0.25, 0.25, 0.25, 0.25)
    cb_results = calc_codebleu(
        [[label] for label in labels], preds, lang="python", weights=weights
    )
    custom_codebleu = (
        +weights[0] * cb_results["ngram_match_score"]
        + weights[1] * cb_results["weighted_ngram_match_score"]
        + weights[2] * cb_results["syntax_match_score"]
        + weights[3] * cb_results["dataflow_match_score"]
    )
    cb_results["codebleu"] = custom_codebleu
    cb_results["codebleuP"] = cb_results["codebleu"] * 100
    result = {**result, **cb_results}

    res = {}
    for k, v in result.items():
        res[f"{split}_CODEBLEU_{k}"] = v

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--recursive", type=bool, default=False, required=False)
    args = parser.parse_args()

    if args.recursive:
        scan_dir(args.input, "exact_match")
        scan_dir(args.input, "codebleu")

    else:
        do_codebleu(args.input, "exact_match")
        do_codebleu(args.input, "codebleu")


if __name__ == "__main__":
    main()
