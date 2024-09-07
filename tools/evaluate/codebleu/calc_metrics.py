import os
import json
import argparse
from typing import Dict, Optional
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

def scan_dir(dir: str):
    datasets = set()
    configs = {}
    gen_job_datasets = ["spp_30k", "sppu_30k", "multiplt-r"]
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
                    process_path = os.path.join(config_path, "gen_output", "generated_generations.txt")
                    if os.path.exists(process_path):
                        print(f"Processing {process_path}")
                        do_codebleu(process_path)
                    process_path = os.path.join(config_path, "gen_output", "generated_predictions.txt")
                    if os.path.exists(process_path):
                        print(f"Processing {process_path}")
                        do_codebleu(process_path)
                

    return configs, datasets

def do_codebleu(dir):
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

        try:
            #time this for me

            import time
            start = time.time()
            results = calc_all_metrics(preds, targets, split)
            end = time.time()
            print(f"Time taken: {end - start}")

            log_metrics(split, results)
            save_metrics(split, results, parent_dir)

            # batch_size = 5
            # batch_count = len(preds) // batch_size if len(preds) % batch_size == 0 else len(preds) // batch_size + 1
            # bresults_list = []
            # pbar = tqdm.tqdm(total=batch_count)
            # for i in range(batch_count):
            #     bpreds = preds[i * batch_size: (i + 1) * batch_size]
            #     btargets = targets[i * batch_size: (i + 1) * batch_size]
            #     bresults = calc_all_metrics(bpreds, btargets, split)
            #     bresults_list.append(bresults)
            #     pbar.update(1)

            # results = {}
            # for k, _ in bresults_list[0].items():
            #     for bresults in bresults_list:
            #         if k in results:
            #             results[k] += bresults[k]
            #         else:
            #             results[k] = bresults[k]
            # for k, v in results.items():
            #     results[k] = v / len(bresults_list)
            # 
            # log_metrics(f"batch_{split}", results)

            # batch_size = 5
            # batch_count = len(preds) // batch_size if len(preds) % batch_size == 0 else len(preds) // batch_size + 1
            # bresults_list = []
            # pbar = tqdm.tqdm(total=batch_count)
            # for i in range(batch_count):
            #     bpreds = preds[i * batch_size: (i + 1) * batch_size]
            #     btargets = targets[i * batch_size: (i + 1) * batch_size]
            #     bresults = calc_all_metrics(bpreds, btargets, split)
            #     bresults_list.append(bresults)
            #     results = {}
            #     for k, _ in bresults_list[0].items():
            #         for bresults in bresults_list:
            #             if k in results:
            #                 results[k] += bresults[k]
            #             else:
            #                 results[k] = bresults[k]
            #     for k, v in results.items():
            #         results[k] = v / len(bresults_list)
            #     score_so_far = results[f"{split}_CODEBLEU_codebleuP"]
            #     pbar.update(1)
            #     pbar.set_description(f"Processed {i} - {score_so_far:.2f}")

            # 
            # log_metrics(f"batch_{split}", results)

            # batch_size = 5
            # batch_count = len(preds) // batch_size if len(preds) % batch_size == 0 else len(preds) // batch_size + 1
            # results = None
            # pbar = tqdm.tqdm(total=batch_count)
            # for i in range(batch_count):
            #     bpreds = preds[i * batch_size: (i + 1) * batch_size]
            #     btargets = targets[i * batch_size: (i + 1) * batch_size]
            #     bresults = calc_all_metrics(bpreds, btargets, split)
            #     if results is None:
            #         results = bresults
            #     else:
            #         for k, v in bresults.items():
            #             results[k] += v
            #             results[k] /= 2
            #     score = results[f"{split}_CODEBLEU_codebleu"]
            #     pbar.update(1)
            #     pbar.set_description(f"Processed {i} - {score}")
            # 
            # log_metrics(f"batch_{split}", results)
            # save_metrics(split, results, parent_dir)
        except Exception as e:
            print(f"Failed, Error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--recursive", type=bool, default=False, required=False)
    args = parser.parse_args()

    if args.recursive:
        scan_dir(args.input)

    else:
        do_codebleu(args.input)

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
    weights = (0.25, 0.25, 0.25, 0.25)
    cb_results = calc_codebleu([[label] for label in labels], preds, lang="python", weights=weights)
    custom_codebleu = (
        + weights[0] * cb_results["ngram_match_score"]
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


if __name__ == "__main__":
    main()
