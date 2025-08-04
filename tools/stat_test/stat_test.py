import argparse
import csv
import os
import random
import json

from tqdm import tqdm

from fs_tools import (
    Task,
    read_generations_from_file,
    save_p_values_csv,
    walk_dataset_dir,
    walk_parent_dir,
)
from metrics import bleu
from stat_eval import evaluate_results

BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

sampling_index_map = {}


def calc_metrics(
    task: Task,
    batch_size,
    data_dir,
    shuffle=False,
    shuffle_count: int | None = None,
):
    metrics = {"exact_match": [], "bleu": [], "codebleu": []}
    results = {"batch_size": batch_size, "metrics": metrics}

    should_do_codebleu = task.dataset in ["spp-30k", "sppu-30k"]

    parent_path = os.path.join(
        data_dir, f"{task.jtype}_{task.dataset}_{task.model}"
    )
    if shuffle:
        file_name = f"{task.remark}_{task.peft_lib}_{task.peft}_shuff-{batch_size}_{shuffle_count}.csv"
    else:
        file_name = (
            f"{task.remark}_{task.peft_lib}_{task.peft}_{batch_size}.csv"
        )
    csv_path = os.path.join(parent_path, file_name)
    if os.path.exists(csv_path):
        with open(csv_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for key, value in row.items():
                    metrics[key].append(float(value))

            batch_count = len(metrics["exact_match"])
            results["batch_count"] = batch_count

    else:
        with open(task.data_path, "r") as file:
            preds, targets = read_generations_from_file(file)
            assert len(preds) == len(
                targets
            ), "Length of predictions and targets should be the same"
            if len(preds) == 0:
                print(f"No predictions found for {task.config_name}")
                return

            if should_do_codebleu:
                metrics["codebleu"] = []

            batch_count: int = (
                shuffle_count
                if shuffle and shuffle_count is not None
                else len(preds) // batch_size
            )
            results["batch_count"] = batch_count

            if shuffle:
                tqdm_desc = f"{task.config_name} | shuffle {shuffle_count}x{batch_size}={shuffle_count * batch_size}"
            else:
                tqdm_desc = f"{task.config_name} | {batch_count}x{batch_size}={batch_count * batch_size}"
            for i in tqdm(
                range(batch_count),
                desc=tqdm_desc,
                leave=False,
            ):
                if shuffle and shuffle_count is not None:
                    group_key = f"{task.jtype}_{task.dataset}_{task.model}_{i}"
                    if group_key in sampling_index_map:
                        sample_indices = sampling_index_map[group_key]
                    else:
                        sampling_index_map[group_key] = random.choices(
                            range(len(preds)), k=batch_size
                        )
                        sample_indices = sampling_index_map[group_key]
                    for j in sample_indices:
                        try:
                            assert j < len(preds)
                        except Exception as e:
                            print(f"Error: {e}")
                            print(f"j: {j}", f"len(preds): {len(preds)}")
                    batch_preds = [preds[j] for j in sample_indices]
                    batch_targets = [targets[j] for j in sample_indices]
                else:
                    start = i * batch_size
                    end = (i + 1) * batch_size
                    batch_preds = preds[start:end]
                    batch_targets = targets[start:end]

                bleu_score = bleu(batch_preds, batch_targets)
                metrics["bleu"].append(bleu_score)

        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
        with open(csv_path, "w", newline="") as csvfile:
            fields = [k for k in metrics.keys() if len(metrics[k]) > 0]
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            for i in range(len(metrics[fields[0]])):
                writer.writerow({key: metrics[key][i] for key in fields})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--recursive", type=bool, default=False, required=False)
    parser.add_argument(
        "--target_datasets", type=str, default=None, required=False
    )
    args = parser.parse_args()

    target_datasets = None
    if args.target_datasets is not None:
        target_datasets = args.target_datasets.split(",")

    batch_size = 100
    shuffle = True
    shuffle_count = 1000

    if target_datasets is not None:
        sample_dir = os.path.join(
            BASE_DATA_DIR,
            f"{args.target_datasets}_bs{batch_size}_s{shuffle}_sc{shuffle_count}",
        )
    else:
        sample_dir = os.path.join(
            BASE_DATA_DIR, f"bs{batch_size}_s{shuffle}_sc{shuffle_count}"
        )

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    global sampling_index_map
    sampling_index_map_path = os.path.join(
        sample_dir, "sampling_index_map.json"
    )
    if os.path.exists(sampling_index_map_path):
        with open(sampling_index_map_path, "r") as file:
            sampling_index_map = json.load(file)

    if args.recursive:
        batched_tasks = walk_parent_dir(
            args.input, target_datasets=target_datasets
        )

    else:
        batched_tasks = [
            walk_dataset_dir(args.input, target_datasets=target_datasets)
        ]

    data_dir = os.path.join(sample_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    final_summaries = {
        "exact_match": ([], []),
        "bleu": ([], []),
        "codebleu": ([], []),
    }

    all_p_values = {}

    bar = tqdm(total=len(batched_tasks))
    for tasks in batched_tasks:
        group_results = []
        group_name = f"{tasks[0].jtype}_{tasks[0].dataset}_{tasks[0].model}"
        bar.set_description(f"Processing {group_name}")
        for task in tasks:
            results = calc_metrics(
                task,
                batch_size=batch_size,
                data_dir=data_dir,
                shuffle=shuffle,
                shuffle_count=shuffle_count,
            )

            with open(sampling_index_map_path, "w") as file:
                json.dump(sampling_index_map, file)

            if results is not None:
                results["task"] = task
                group_results.append(results)

        summaries, stat_results, p_values = evaluate_results(group_results)

        print("")
        print("=" * 100)
        print(f"Results for {group_name}")
        for summary in summaries:
            print(
                f"-> {(summary['name'] + ':').ljust(20, ' ')}\t{'\t'.join([f'{metric} = {data['average']:.3f} ({'normal' if data['normal'] else 'not-norm'})' for metric, data in summary['scores'].items()])}"
            )
        print(p_values)
        all_p_values[group_name] = {
            "metadata": {
                "job": tasks[0].jtype,
                "dataset": tasks[0].dataset,
                "model": tasks[0].model,
            },
            "p_values": p_values,
        }
        print("")
        for metric, data in stat_results.items():
            anova = data["anova"]
            ttest_failed = data["ttest_failed"]
            if anova:
                if len(ttest_failed) == 0:
                    print(
                        f"-> {metric}:  All baseline differences are statistically significant"
                    )
                    final_summaries[metric][0].append(group_name)
                else:
                    print(
                        f"-> {metric}:  Some baseline differences are not statistically significant"
                    )
                    final_summaries[metric][1].append(group_name)
                    print("Failed comparisons:")
                    for f in ttest_failed:
                        print("\t" + f)
            else:
                print(f"-> {metric}:  ANOVA test failed")
                final_summaries[metric][1].append(group_name)

        bar.update(1)

    print("")
    print("=" * 100)
    print("Saving p-values to csv")
    save_p_values_csv(sample_dir, all_p_values)
    for metric, (passed, failed) in final_summaries.items():
        print("")
        print("=" * 100)
        print(f"Final Results for {metric}")
        print(f"Passed: {len(passed)}")
        print(f"Failed: {len(failed)}")
        print("Failed groups:")
        for f in failed:
            print("\t" + f)


if __name__ == "__main__":
    main()
