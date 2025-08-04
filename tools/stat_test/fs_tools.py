import os
from typing import Dict
import re
import csv

keys = [
    ("->Example:", "example"),
    ("->Prompt:", "prompt"),
    ("->Target:", "target"),
    ("->Pred:", "pred"),
    ("->Output:", "output"),
    ("->Inp_Tokens:", "inp_tokens"),
    ("->Out_Tokens:", "out_tokens"),
    ("->New_Tokens:", "new_tokens"),
]

measure = "pred"


class Task:
    @staticmethod
    def get_data_path(config_path: str):
        return os.path.join(
            config_path, "gen_output", "generated_generations.txt"
        )

    def __init__(
        self,
        config_splits: list[str],
        config_path: str,
        config_name: str,
        data_path: str,
    ):
        self.remark = config_splits[0]
        self.jtype = config_splits[1]
        self.model = config_splits[2]
        self.dataset = config_splits[3]
        self.peft_lib = config_splits[4]
        self.peft = config_splits[5]
        self.config_path = config_path
        self.config_name = config_name
        self.data_path = data_path
        pass


def read_generations_from_file(file, line_limit=1000000):
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
                continue
        for key, value in keys:
            if line.startswith(key):
                cursor = value
                continue
        if buffer_dict is not None and cursor is not None:
            if cursor in buffer_dict:
                buffer_dict[cursor] += line
            else:
                buffer_dict[cursor] = line

    return preds, refs


def walk_dataset_dir(
    dataset_path: str, target_datasets: list[str] | None = None
):
    dataset_name = dataset_path.split("/")[-1]
    jobs = []
    if target_datasets is not None:
        valid_datasets = target_datasets
    else:
        valid_datasets = [
            "spp_30k",
            # "sppu_30k",
            "multiplt-r",
            "csn-python",
            "csn-java",
            "csn-go",
            "csn-php",
            "csn-ruby",
            "csn-javascript",
            "rsum-combined",
        ]
    valid_pefts = [
        ("ah", "compacter"),
        ("pl", "lora"),
        ("pl", "ia3"),
        ("none", "none"),
    ]
    valid_remakes = ["norm", "infer", "full"]
    if dataset_name in valid_datasets:
        for config_path in os.scandir(dataset_path):
            config_path = config_path.path
            config_name = config_path.split("/")[-1]
            if re.search("(spp.*k)", config_name):
                matched = re.search("(spp.*k)", config_name).group(0)
                config_name = config_name.replace(
                    matched, matched.replace("_", "-")
                )
            splits = config_name.split("_")

            data_path = Task.get_data_path(config_path)
            if (
                os.path.exists(data_path)
                and len(splits) == 6
                and all(splits)
                and splits[0] in valid_remakes
                and (splits[4], splits[5]) in valid_pefts
            ):
                task = Task(splits, config_path, config_name, data_path)
                jobs.append(task)
    else:
        print(f"Unexpected dataset: {dataset_name}, skipping...")

    return jobs


def walk_parent_dir(dir: str, target_datasets: list[str] | None = None):
    batched_tasks: list[list[Task]] = []
    for jtype in os.scandir(dir):
        jtype_path = jtype.path
        jtype_name = jtype_path.split("/")[-1]

        if jtype_name not in ["gen", "sum"]:
            raise ValueError(f"Expected 'gen' or 'sum', got {jtype_name}")

        for model_path in os.scandir(jtype_path):
            model_path = model_path.path

            for dataset_path in os.scandir(model_path):
                dataset_path = dataset_path.path

                tasks = walk_dataset_dir(
                    dataset_path, target_datasets=target_datasets
                )
                if len(tasks) > 1:
                    batched_tasks.append(tasks)

    return batched_tasks


def save_p_values_csv(base_path, all_p_values):
    csv_path = os.path.join(base_path, "p_values.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fields = [
            "group",
            "config",
            "task",
            "model",
            "data",
            "peft",
            "metric",
            "p_value",
            "significant",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for group, group_values in all_p_values.items():
            metadata = group_values["metadata"]
            p_values = group_values["p_values"]
            csv_row = {
                "group": group,
                "task": metadata["job"],
                "data": metadata["dataset"],
                "model": metadata["model"],
                "metric": "bleu",
            }

            if "anova_fail" in p_values:
                csv_row["config"] = "all"
                csv_row["p_value"] = "anova_fail"
                csv_row["significant"] = "anova_fail"
                writer.writerow(csv_row)
            else:
                csv_rows = []
                for config, p_value in p_values.items():
                    current_csv_row = csv_row.copy()
                    peft = config.split("-")[-1]
                    current_csv_row["config"] = config
                    current_csv_row["peft"] = peft
                    current_csv_row["p_value"] = p_value
                    current_csv_row["significant"] = (
                        "baseline"
                        if p_value == -1
                        else "yes" if p_value < 0.05 else "no"
                    )
                    csv_rows.append(current_csv_row)
                csv_rows = sorted(
                    csv_rows,
                    key=lambda x: {"n": 0, "c": 1, "l": 2, "i": 3}.get(
                        x["peft"][0], float("inf")
                    ),
                )
                writer.writerows(csv_rows)
