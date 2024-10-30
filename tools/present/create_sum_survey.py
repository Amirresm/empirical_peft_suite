import argparse
import gzip
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable
import csv
import random


@dataclass()
class ConfigMeta:
    remark: str
    job: str
    model: str
    dataset: str
    peft_lib: str
    peft: str

    _dirname: str = ""

    def __post_init__(self):
        self._dirname = f"{self.remark}_{self.job}_{self.model}_{self.dataset}_{self.peft_lib}_{self.peft}"

    def get_path(self, base_path: str):
        return os.path.join(
            base_path, self.job, self.model, self.dataset, self._dirname
        )

    def get_dirname(self):
        return self._dirname

    def __str__(self):
        return self._dirname

    @staticmethod
    def from_dirname(dirname: str):
        match = re.search("(spp.*k)", dirname)
        original = None
        if match:
            matched = match.group(0)
            original = matched
            dirname = dirname.replace(matched, matched.replace("_", "-"))
        splits = dirname.split("_")
        if len(splits) != 6:
            return None
        parts = {
            "remark": splits[0],
            "job": splits[1],
            "model": splits[2],
            "dataset": original or splits[3],
            "peft_lib": splits[4],
            "peft": splits[5],
        }
        return ConfigMeta(**parts)


def scan_dirtree(dir: str, filter_dataset_name: str) -> list[ConfigMeta]:
    configs: list[ConfigMeta] = []
    for job_path in os.scandir(dir):
        job_path = job_path.path
        # job_name = job_path.split("/")[-1]

        for model_path in os.scandir(job_path):
            model_path = model_path.path
            # model_name = model_path.split("/")[-1]

            for dataset_path in os.scandir(model_path):
                dataset_path = dataset_path.path
                dataset_name = dataset_path.split("/")[-1]
                if filter_dataset_name == dataset_name:
                    # batch = []
                    # name = f"{job_name}_{model_name}_{dataset_name}"

                    for config_path in os.scandir(dataset_path):
                        config_path = config_path.path
                        config_name = config_path.split("/")[-1]
                        config_meta = ConfigMeta.from_dirname(config_name)
                        if (
                            config_meta is not None
                            and config_meta.remark in ["norm"]
                            and config_meta.peft in ["lora"]
                        ):
                            # batch.append(config_meta)
                            configs.append(config_meta)

                    # configs.append((name, batch))

    return configs


def read_generations(file, line_limit=1000000):
    lines = file.readlines()
    buffer_dict: Dict | None = None
    beginning_regex = re.compile(r"\d+=========")
    cursor = None

    out_list = []

    keys = [
        ("->Example:", "example"),
        ("->Prompt:", "prompt"),
        ("->Target:", "target"),
        ("->Pred:", "pred"),
        ("->Output:", "output"),
        ("->Inp_Tokens:", "inp_Tokens"),
        ("->Out_Tokens:", "out_Tokens"),
        ("->New_Tokens:", "new_Tokens"),
    ]

    for line in lines:
        line_limit -= 1
        if line_limit == 0:
            break
        if beginning_regex.match(line):
            if buffer_dict is None:
                buffer_dict = {}
                continue
            else:
                out_list.append(buffer_dict)
                buffer_dict = {}
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
    out_list.append(buffer_dict)

    return out_list


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if isinstance(line, str) and any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def save_csv(rows, fields, save_path):
    DELIMITER = "~"
    with open(save_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fields,
            delimiter=DELIMITER,
            quotechar='"',
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({fields[i]: row[i] for i in range(len(fields))})


BASE_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "sum_survey")


model_name_map = {}
_current_model_id = 0


def map_model_name(model_name):
    global _current_model_id
    if model_name in model_name_map:
        return model_name_map[model_name]
    model_name_map[model_name] = f"model_{_current_model_id}"
    _current_model_id += 1
    return model_name_map[model_name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--recursive", type=bool, default=False, required=False)
    args = parser.parse_args()

    base_dir = os.path.abspath(args.input)
    if not os.path.exists(BASE_RESULTS_DIR):
        os.mkdir(BASE_RESULTS_DIR)

    target_datasets = ["csn-python", "rsum-combined"]
    original_datasets = {
        "rsum-combined": "/home/amirreza/projects/ai/data/rsum/Rcombine/test.jsonl",
        "csn-python": "/home/amirreza/projects/ai/data/CodeSearchNet/python/test.jsonl",
    }

    for target_dataset in target_datasets:
        configs = scan_dirtree(args.input, target_dataset)
        is_R = target_dataset == "rsum-combined"
        data_per_model = {}

        sample_count = 25
        random_sample = []
        for c in configs:
            if os.path.exists(
                os.path.join(
                    c.get_path(base_dir),
                    "gen_output",
                    "generated_generations.txt",
                )
            ):
                with open(
                    os.path.join(
                        c.get_path(base_dir),
                        "gen_output",
                        "generated_generations.txt",
                    ),
                    "r",
                ) as source_file:
                    original_samples = list(
                        stream_jsonl(original_datasets[target_dataset])
                    )
                    reference_samples = read_generations(source_file)
                    reference_samples = [
                        {
                            **sample,
                            "original_target": "".join(original["docstring"]),
                            "original_prompt": "".join(original["code_tokens"])
                            if is_R
                            else original["code"],
                            "index": i,
                        }
                        for i, (sample, original) in enumerate(
                            zip(reference_samples, original_samples)
                        )
                    ]
                    random_sample = random.sample(
                        reference_samples,
                        sample_count,
                    )
                    # for sample in random_sample:
                    #     sample["code"] = original_samples[sample["index"]]

                    break

        # for sample in random_sample:
        #     print("====")
        #     print(sample["target"])
        #     print(sample["original_target"])

        for current_config in configs:
            name = str(current_config)
            print(f"Working on {name}")
            source_file_name = os.path.join(
                current_config.get_path(base_dir),
                "gen_output",
                "generated_generations.txt",
            )
            if not os.path.exists(source_file_name):
                print(f"File {source_file_name} does not exist")
                continue
            data_per_model[name] = []
            with open(source_file_name, "r") as source_file:
                generations = read_generations(source_file)
                for random_s in random_sample:
                    data_per_model[name].append(generations[random_s["index"]]["pred"])

        rows = []
        fields = ["index", "original_prompt", "original_target"]
        fields += list(data_per_model.keys())
        for i, s in enumerate(random_sample):
            index = s["index"]
            original_prompt = s["original_prompt"].strip()
            # remove everything between """ and """
            re_prompt = re.compile(r'""".*?"""', re.DOTALL)
            original_prompt = re_prompt.sub("", original_prompt)

            original_target = s["original_target"].strip()
            row = [
                    index,
                    original_prompt,
                    original_target,
            ]
            for model in data_per_model.values():
                row.append(model[i].strip())
            rows.append(row)
        save_csv(
            rows,
            fields,
            os.path.join(BASE_RESULTS_DIR, f"{target_dataset}_survey.csv"),
        )

        # model_map_rows = [[name, id] for name, id in model_name_map.items()]
        # save_csv(
        #     model_map_rows,
        #     ["model", "id"],
        #     os.path.join(BASE_RESULTS_DIR, f"{target_dataset}_model_map.txt"),
        # )


if __name__ == "__main__":
    main()
