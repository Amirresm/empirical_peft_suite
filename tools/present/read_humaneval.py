import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict
import csv


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
        # /home/amirreza/projects/ai/outputs/peftsuite_results/gen/codellama-7b/spp_30k/infer_gen_codellama-7b_spp_30k_none_none
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


def read_humaneval_python_from_file(file, line_limit=1000000):
    lines = file.readlines()
    buffer_dict: Dict | None = None
    beginning_regex = re.compile(r"\d+=========")
    cursor = None

    out_list = []

    keys = [
        ("->Task:", "task"),
        ("->Passed:", "passed"),
        ("->Result:", "result"),
        ("->Completion:", "completion"),
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

    return out_list


def read_humaneval_r_from_file(dir):
    rows = []
    parent_dir = os.path.dirname(dir)
    problem_files = os.listdir(os.path.join(parent_dir, "humaneval_r_problems_output"))
    completion_files = os.listdir(os.path.join(parent_dir, "humaneval_r_problems"))
    problem_files = sorted(problem_files, key=lambda x: int(x.split("_")[1]))
    completion_files = sorted(completion_files, key=lambda x: int(x.split("_")[1]))
    for problem_file, completion_file in zip(problem_files, completion_files):
        with open(
            os.path.join(parent_dir, "humaneval_r_problems_output", problem_file), "r"
        ) as file:
            problem = json.load(file)
            problem["stdout"] = problem["results"][0]["stdout"]
            problem["stderr"] = problem["results"][0]["stderr"]
            problem["exit_code"] = problem["results"][0]["exit_code"]
            problem["status"] = problem["results"][0]["status"]
        with open(
            os.path.join(parent_dir, "humaneval_r_problems", completion_file), "r"
        ) as file:
            completion = json.load(file)
            completion["completions"] = completion["completions"][0]
        row = {
            **problem,
            **completion,
        }
        rows.append(row)

    return rows


def get_csv(rows, save_path):
    with open(save_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


BASE_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "humaneval")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--recursive", type=bool, default=False, required=False)
    args = parser.parse_args()

    base_dir = os.path.abspath(args.input)

    configs = scan_dirtree(args.input)

    if not os.path.exists(BASE_RESULTS_DIR):
        os.mkdir(BASE_RESULTS_DIR)

    for name, config_batch in configs:
        filter = "norm|infer|full"
        if len(config_batch) == 0:
            continue
        source_file_name = os.listdir(
            os.path.join(config_batch[0].get_path(base_dir), "gen_output")
        )
        source_file_name = [
            f.split("/")[-1]
            for f in source_file_name
            if f.endswith("generated_humaneval.txt") or "humaneval_r_problems" in f
        ]
        if len(source_file_name) == 0:
            continue
        source_file_name = source_file_name[0]
        print(f"Working on {name}")

        separate_rows = {}

        for config in config_batch:
            if config.remark not in filter.split("|"):
                continue
            generated_file = os.path.join(
                config.get_path(base_dir), "gen_output", source_file_name
            )
            header_text = f"{config.peft}({config.peft_lib})({config.remark})"
            if (
                os.path.isdir(generated_file)
                and "humaneval_r_problems" in generated_file
            ):
                out_list = read_humaneval_r_from_file(generated_file)
                separate_rows[header_text] = [
                    (r["name"], r["results"][0]["status"]) for r in out_list
                ]
            elif os.path.exists(generated_file):
                with open(generated_file, "r") as file:
                    out_list = read_humaneval_python_from_file(file)
                separate_rows[header_text] = [
                    (
                        r["task"],
                        "OK"
                        if r["result"].strip() == "passed"
                        else r["result"].strip(),
                    )
                    for r in out_list
                ]
            else:
                print(f"File {generated_file} not found.")

        rows = {}
        for header in separate_rows:
            for row_name, cell in separate_rows[header]:
                row_name = row_name.strip().replace("/", "_")
                if row_name in rows:
                    rows[row_name].update({header: cell})
                else:
                    rows[row_name] = {"task": row_name, header: cell}

        get_csv(list(rows.values()), os.path.join(BASE_RESULTS_DIR, f"{name}.csv"))


if __name__ == "__main__":
    main()
