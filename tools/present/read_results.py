import os
import re
import json
import csv
from argparse import ArgumentParser
from pathlib import Path


class ConfigMeta:
    def __init__(self, remark, job, model, dataset, peft_lib, peft):
        self.remark = remark
        self.job = job
        self.model = model
        self.dataset = dataset
        self.peft_lib = peft_lib
        self.peft = peft

    def to_name(self):
        return f"{self.remark}_{self.job}_{self.model}_{self.dataset}_{self.peft_lib}_{self.peft}"


class Config:
    def __init__(self, config_path: str):
        self.valid = False
        self.has_results = False

        self.path = config_path
        self.name = config_path.split("/")[-1]

        self.meta = ConfigMeta(None, None, None, None, None, None)

        self.all_results = {}

        self.parse()
        self.read_all_results()

    def parse(self):
        name = self.name
        sppds_regex = "(spp.*k)"
        original_ds = None
        if re.search(sppds_regex, name):
            matched = re.search(sppds_regex, name).group(0)
            original_ds = matched
            ds_name = matched.replace("_", "-")
            name = name.replace(matched, ds_name)

        splits = name.split("_")
        if len(splits) not in [6] or not all(splits):
            return
        self.valid = True
        if original_ds:
            splits[3] = original_ds
        self.meta = ConfigMeta(*splits)

    def read_all_results(self):
        if not self.valid:
            return
        all_results_path = os.path.join(self.path, "all_results.json")
        other_results_paths = [p for p in Path(self.path).glob("*_results.json")]
        all_results = {}
        if os.path.exists(all_results_path):
            with open(all_results_path, "r") as f:
                all_results = f.read()

            try:
                all_results = json.loads(all_results)
                self.has_results = True
                self.all_results = all_results
                self.all_results = {**self.all_results, **self.meta.__dict__}
            except Exception as e:
                print(f"            Error: {e}")

        for other_result in other_results_paths:
            with open(other_result, "r") as f:
                other_results = f.read()
                try:
                    other_results = json.loads(other_results)
                    self.all_results = {**self.all_results, **other_results}
                except Exception as e:
                    print(f"            Error: {e}")

    def __str__(self):
        if not self.valid:
            return f"Invalid Config: {self.name}"
        if not self.has_results:
            return f"{self.meta.remark}_{self.meta.job}_{self.meta.model}_{self.meta.dataset}_{self.meta.peft_lib}_{self.meta.peft}"
        return f"{self.meta.remark}_{self.meta.job}_{self.meta.model}_{self.meta.dataset}_{self.meta.peft_lib}_{self.meta.peft}: {len(self.all_results)}"


def scan_dir(dir: str):
    datasets = set()
    configs = {}
    # models = [p for p in Path(dir).glob("*.results.json")]
    for model in os.scandir(dir):
        model = model.path
        model_name = model.split("/")[-1]
        # print(f"Model: {model_name}")

        for dataset in os.scandir(model):
            dataset = dataset.path
            dataset_name = dataset.split("/")[-1]
            datasets.add(dataset_name)
            # print(f"    Dataset: {dataset_name}")

            for config_path in os.scandir(dataset):
                config_path = config_path.path
                config = Config(config_path)

                if config.has_results:
                    # print(f"        {config}")
                    configs[config.name] = config

    return configs, datasets


def get_fields(configs):
    fixed_fields = [
        "job",
        "dataset",
        "remark",
        "model",
        "peft_lib",
        "peft",
    ]
    field_groups = [
        "humaneval",
        "generate",
        "predict",
        "eval",
        "train",
    ]
    field_subgroups = [
        "BLEU2",
        "ROUGE",
        "BLEU",
        "loss",
        "pass@",
        "total_gpu_time",
        "runtime",
        "peak_memory",
        "samples",
    ]
    filters = [
        r"^.*_BLEU.?_(precisions|bleu|brevity_penalty|length_ratio)$",
        # r"^.*_BLEU.?_((reference|translation)_length)$",
        # r"^.*ROUGE.*$",
        r"^.*per_second.*$",
    ]
    all_fields = set()
    for c in configs:
        all_fields.update([f for f in c.all_results.keys() if f not in fixed_fields])
    main_fields = []
    all_fields = sorted(all_fields)
    for fg in field_groups:
        fg_fields = [f for f in all_fields if f.startswith(fg)]
        fg_fields_sorted = []
        for sg in field_subgroups:
            sg_fields = [f for f in fg_fields if sg in f]
            for f in sg_fields:
                if f not in fg_fields_sorted:
                    fg_fields_sorted.append(f)
            for f in fg_fields:
                if f not in fg_fields_sorted:
                    fg_fields_sorted.append(f)

        for f in fg_fields_sorted:
            if f not in main_fields:
                main_fields.append(f)

    for f in all_fields:
        if f not in main_fields:
            main_fields.append(f)

    main_fields = [
        f
        for f in main_fields
        if not any([bool(re.search(filter, f)) for filter in filters])
    ]

    all_fields = fixed_fields + main_fields

    return all_fields


def sorted_rows(configs, fields):
    sort_by = [
        "job",
        "dataset",
        "model",
        "remark",
        "peft_lib",
        "peft",
    ]
    for by in reversed(sort_by):
        if by[0] == "-":
            by = by[1:]
            configs.sort(key=lambda x: x[by], reverse=True)
        configs.sort(key=lambda x: x[by])

    configs = list(map(lambda c: {k: c[k] for k in c if k in fields}, configs))
    return configs


def get_csv(dataset, configs):
    ds_configs = [c for c in configs.values() if c.meta.dataset == dataset]

    all_fields = get_fields(ds_configs)

    rows = sorted_rows([c.all_results for c in ds_configs], all_fields)

    with open(f"{dataset}.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)


def run_all(dir, dataset):
    job_path = "sum" if dataset == "csn-python" else "gen"
    configs, datasets = scan_dir(os.path.join(dir, job_path))

    get_csv(dataset, configs)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="The root directory of the results",
    )
    args = parser.parse_args()

    dir = args.root_dir

    run_all(dir, "spp_30k")
    run_all(dir, "csn-python")
    run_all(dir, "sppu_30k")
    run_all(dir, "multiplt-r")


if __name__ == "__main__":
    main()
