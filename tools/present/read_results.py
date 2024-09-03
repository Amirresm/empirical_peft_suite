import os
import re
import json
import csv
from argparse import ArgumentParser


datasets = set()
configs = {}

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

    def __str__(self):
        if not self.valid:
            return f"Invalid Config: {self.name}"
        if not self.has_results:
            return f"{self.meta.remark}_{self.meta.job}_{self.meta.model}_{self.meta.dataset}_{self.meta.peft_lib}_{self.meta.peft}"
        return f"{self.meta.remark}_{self.meta.job}_{self.meta.model}_{self.meta.dataset}_{self.meta.peft_lib}_{self.meta.peft}: {len(self.all_results)}"


def scan_gen(dir: str):
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

def get_csv(dataset, configs):
    ds_configs = [c for c in configs.values() if c.meta.dataset == dataset]
    fixed_fields = ["remark", "job", "model", "dataset", "peft_lib", "peft"]
    all_fields = set()
    for c in ds_configs:
        all_fields.update([f for f in c.all_results.keys() if f not in fixed_fields])
    all_fields = fixed_fields + list(all_fields)

    with open(f"{dataset}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows([c.all_results for c in ds_configs])

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir", type=str, required=True, help="The root directory of the results"
    )
    args = parser.parse_args()

    dir = args.root_dir

    scan_gen(os.path.join(dir, "gen"))

    get_csv("spp_30k", configs)


if __name__ == "__main__":
    main()
