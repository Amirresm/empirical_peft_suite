from dataclasses import dataclass
from typing import Dict
import os
import re


class DatasetInstances:
    SPP = "spp"
    CSN = "csn"
    MULTIPLT = "multiplt"
    UNKNOWN = "unknown"


def find_nth(haystack: str, needle: str, n: int) -> int:
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start


def join_prefix_prompt(prefix, prompt):
    return f"{prefix}{prompt}"


# ====== CSN
def csn_create_prompt(input):
    return f"{input}\n# summary:\n"


def csn_join(prompt, completion):
    return f"{prompt}{completion}"


def csn_split(input):
    splits = input.split("# summary:")
    prompt = splits[0]
    if len(splits) == 1:
        return input, input
    completion = splits[1]
    return prompt, completion


# ====== SPP
def spp_create_prompt(input):
    return f"{input}"


def spp_join(prompt, completion):
    return f"{prompt}{completion}"


def spp_split(input):
    idx_split = find_nth(input, '"""', 2) + 3
    if idx_split == -1:
        return input, input
    prompt = input[:idx_split]
    completion = input[idx_split:]
    return prompt, completion


# ====== MULTIPLT
def multiplt_create_prompt(input):
    return f"{input}"


def multiplt_join(prompt, completion):
    return f"{prompt}{completion}"


def multiplt_split(input):
    content = input
    comment = ""
    signature = ""
    code = ""
    for line in content.splitlines():
        if line.startswith("#") and not signature:
            comment += line + "\n"
        else:
            if not signature:
                signature = line + "\n"
            else:
                code += line + "\n"
    prompt = comment + signature
    completion = code
    return prompt, completion


def modify_prompt(prompt, ds_type):
    match ds_type:
        case DatasetInstances.CSN:
            return csn_create_prompt(prompt)
        case DatasetInstances.SPP:
            return spp_create_prompt(prompt)
        case DatasetInstances.MULTIPLT:
            return multiplt_create_prompt(prompt)
        case _:
            return prompt


def join_columns(prompt, completion, ds_type):
    match ds_type:
        case DatasetInstances.CSN:
            return csn_join(prompt, completion)
        case DatasetInstances.SPP:
            return spp_join(prompt, completion)
        case DatasetInstances.MULTIPLT:
            return multiplt_join(prompt, completion)
        case _:
            return prompt


def split_column(example, ds_type):
    match ds_type:
        case DatasetInstances.CSN:
            return csn_split(example)
        case DatasetInstances.SPP:
            return spp_split(example)
        case DatasetInstances.MULTIPLT:
            return multiplt_split(example)
        case _:
            return example, ""


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


def scan_results_dir_for_datasets(dir: str) -> list[tuple[str, list[ConfigMeta]]]:
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
