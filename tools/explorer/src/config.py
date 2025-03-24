import os
import re
from dataclasses import dataclass


@dataclass
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
        return os.path.join(base_path, self.job, self.model, self.dataset, self._dirname)

    def get_dirname(self):
        return self._dirname

    def __str__(self):
        return self._dirname

    def to_dict(self):
        return {
            "remark": self.remark,
            "job": self.job,
            "model": self.model,
            "dataset": self.dataset,
            "peft_lib": self.peft_lib,
            "peft": self.peft,
        }

    @staticmethod
    def from_dict(data: dict):
        return ConfigMeta(**data)

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
