import difflib
import json
import os
import re
from typing import Dict

import tqdm
from rich.console import Console
from rich.text import Text


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


def process_buffer_dict(index, buffer_dict: Dict, ds_instance):
    if not buffer_dict["input"] or not buffer_dict["gold"] or not buffer_dict["pred"]:
        raise ValueError(f"Buffer dict {index} is missing keys:\n{buffer_dict}")

    prompt = buffer_dict["input"]
    target = buffer_dict["gold"]
    pred = buffer_dict["pred"]

    striped_pred = re.sub(r"^[\r\n]", "", pred)
    striped_target = re.sub(r"^[\r\n]", "", target)
    output = join_columns(prompt, striped_pred, ds_instance)
    example = join_columns(prompt, striped_target, ds_instance)

    inp_tokens = len(prompt.split(" "))
    out_tokens = len(output.split(" "))
    new_tokens = out_tokens - inp_tokens

    out_dict = {
        "index": index,
        "example": example,
        "prompt": prompt,
        "target": target,
        "pred": pred,
        "output": output,
        "inp_tokens": inp_tokens,
        "out_tokens": out_tokens,
        "new_tokens": new_tokens,
    }
    return out_dict


# def read_generations_from_file(file, ds_instance, processed_limit=1000000, total=None):
#     out_list = []
#     bar = tqdm.tqdm(total=total)
#     lines = file.readlines()

#     keys = [
#         ("Pred:", "pred"),
#         ("Gold:", "gold"),
#         ("Input:", "input"),
#     ]

#     cursor = None
#     buffer_dict: Dict | None = None
#     beginning_regex = re.compile(r".*INFO: \d+===.*")
#     end_regex = re.compile(r".*INFO: \d+ generations saved to.*")
#     starting_line_met = False
#     processed = 0
#     for line in lines:
#         if processed_limit == 0:
#             break
#         if "INFO: *** Generate ***" in line:
#             starting_line_met = True
#             continue
#         if end_regex.match(line) and buffer_dict is not None:
#             processed += 1
#             # save buffer to file
#             out_dict = process_buffer_dict(processed, buffer_dict, ds_instance)
#             out_list.append(out_dict)
#             processed_limit -= 1
#             cursor = None
#             buffer_dict = {}
#             bar.update(1)
#             break
#         if not starting_line_met:
#             continue

#         if beginning_regex.match(line):
#             if buffer_dict is None:
#                 buffer_dict = {}
#                 continue
#             else:
#                 processed += 1
#                 # save buffer to file
#                 out_dict = process_buffer_dict(processed, buffer_dict, ds_instance)
#                 out_list.append(out_dict)
#                 processed_limit -= 1
#                 cursor = None
#                 buffer_dict = {}
#                 bar.update(1)
#                 continue

#         new_cursor_set = False
#         for key, value in keys:
#             if line.startswith(key):
#                 cursor = value
#                 new_cursor_set = True
#                 break
#         if new_cursor_set:
#             continue
#         if cursor is None and processed == 0:
#             cursor = "input"

#         if buffer_dict is not None and cursor is not None:
#             if cursor in buffer_dict:
#                 buffer_dict[cursor] += line
#             else:
#                 buffer_dict[cursor] = line

#         else:
#             continue

#     return out_list


def read_generations_from_file(file, line_limit=1000000):
    bar = tqdm.tqdm()
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
                bar.update(1)
                bar.set_description(f"Processed {len(out_list)}")
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
        with open(os.path.join(parent_dir, problem_file), "r") as file:
            problem = json.load(file)
        with open(os.path.join(parent_dir, completion_file), "r") as file:
            completion = json.load(file)
        row = {
            **problem,
            **completion,
        }
        rows.append(row)

    return rows


def print_text(string, limit=250):
    string = string[:limit]
    print(string)


def print_diff(string1, string2, limit=250):
    string1 = string1[:limit]
    string2 = string2[:limit]
    console = Console()
    diff = difflib.ndiff(string1, string2)
    diff = list(diff)
    text = Text()

    for line in diff:
        if line.startswith("-"):
            text.append(
                line[2:], style="bold white on red"
            )  # Highlight deletions in red
        elif line.startswith("+"):
            text.append(
                line[2:], style="bold white on green"
            )  # Highlight additions in green
        else:
            text.append(line[2:], style="white")  # Keep matching characters in white

    console.print(text)
