import re
import os
import argparse
from typing import Dict
import tqdm
from text_utils import DatasetInstances, split_column, join_columns


def scan_dir(dir: str):
    datasets = set()
    configs = {}
    gen_job_datasets = [
        "spp_30k",
        "sppu_30k",
        "multiplt-r",
        "csn-python",
        "rsum-combined",
    ]
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
                    ds_instance = (
                        DatasetInstances.SPP
                        if "spp" in dataset_name
                        else DatasetInstances.MULTIPLT
                        if "multiplt" in dataset_name
                        else DatasetInstances.CSN
                        if "csn" in dataset_name or "rsum" in dataset_name
                        else None
                    )
                    print(f"Processing {ds_instance} in {config_path}")
                    do_the_work(config_path, ds_instance)

    return configs, datasets


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

    out = ""
    out += f"{index}=========\n"
    out += f"->Example:\n{example}"
    out += f"->Prompt:\n{prompt}"
    out += f"->Target:\n{target}"
    out += f"->Pred:\n{pred}"
    out += f"->Output:\n{output}"
    out += f"->Inp_Tokens:\n{inp_tokens}\n"
    out += f"->Out_Tokens:\n{out_tokens}\n"
    out += f"->New_Tokens:\n{new_tokens}\n"
    # out += f"->Inp_Tokens:\n{0}\n"
    # out += f"->Out_Tokens:\n{0}\n"
    # out += f"->New_Tokens:\n{0}\n"
    out += "--\n\n\n"
    return out


def read_generations_from_file(
    file, out_file, ds_instance, processed_limit=1000000, total=None
):
    bar = tqdm.tqdm(total=total)
    lines = file.readlines()

    keys = [
        ("Pred:", "pred"),
        ("Gold:", "gold"),
        ("Input:", "input"),
    ]

    cursor = None
    buffer_dict: Dict | None = None
    beginning_regex = re.compile(r".*INFO: \d+===.*")
    end_regex = re.compile(r".*INFO: \d+ generations saved to.*")
    starting_line_met = False
    processed = 0
    with open(out_file, "w") as out_file:
        for line in lines:
            if processed_limit == 0:
                break
            if "INFO: *** Generate ***" in line:
                starting_line_met = True
                continue
            if end_regex.match(line) and buffer_dict is not None:
                processed += 1
                # save buffer to file
                text = process_buffer_dict(processed, buffer_dict, ds_instance)
                out_file.write(text)
                processed_limit -= 1
                cursor = None
                buffer_dict = {}
                bar.update(1)
                break
            if not starting_line_met:
                continue

            if beginning_regex.match(line):
                if buffer_dict is None:
                    buffer_dict = {}
                    continue
                else:
                    processed += 1
                    # save buffer to file
                    text = process_buffer_dict(processed, buffer_dict, ds_instance)
                    out_file.write(text)
                    processed_limit -= 1
                    cursor = None
                    buffer_dict = {}
                    bar.update(1)
                    continue

            new_cursor_set = False
            for key, value in keys:
                if line.startswith(key):
                    cursor = value
                    new_cursor_set = True
                    break
            if new_cursor_set:
                continue
            if cursor is None and processed == 0:
                cursor = "input"

            if buffer_dict is not None and cursor is not None:
                if cursor in buffer_dict:
                    buffer_dict[cursor] += line
                else:
                    buffer_dict[cursor] = line

            else:
                continue


def do_the_work(dir, ds_instance):
    generation_dir = os.path.join(dir, "gen_output")
    generation_file = os.path.join(generation_dir, "generated_generations.txt")
    if os.path.exists(generation_file):
        with open(generation_file, "r") as file:
            # check if file is not empty
            first_char = file.read(1)
            if first_char:
                print(f"Omitting {generation_file}: File is not empty")
                return

    process_path = os.path.join(dir, "job_report.log")
    # if generation_dir path exists
    if os.path.exists(process_path) and os.path.isdir(generation_dir):
        print(f"Processing {ds_instance} in {dir}")
        with open(process_path, "r") as file:
            match ds_instance:
                case DatasetInstances.SPP:
                    total = 1484
                case DatasetInstances.MULTIPLT:
                    total = 1860
                case DatasetInstances.CSN:
                    total = 14918
                case _:
                    total = None
            read_generations_from_file(
                file,
                generation_file,
                ds_instance,
                # processed_limit=10,
                total=total,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--recursive", type=bool, default=False, required=False)
    args = parser.parse_args()

    if args.recursive:
        scan_dir(args.input)

    else:
        do_the_work(args.input, DatasetInstances.CSN)


if __name__ == "__main__":
    main()
