import logging
import os
import argparse
from typing import Literal

from src.fs_tools import load_cache_from_disk, save_cache_to_disk, scan_dirtree
from src.tui import Prompter, clear_screen
from src.text_utils import (
    read_generations_from_file,
    read_humaneval_python_from_file,
    read_humaneval_r_from_file,
)
from src.tui_utils import (
    Options,
    data_menu,
    get_filtered_config_rows_pairs,
    print_text,
    tui_compare,
    tui_show_all,
)

for key in logging.Logger.manager.loggerDict:
    print(key)

CACHE_DIRECTORY = os.path.join(os.path.dirname(__file__), "cache")


def load_from_data(
    prompter, configs, base_dir
) -> tuple[list | Literal["break"] | Literal["continue"], Options | None, str]:
    choice = prompter.prompt()
    if choice in ["exit", "q"]:
        return "break", None, ""
    print(f"Query: {choice}")

    options = Options(
        reference_config_name=None,
        diff=False,
        repr=False,
        mode="compare",
        filter="norm|infer|full",
    )

    config_batch = next((c for name, c in configs if name == choice), None)
    if config_batch is None:
        print("No such config found.")
        return "continue", None, ""
    config_rows_pairs = []

    source_file_name = os.listdir(
        os.path.join(config_batch[0].get_path(base_dir), "gen_output")
    )
    source_file_name = [
        f.split("/")[-1]
        for f in source_file_name
        if f.endswith(".txt") or "humaneval_r_problems" in f
    ]
    source_file_name = prompter.prompt(
        message="Select source file: ", data=source_file_name
    )
    for config in config_batch:
        if config.remark not in options.filter.split("|"):
            continue
        generated_file = os.path.join(
            config.get_path(base_dir), "gen_output", source_file_name
        )
        if os.path.isdir(generated_file) and "humaneval_r_problems" in generated_file:
            out_list = read_humaneval_r_from_file(generated_file)
            config_rows_pairs.append((config, out_list))
            options.shared_fields = ["name", "prompt"]
            options.compared_fields = ["stdout", "stderr", "exit_code", "status"]
            options.main_field = "completions"
        elif os.path.exists(generated_file):
            if source_file_name == "generated_humaneval.txt":
                with open(generated_file, "r") as file:
                    out_list = read_humaneval_python_from_file(file)
                config_rows_pairs.append((config, out_list))
                options.shared_fields = ["task"]
                options.compared_fields = ["result"]
                options.main_field = "completion"
            else:
                with open(generated_file, "r") as file:
                    out_list = read_generations_from_file(file)
                config_rows_pairs.append((config, out_list))
                options.shared_fields = ["prompt", "target"]
                options.compared_fields = ["pred"]
                options.main_field = "pred"
        else:
            print(f"File {generated_file} not found.")

    save_cache_to_disk(config_rows_pairs, options, CACHE_DIRECTORY, choice)
    return config_rows_pairs, options, choice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False)
    args = parser.parse_args()

    if not os.path.exists(CACHE_DIRECTORY):
        os.makedirs(CACHE_DIRECTORY)

    base_dir = None
    configs = None
    if args.input is not None:
        base_dir = os.path.abspath(args.input)
        configs = scan_dirtree(args.input)
        bank = [c for c, _ in configs]
        prompter = Prompter(bank)
    else:
        # reading from cache
        cache_entries = os.listdir(CACHE_DIRECTORY)
        prompter = Prompter(cache_entries)

    while True:
        clear_screen()

        view_name = ""
        if args.input is not None and configs is not None and base_dir:
            config_rows_pairs, options, view_name = load_from_data(
                prompter, configs, base_dir
            )
        else:
            choice = prompter.prompt()
            if choice in ["exit", "q"]:
                break

            config_rows_pairs, options, view_name = load_cache_from_disk(
                os.path.join(CACHE_DIRECTORY, choice)
            )

        if config_rows_pairs == "break":
            break
        if config_rows_pairs == "continue" or options is None:
            continue

        if len(config_rows_pairs) == 0:
            print("No generations found.")
            input("Press enter to continue...")
            continue

        options.config_filter = [str(c) for c, _ in config_rows_pairs]
        filtered_config_rows_pairs = get_filtered_config_rows_pairs(
            config_rows_pairs, options
        )

        while True:
            clear_screen()
            if not options.filters_executed:
                filtered_config_rows_pairs = get_filtered_config_rows_pairs(
                    config_rows_pairs, options
                )
            reference_config, reference_row = next(
                (
                    (config, row)
                    for config, row in filtered_config_rows_pairs
                    if str(config) == options.reference_config_name
                ),
                filtered_config_rows_pairs[0],
            )

            print(
                f"Showing {reference_config} | {options.cursor + 1}/{len(reference_row)}: {options.mode}"
            )

            match options.mode:
                case "all":
                    tui_show_all(reference_row[options.cursor])
                case "compare":
                    if len(reference_row) == 0:
                        print("No data to compare.")
                    else:
                        config_row_pairs = [
                            (c, o[options.cursor])
                            for c, o in filtered_config_rows_pairs
                        ]
                        tui_compare(
                            config_row_pairs,
                            options=options,
                        )
                case "compare-within":
                    for key in ["prompt", "target", "pred"]:
                        print(f"=> {key} ================")
                        print_text(reference_row[options.cursor][key])
                case "prompt":
                    print(reference_row[options.cursor]["prompt"])
                case "target":
                    print(reference_row[options.cursor]["target"])
                case "pred":
                    print(reference_row[options.cursor]["pred"])
                case "output":
                    print(reference_row[options.cursor]["output"])

            options = data_menu(
                options,
                prompter=prompter,
                config_names=[str(c) for c, _ in filtered_config_rows_pairs],
                row_count=len(reference_row),
            )

            if options.break_loop:
                break


if __name__ == "__main__":
    main()
