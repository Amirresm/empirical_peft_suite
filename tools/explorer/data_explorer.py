import logging
import os
import argparse
import keyboard

from src.fs_tools import scan_dirtree
from src.metrics import pairwise_metrics
from src.tui import Prompter, clear_screen
from src.text_utils import (
    print_diff,
    print_text,
    read_generations_from_file,
    read_humaneval_r_from_file,
)
from src.tui_utils import Options, data_menu, tui_compare, tui_show_all

for key in logging.Logger.manager.loggerDict:
    print(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--recursive", type=bool, default=False, required=False)
    args = parser.parse_args()

    # convert to absolute path
    base_dir = os.path.abspath(args.input)

    configs = scan_dirtree(args.input)

    bank = [c for c, _ in configs]

    prompter = Prompter(bank)

    while True:
        clear_screen()
        choice = prompter.prompt()
        if choice == "exit":
            break
        print(f"Query: {choice}")

        options = Options(
            reference_config_name=None,
            diff=False,
            mode="compare",
            filter="norm|infer|full",
        )

        config_batch = next((c for name, c in configs if name == choice), None)
        if config_batch is None:
            print("No such config found.")
            continue
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
            if (
                os.path.isdir(generated_file)
                and "humaneval_r_problems" in generated_file
            ):
                out_list = read_humaneval_r_from_file(generated_file)
                config_rows_pairs.append((config, out_list))
                options.shared_fields = ["prompt"]
                options.compared_fields = ["stdout", "stderr", "exit_code", "status"]
                options.main_field = "completions"
            elif os.path.exists(generated_file):
                with open(generated_file, "r") as file:
                    out_list = read_generations_from_file(file)
                config_rows_pairs.append((config, out_list))
                options.shared_fields = ["prompt", "target"]
                options.compared_fields = ["pred"]
                options.main_field = "pred"
            else:
                print(f"File {generated_file} not found.")

        if len(config_rows_pairs) == 0:
            print("No generations found.")
            input("Press enter to continue...")
            continue

        options.config_filter = [str(c) for c, _ in config_rows_pairs]

        reference_config, reference_row = next(
            (
                row
                for config, row in config_rows_pairs
                if str(config) == options.reference_config_name
            ),
            config_rows_pairs[0],
        )

        while True:
            clear_screen()
            print(
                f"Showing {reference_config} | {options.cursor + 1}/{len(reference_row)}: {options.mode}"
            )

            config_row_pairs = [(c, o[options.cursor]) for c, o in config_rows_pairs]
            match options.mode:
                case "all":
                    tui_show_all(reference_row[options.cursor])
                case "compare":
                    tui_compare(
                        config_row_pairs,
                        options=options,
                        # options.reference_config_name,
                        # options.filter,
                        # options.config_filter,
                        # options.diff,
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
                config_names=[str(c) for c, _ in config_rows_pairs],
                row_count=len(reference_row),
            )

            if options.break_loop:
                break


if __name__ == "__main__":
    main()
