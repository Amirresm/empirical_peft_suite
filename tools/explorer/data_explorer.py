import logging
import os
import argparse
import keyboard

from src.fs_tools import scan_dirtree
from src.metrics import pairwise_metrics
from src.tui import Prompter, clear_screen
from src.text_utils import print_diff, print_text, read_generations_from_file

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
        config_batch = next((c for name, c in configs if name == choice), None)
        if config_batch is None:
            print("No such config found.")
            continue
        folan = []
        filter = "norm|infer|full"
        for config in config_batch:
            if config.remark not in filter.split("|"):
                continue
            generated_file = os.path.join(
                config.get_path(base_dir), "gen_output", "generated_generations.txt"
            )
            if os.path.exists(generated_file):
                with open(generated_file, "r") as file:
                    out_list = read_generations_from_file(file)
                folan.append((config, out_list))
            else:
                print(f"File {generated_file} not found.")

        if len(folan) == 0:
            print("No generations found.")
            input("Press enter to continue...")
            continue

        all_modes = [
            "all",
            "prompt",
            "target",
            "pred",
            "output",
            "compare",
            "compare-within",
        ]
        mode = "compare"
        diff = False
        reference_id = 0 # 0 is target

        cursor = 0
        main_config, main_out_list = folan[0]
        done = False

        while True:
            clear_screen()
            print(f"Showing {main_config} | {cursor + 1}/{len(main_out_list)}: {mode}")

            match mode:
                case "all":
                    for key, value in main_out_list[cursor].items():
                        print(f"=> {key} ================")
                        print_text(value)
                case "compare":
                    reference_config = None
                    match reference_id:
                        case 0:
                            reference = main_out_list[cursor]["target"]
                        case _:
                            reference_index = min(reference_id - 1, len(folan) - 1)
                            reference = folan[reference_index][1][cursor]["pred"]
                            reference_config = str(folan[reference_index][0])
                    for key in ["prompt", "target"]:
                        print(f"=> {key} ================")
                        print(main_out_list[cursor][key])
                    for config, out_list in folan:
                        is_reference = str(config) == reference_config
                        
                        if config.remark in filter.split("|"):
                            metrics = pairwise_metrics(
                                reference,
                                out_list[cursor]["pred"],
                            )
                            metrics = ", ".join([
                                f"{k}={v:.2f}" for k, v in metrics.items()
                            ])
                            print(
                                f"=> pred: {"O" if is_reference else ""} {config.remark} {config.peft}({config.peft_lib}) > {metrics} ----------------"
                            )
                            if diff:
                                print_diff(
                                    reference,
                                    out_list[cursor]["pred"],
                                )
                            else:
                                print_text(out_list[cursor]["pred"])
                case "compare-within":
                    for key in ["prompt", "target", "pred"]:
                        print(f"=> {key} ================")
                        print_text(main_out_list[cursor][key])
                case "prompt":
                    print(main_out_list[cursor]["prompt"])
                case "target":
                    print(main_out_list[cursor]["target"])
                case "pred":
                    print(main_out_list[cursor]["pred"])
                case "output":
                    print(main_out_list[cursor]["output"])
            while True:
                event = keyboard.read_event()
                if event.event_type == keyboard.KEY_DOWN:
                    key = event.name
                    match key:
                        case "0":
                            reference_id = 0
                        case "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9":
                            reference_id = int(key)
                        case "down" | "j":
                            cursor = min(cursor + 1, len(main_out_list) - 1)
                        case "up" | "k":
                            cursor = max(cursor - 1, 0)
                        case "q":
                            done = True
                            break
                        case "esc":
                            cursor = 0
                        case "d":
                            diff = not diff
                        case "m":
                            mode = prompter.prompt(message="Select mode: ", data=all_modes)
                        case "c":
                            options = [str(c) for c, _ in folan]
                            c_name = prompter.prompt(
                                message="Select config: ", data=options
                            )
                            selected_config, selected_out_list = next(
                                ((c, o) for c, o in folan if str(c) == c_name), (None, None)
                            )
                            if (
                                selected_config is not None
                                and selected_out_list is not None
                            ):
                                main_config = selected_config
                                main_out_list = selected_out_list

                        case "g":
                            try:
                                index = int(input("Enter index: ")) - 1
                                if 0 <= index < len(main_out_list):
                                    cursor = index
                                else:
                                    print(index)
                                    raise ValueError
                            except ValueError as e:
                                print("Invalid index:", e)
                                continue
                        case _:
                            input()
                            continue

                    break

            if done: 
                break

if __name__ == "__main__":
    main()
