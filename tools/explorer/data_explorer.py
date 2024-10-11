import re
import os
import argparse
import keyboard

from src.fs_tools import scan_dirtree
from src.tui import Prompter, clear_screen
from src.text_utils import print_diff, print_text, read_generations_from_file


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
        for config in config_batch:
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
        filter = "norm|infer|full"
        diff = False

        cursor = 0
        main_config, main_out_list = folan[0]

        while True:
            clear_screen()
            print(f"Showing {main_config} | {cursor + 1}/{len(main_out_list)}: {mode}")

            match mode:
                case "all":
                    for key, value in main_out_list[cursor].items():
                        print(f"=> {key} ================")
                        print_text(value)
                case "compare":
                    for key in ["prompt", "target"]:
                        print(f"=> {key} ================")
                        print(main_out_list[cursor][key])
                    for config, out_list in folan:
                        if config.remark in filter.split("|"):
                            print(f"=> pred: {config.remark} {config.peft}({config.peft_lib}) ----------------")
                            if diff:
                                print_diff(main_out_list[cursor]["target"], out_list[cursor]["pred"])
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
            event = keyboard.read_event()
            if event.event_type == keyboard.KEY_DOWN:
                key = event.name
                match key:
                    case "down" | "j":
                        cursor = min(cursor + 1, len(main_out_list) - 1)
                    case "up" | "k":
                        cursor = max(cursor - 1, 0)
                    case "q":
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
                            input("Press enter to continue...")
                    case _:
                        print(f"Unknown key {key}")


if __name__ == "__main__":
    main()
