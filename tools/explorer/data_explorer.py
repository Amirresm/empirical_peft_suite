import os
import argparse
import keyboard

from src.fs_tools import scan_dirtree
from src.tui import Prompter, clear_screen
from src.text_utils import DatasetInstances, read_generations_from_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--recursive", type=bool, default=False, required=False)
    args = parser.parse_args()

    # convert to absolute path
    base_dir = os.path.abspath(args.input)

    configs = scan_dirtree(args.input)

    bank = [c.get_dirname() for c in configs]

    prompter = Prompter(bank)

    while True:
        clear_screen()
        choice = prompter.prompt()
        if choice == "exit":
            break

        print(f"Query: {choice}")
        config = next((c for c in configs if c.get_dirname() == choice), None)
        if config is None:
            print("No such config found.")
            continue
        generated_file = os.path.join(
            config.get_path(base_dir), "gen_output", "generated_generations.txt"
        )
        with open(generated_file, "r") as file:
            out_list = read_generations_from_file(file)

            all_modes = ["all", "prompt", "target", "pred", "output", "compare"]
            mode = "compare"

            cursor = 0

            while True:
                clear_screen()
                print(f"Showing {cursor + 1}/{len(out_list)}: {mode}")

                match mode:
                    case "all":
                        for key, value in out_list[cursor].items():
                            print(f"=> {key}:")
                            print(value)
                    case "compare":
                        for key in ["prompt", "target", "pred"]:
                            print(f"=> {key}:")
                            print(out_list[cursor][key])
                    case "prompt":
                        print(out_list[cursor]["prompt"])
                    case "target":
                        print(out_list[cursor]["target"])
                    case "pred":
                        print(out_list[cursor]["pred"])
                    case "output":
                        print(out_list[cursor]["output"])
                event = keyboard.read_event()
                if event.event_type == keyboard.KEY_DOWN:
                    key = event.name
                    match key:
                        case "down" | "j":
                            cursor = min(cursor + 1, len(out_list) - 1)
                        case "up" | "k":
                            cursor = max(cursor - 1, 0)
                        case "q":
                            break
                        case "esc":
                            cursor = 0
                        case "m":
                            mode = prompter.prompt(
                                message="Select mode: ", data=all_modes
                            )

                        case "g":
                            try:
                                index = int(input("Enter index: ")) - 1
                                if 0 <= index < len(out_list):
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
