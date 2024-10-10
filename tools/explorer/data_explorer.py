import argparse

from src.fs_tools import scan_dirtree
from src.tui import Prompter, clear_screen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--recursive", type=bool, default=False, required=False)
    args = parser.parse_args()

    configs = scan_dirtree(args.input)

    bank = [c.get_dirname() for c in configs]

    prompter = Prompter(bank)

    while True:
        choice = prompter.prompt()
        if choice == "exit":
            break

        clear_screen()
        print(f"Query: {choice}")
        config = next((c for c in configs if c.get_dirname() == choice), None)
        print(config)


if __name__ == "__main__":
    main()
