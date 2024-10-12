from dataclasses import dataclass, field
from typing import Any, Dict

from rich.console import Console
from rich.text import Text

from src.config import ConfigMeta
from src.metrics import pairwise_metrics
from src.text_utils import print_diff, print_text

import keyboard

from src.tui import Prompter, clear_screen


@dataclass
class Options:
    reference_config_name: str | None
    diff: bool
    mode: str
    filter: str
    cursor: int = 0
    break_loop: bool = False
    config_filter: list[str] = field(default_factory=list)
    shared_fields: list[str] = field(default_factory=list)
    compared_fields: list[str] = field(default_factory=list)
    main_field: str = "pred"
    all_fields: list[str] = field(default_factory=list)


def tui_show_all(row: Dict[str, Any]):
    for key, value in row.items():
        print(f"=> {key} ================")
        print_text(value)


def tui_compare(
    config_row_pairs: list[tuple[ConfigMeta, Dict[str, Any]]],
    options: Options,
    # reference_config: str | None,
    # filter: str,
    # config_filter: list[str],
    # diff: bool,
):
    console = Console()
    reference_row = next(
        (
            row
            for config, row in config_row_pairs
            if str(config) == options.reference_config_name
        ),
        config_row_pairs[0][1],
    )

    reference = None
    if options.reference_config_name is None:
        if "target" in reference_row:
            reference = reference_row["target"]
    else:
        reference = reference_row[options.main_field]

    for key in options.shared_fields:
        heading_text = Text()
        heading_text.append("=" * 20, style="bold black on grey54")
        if options.reference_config_name is None and key == "target":
            heading_text.append("===")
            heading_text.append("REF>", style="bold white on green")
            heading_text.append(f" {key}:")
        else:
            heading_text.append(f"======> {key}:")
        console.print(heading_text)
        print(reference_row[key])

    for config, row in config_row_pairs:
        is_reference = str(config) == options.reference_config_name
        if (
            config.remark in options.filter.split("|")
            and str(config) in options.config_filter
        ):
            if reference is not None:
                metrics = pairwise_metrics(reference, row[options.main_field])
                metrics = ", ".join([f"{k}={v:.2f}" for k, v in metrics.items()])
            else:
                metrics = ""
            section_text = Text()
            section_text.append("=" * 50, style="bold black on white")
            if is_reference:
                section_text.append("===")
                section_text.append("REF>", style="bold white on green")
                section_text.append(f" {options.main_field}:")
            else:
                section_text.append("======> pred:")
            section_text.append(f" {config.remark} ")
            if config.peft == "lora":
                section_text.append(f"{config.peft}", style="bold green")
            if config.peft == "compacter":
                section_text.append(f"{config.peft}", style="bold red")
            if config.peft == "ia3":
                section_text.append(f"{config.peft}", style="bold blue")
            if config.peft in ["full", "none"]:
                section_text.append(f"{config.peft}", style="bold yellow")
            section_text.append(f"({config.peft_lib}) > {metrics}")
            console.print(section_text)
            if options.diff:
                print_diff(
                    reference,
                    row[options.main_field],
                )
            else:
                print_text(row[options.main_field])


def data_menu(
    options: Options, prompter: Prompter, config_names: list[str], row_count: int
) -> Options:
    all_modes = [
        "all",
        "prompt",
        "target",
        "pred",
        "output",
        "compare",
        "compare-within",
    ]
    while True:
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_DOWN:
            key = event.name
            match key:
                case "0":
                    options.reference_config_name = None
                case "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9":
                    reference_id = int(key)
                    reference_index = min(reference_id - 1, len(config_names) - 1)
                    options.reference_config_name = config_names[reference_index]
                case "down" | "j":
                    options.cursor = min(options.cursor + 1, row_count - 1)
                case "up" | "k":
                    options.cursor = max(options.cursor - 1, 0)
                case "q":
                    options.break_loop = True
                    break
                case "esc":
                    options.cursor = 0
                case "d":
                    options.diff = not options.diff
                case "m":
                    options.mode = prompter.prompt(
                        message="Select mode: ", data=all_modes
                    )
                case "g":
                    try:
                        index = int(input("Enter index: ")) - 1
                        if 0 <= index < row_count:
                            options.cursor = index
                        else:
                            print(index)
                            raise ValueError
                    except ValueError as e:
                        print("Invalid index:", e)
                        continue
                case "f":
                    options = filter_menu(options, config_names)
                    break
                case _:
                    # input()
                    continue

            break

    return options


def filter_menu(options: Options, config_names: list[str]) -> Options:
    console = Console()

    selected_index = 0
    while True:
        clear_screen()
        text = Text()
        text.append("Filter by config name:\n")

        for i, config in enumerate(config_names):
            config_str = (
                f"{"O-> " if config in options.config_filter else ""}{config}\n"
            )
            if selected_index == i:
                text.append(config_str, style="bold black on white")
            else:
                text.append(config_str)
        console.print(text)

        event = keyboard.read_event(suppress=True)
        if event.event_type == keyboard.KEY_DOWN:
            key = event.name
            match key:
                case "down" | "j":
                    selected_index = min(selected_index + 1, len(config_names) - 1)
                case "up" | "k":
                    selected_index = max(selected_index - 1, 0)
                case "space":
                    if config_names[selected_index] in options.config_filter:
                        options.config_filter.remove(config_names[selected_index])
                    else:
                        options.config_filter.append(config_names[selected_index])

                case "q":
                    break

    return options
