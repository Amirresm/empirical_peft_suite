import difflib
from enum import Enum
import sys
import termios
import tty
from dataclasses import dataclass, field
from typing import Any, Dict

import keyboard
from rich.console import Console
from rich.text import Text

from src.config import ConfigMeta
from src.metrics import pairwise_metrics
from src.tui import Prompter, clear_screen


@dataclass
class Options:
    reference_config_name: str | None
    diff: bool
    repr: bool
    mode: str
    filter: str
    cursor: int = 0
    break_loop: bool = False
    config_filter: list[str] = field(default_factory=list)
    results_filter: str = "None"
    shared_fields: list[str] = field(default_factory=list)
    compared_fields: list[str] = field(default_factory=list)
    main_field: str = "pred"
    all_fields: list[str] = field(default_factory=list)


print_repr = False
print_limit = 250


def read_ch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def consume_key():
    read_ch()


def tui_show_all(row: Dict[str, Any]):
    for key, value in row.items():
        print(f"=> {key} ================")
        print_text(value)


class HEResults(Enum):
    Ok = "OK"
    Exception = "Exception"
    SyntaxError = "SyntaxError"
    Unknown = "Unknown"


def get_he_results(config: ConfigMeta, row: Dict[str, Any]):
    if config.dataset == "multiplt-r":
        if row["status"] == "OK":
            return HEResults.Ok
        elif row["status"] == "Exception":
            return HEResults.Exception
        elif row["status"] == "SyntaxError":
            return HEResults.SyntaxError
        return HEResults.Unknown
    elif "spp" in config.dataset:
        if row["result"] == "passed":
            return HEResults.Ok
        elif row["result"] == "failed:":
            return HEResults.Exception
        elif "failed:" in row["result"]:
            return HEResults.SyntaxError
        return HEResults.Unknown

    raise ValueError(f"Unknown dataset for HumanEval: {config.dataset}")


def filter_by_results(config: ConfigMeta, row: Dict[str, Any], results_filter: str):
    if results_filter == "None":
        return True
    elif (
        results_filter == "lca-all"
        and config.peft in ["lora", "compacter", "ia3"]
        and get_he_results(config, row) in [HEResults.Ok, HEResults.Exception]
    ):
        return True
    elif (
        results_filter == "lca-ok"
        and config.peft in ["lora", "compacter", "ia3"]
        and get_he_results(config, row) in [HEResults.Ok]
    ):
        return True
    elif (
        results_filter == "lca-lgc"
        and config.peft in ["lora", "compacter", "ia3"]
        and get_he_results(config, row) in [HEResults.Exception]
    ):
        return True
    elif (
        results_filter == "lc-all"
        and config.peft in ["lora", "compacter"]
        and get_he_results(config, row) in [HEResults.Ok, HEResults.Exception]
    ):
        return True
    elif (
        results_filter == "lc-ok"
        and config.peft in ["lora", "compacter"]
        and get_he_results(config, row) in [HEResults.Ok]
    ):
        return True
    elif (
        results_filter == "lc-lgc"
        and config.peft in ["lora", "compacter"]
        and get_he_results(config, row) in [HEResults.Exception]
    ):
        return True

    return False


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
        print_shared_header(
            console, key, options.reference_config_name is None and key == "target"
        )
        print_text(reference_row[key])

    if (
        options.results_filter == "None"
        or (
            options.results_filter.startswith("lca")
            and all([
                filter_by_results(c, r, options.results_filter)
                for c, r in config_row_pairs
                if c.peft in ["lora", "compacter", "ia3"]
            ])
        )
        or (
            options.results_filter.startswith("lc")
            and all([
                filter_by_results(c, r, options.results_filter)
                for c, r in config_row_pairs
                if c.peft in ["lora", "compacter"]
            ])
        )
    ):
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

                print_per_config_header(
                    console,
                    is_reference,
                    options.main_field,
                    config.remark,
                    config.peft,
                    config.peft_lib,
                    metrics,
                )
                print_compared_fields([
                    (field, row[field])
                    for field in options.compared_fields
                    if field != options.main_field
                ])
                print_main_field(row[options.main_field], reference, options.diff)
    else:
        console.print(Text("Filtered out by results filter", style="bold red"))


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
            consume_key()
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
                        limit = int(input("Enter index: ")) - 1
                        if 0 <= limit < row_count:
                            options.cursor = limit
                        else:
                            print(limit)
                            raise ValueError
                    except ValueError as e:
                        print("Invalid index:", e)
                        input("Press Enter to continue")
                case "l":
                    global print_limit
                    try:
                        limit = int(input(f"Enter new limit (current={print_limit}): "))
                        print_limit = limit
                    except ValueError as e:
                        print("Invalid limit:", e)
                        input("Press Enter to continue")
                case "s":
                    global print_repr
                    print_repr = not print_repr
                    options.repr = not options.repr
                case "f":
                    options = filter_menu(options, config_names)
                    break
                case "r":
                    options = results_filter_menu(options)
                    break
                case _:
                    # input()
                    pass

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
            consume_key()
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


def results_filter_menu(options: Options) -> Options:
    console = Console()

    choices = ["None", "lca-all", "lca-ok", "lca-lgc", "lc-all", "lc-ok", "lc-lgc"]

    if options.results_filter in choices:
        selected_index = choices.index(options.results_filter)
    else:
        selected_index = 0

    while True:
        clear_screen()
        text = Text()
        text.append("Filter by results:\n")

        for i, choice in enumerate(choices):
            label_str = (
                f"{"O-> " if choice == options.results_filter else ""}{choice}\n"
            )
            if selected_index == i:
                text.append(label_str, style="bold black on white")
            else:
                text.append(label_str)
        console.print(text)

        event = keyboard.read_event(suppress=True)
        if event.event_type == keyboard.KEY_DOWN:
            consume_key()
            key = event.name
            match key:
                case "down" | "j":
                    selected_index = min(selected_index + 1, len(choices) - 1)
                case "up" | "k":
                    selected_index = max(selected_index - 1, 0)
                case "space":
                    if choices[selected_index] == options.results_filter:
                        options.results_filter = None
                    else:
                        options.results_filter = choices[selected_index]
                    break

    return options


def print_text(string):
    global print_repr
    global print_limit
    if isinstance(string, list):
        string = str(string)
    if len(string) > print_limit:
        string = string[-print_limit:]
    if print_repr:
        print(repr(string))
    else:
        print(string)


def print_diff(string1, string2):
    global print_repr
    global print_limit

    if isinstance(string1, list):
        string1 = str(string1)
    if isinstance(string2, list):
        string2 = str(string2)
    if print_repr:
        string1 = repr(string1)
        string2 = repr(string2)

    if len(string1) > print_limit:
        string1 = string1[-print_limit:]
    if len(string2) > print_limit:
        string2 = string2[-print_limit:]
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


def print_per_config_header(
    console, is_reference, field, remark, peft, peft_lib, metrics
):
    section_text = Text()
    section_text.append("=" * 50, style="bold black on white")
    if is_reference:
        section_text.append("===")
        section_text.append("REF>", style="bold white on green")
        section_text.append(f" {field}:")
    else:
        section_text.append("======> pred:")
    section_text.append(f" {remark} ")
    if peft == "lora":
        section_text.append(f"{peft}", style="bold green")
    if peft == "compacter":
        section_text.append(f"{peft}", style="bold red")
    if peft == "ia3":
        section_text.append(f"{peft}", style="bold blue")
    if peft in ["full", "none"]:
        section_text.append(f"{peft}", style="bold yellow")
    section_text.append(f"({peft_lib}) > {metrics}")
    console.print(section_text)


def print_shared_header(console, key, is_reference):
    heading_text = Text()
    heading_text.append("=" * 20, style="bold black on grey54")
    if is_reference:
        heading_text.append("===")
        heading_text.append("REF>", style="bold white on green")
        heading_text.append(f" {key}:")
    else:
        heading_text.append(f"======> {key}:")
    console.print(heading_text)


def print_main_field(text, reference, diff):
    if diff:
        if reference is not None:
            print_diff(
                reference,
                text,
            )
        else:
            print_text("Diff disabled: no reference")
    else:
        print_text(text)


def print_compared_fields(key_text_pairs):
    short_text = ""
    long_text = ""
    for i, (key, text) in enumerate(key_text_pairs):
        text = str(text)
        if len(text) < 30:
            atom = f"{key}: {text}"
            short_text = (
                f"{atom} | "
                if short_text == "" and i < len(key_text_pairs) - 1
                else f"{atom}"
                if short_text == "" and i == len(key_text_pairs) - 1
                else short_text + f"{atom} | "
                if i < len(key_text_pairs) - 1
                else short_text + f"{atom}"
            )
        else:
            atom = f"{key}:\n{text}"
            long_text = (
                f"{atom}\n"
                if long_text == "" and i < len(key_text_pairs) - 1
                else f"{atom}"
                if long_text == "" and i == len(key_text_pairs) - 1
                else long_text + f"{atom}\n"
                if i < len(key_text_pairs) - 1
                else long_text + f"{atom}"
            )

    print_text(short_text + long_text)
    print_text("----------")
