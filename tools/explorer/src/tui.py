import os

from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion
from fuzzyfinder import fuzzyfinder


class FuzzyCompleter(Completer):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def get_completions(self, document, complete_event):
        user_input = document.text

        matches = fuzzyfinder(user_input, self.data, sort_results=False)

        for match  in matches:
            yield Completion(match, start_position=-len(user_input))


class Prompter:
    def __init__(self, data):
        self.completer = FuzzyCompleter(data)

    def prompt(self, message="Enter query: ", data = None):
        if data is not None:
            return prompt(message, completer=FuzzyCompleter(data))
        return prompt(message, completer=self.completer)


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
    # print("\033[H\033[J")
