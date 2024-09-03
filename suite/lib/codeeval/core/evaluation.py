from transformers.trainer_utils import Dict
from .data import write_jsonl, read_problems
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from tqdm import tqdm
import itertools
import typing

BatchGenerator = typing.Callable[
    [PreTrainedModel, PreTrainedTokenizer, list[str], bool], list[str]
]


# reference: https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")


def split_batch(samples: list[str], size=4):
    mini_batches = []

    for i in range(0, len(samples), size):
        mini_batches.append(samples[i : i + size])

    return mini_batches


def run_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_samples_per_task: int,
    out_path: str,
    generate_batch_completion: BatchGenerator,
    problems: Dict[str, Dict],
    format_tabs: bool = False,
    limit: int | None = None,
    batch_size: int = 4,
):
    # problems = read_problems()
    if limit is not None and limit > 0 and limit < len(problems):
        problems = dict(itertools.islice(problems.items(), limit))

    # problems = dict(itertools.islice(problems.items(), 20))
    samples = []

    batched_problems = [[]]
    for task_id in problems:
        if format_tabs:
            prompt = problems[task_id]["prompt"].replace("    ", "\t")
        else:
            prompt = problems[task_id]["prompt"]

        for _ in range(num_samples_per_task):
            if len(batched_problems[-1]) < batch_size:
                batched_problems[-1].append((task_id, prompt))
            else:
                batched_problems.append([(task_id, prompt)])

    pbar = tqdm(total=len(problems) * num_samples_per_task)
    for batch in batched_problems:
        batch_problems = [x[1] for x in batch]
        batch_completions = generate_batch_completion(
            model,
            tokenizer,
            batch_problems,
            True if num_samples_per_task != batch_size else False,
        )
        for i, sample in enumerate(batch_completions):
            task_id = batch[i][0]
            result = dict(
                task_id=task_id,
                completion=sample,
            )

            samples += [result]
        pbar.update(len(batch))
        pbar.set_description(f"Humaneval batch_size={batch_size}")

    write_jsonl(out_path, samples)

    return samples
