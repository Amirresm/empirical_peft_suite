import random
import argparse
import gzip
import json
import os
from collections.abc import Iterable
from typing import Dict
import pathlib

from openai import OpenAI

from text_utils import DatasetInstances, join_columns, read_generations, split_column


def gpt_call(client, language, sample):
    # prompt = '''Please modify the doc string of the following function to mimic a user spelling typo, grammatical error and incorrect synonyms. You must introduce at least TWO spelling errors (typos) and ONE improper synonym. You must also slightly paraphrase to introduce grammatical errors.
    prompt = '''Your job is to rewrite doc string to mimic a real-world user who is not proficient in english. Please rewrite the sentence to include misspellings, typos, and grammar mistakes. Then while keeping the errors, rephrase it so that it conveys the same idea but sounds different from the original version.
Only change the description of the function in the doc string. Do not change the example usecases in the doc string. Do not change the function signature, body or the name.
Only generate the modified doc string and function header. Do not add extra text. The output should have the same format as the input. 
Example input for Python:
def is_palindrome_string(test_str):
    """ Check if the given string is a palindrome or not.
    >>> is_palindrome_string("radar")
    True
    >>> is_palindrome_string("racecar")
    True
    >>> is_palindrome_string("raceca")
    False
    """

Example output for Python:
def is_palindrome_string(test_str):
    """ Is gave text palindorm?
    >>> is_palindrome_string("radar")
    True
    >>> is_palindrome_string("racecar")
    True
    >>> is_palindrome_string("raceca")
    False
    """


Example input for R:
# A quadratic cutoff that goes to zero smoothly at the cutoff boundary.
# Args:
#     r_cut (float): Cutoff value (in angstrom).
#     ri (float): Interatomic distance.
#     ci (float): Cartesian coordinate divided by the distance.
# Returns:
#     (float, float): Cutoff value and its derivative.
quadratic_cutoff <- function(r_cut, ri, ci) {

Example input for R:
# A curve drops down gradually to nothing right at the edge.
# Args:
#     r_cut (float): Cutoff value (in angstrom).
#     ri (float): Interatomic distance.
#     ci (float): Cartesian coordinate divided by the distance.
# Returns:
#     (float, float): Cutoff value and its derivative.
quadratic_cutoff <- function(r_cut, ri, ci) {


Input '''

    instruction = f"{prompt}for {language}: {sample}"
    # print(f"Instruction: {instruction}")
    # return sample
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": instruction,
            },
        ],
    )

    raw = completion.choices[0].message.content
    if raw.startswith("```r"):
        raw = raw[len("```r") :]
    if raw.startswith("```python"):
        raw = raw[len("```python") :]
    if raw.endswith("```"):
        raw = raw[: -len("```")]
    return raw


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if isinstance(line, str) and any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))


def main():
    random.seed(43)
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_count", type=int)
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=False)
    parser.add_argument("--results_dir", type=str, required=False)
    args = parser.parse_args()
    client = OpenAI(
        project="proj_7Q7BWecL9I2gQWvimk0efdlK",
    )
    if args.data_dir is None and args.results_dir is None:
        print("Please provide either data_dir or results_dir")
        exit(1)
    sample_count = args.sample_count

    random_sample_set_dict = {}
    if args.data_dir is not None:
        print(f"Data Directory: {args.data_dir}")
        for ds_parent in os.listdir(args.data_dir):
            ds_jsonl = next(
                iter(os.listdir(os.path.join(args.data_dir, ds_parent))), None
            )
            if ds_jsonl is None:
                continue
            ds_type = (
                DatasetInstances.SPP
                if "spp" in ds_parent
                else DatasetInstances.MULTIPLT
            )

            if os.path.exists(f"{ds_parent}-original.jsonl"):
                os.remove(f"{ds_parent}-original.jsonl")
            if os.path.exists(f"{ds_parent}-modified.jsonl"):
                os.remove(f"{ds_parent}-modified.jsonl")

            samples = list(
                stream_jsonl(os.path.join(args.data_dir, ds_parent, ds_jsonl))
            )
            samples = [{**sample, "index": i} for i, sample in enumerate(samples)]
            random_sample_set = random.sample(
                samples,
                sample_count,
            )
            random_sample_set_dict[ds_parent] = random_sample_set

    if args.results_dir is not None:
        print(f"Results Directory: {args.results_dir}")
        paths = {
            "spp-30k": "gen/codellama-7b/spp_30k/norm_gen_codellama-7b_spp_30k_pl_lora/gen_output/generated_generations.txt",
            "multiplt-r": "gen/codellama-7b/multiplt-r/norm_gen_codellama-7b_multiplt-r_pl_lora/gen_output/generated_generations.txt",
        }
        for ds_name, path in paths.items():
            if not os.path.exists(os.path.join(args.results_dir, path)):
                print(f"Path for {ds_name} does not exist, check 'results_dir'")
                continue
            ds_type = (
                DatasetInstances.SPP if "spp" in ds_name else DatasetInstances.MULTIPLT
            )

            previous_original_samples = []
            previous_original_samples_files = pathlib.Path(".").glob("*original*.jsonl")
            for file in previous_original_samples_files:
                previous_original_samples.extend(stream_jsonl(str(file)))
            previous_original_samples = [s["index"] for s in previous_original_samples]

            with open(os.path.join(args.results_dir, path), "r") as fp:
                unformatted_samples = read_generations(fp)
                formatted_samples = [
                    {
                        "prompt": usample["prompt"],
                        "completion": usample["target"],
                        "index": i,
                    }
                    if ds_type == DatasetInstances.MULTIPLT
                    else {"code": usample["example"], "index": i}
                    for i, usample in enumerate(unformatted_samples)
                ]
                # random_sample_set = random.sample(
                #     formatted_samples,
                #     sample_count,
                # )
                random_sample_set = []
                for i in range(sample_count):
                    rs = random.choice(formatted_samples)
                    while rs["index"] in previous_original_samples or rs["index"] in [s["index"] for s in random_sample_set]:
                        rs = random.choice(formatted_samples)
                    random_sample_set.append(rs)

                random_sample_set = [
                    ({
                        k: (
                            property[:-1]
                            if isinstance(property, str) and property.endswith("\n\n")
                            else property
                        )
                        for k, property in sample.items()
                    })
                    for sample in random_sample_set
                ]
                random_sample_set_dict[ds_name] = random_sample_set

    for ds_name, random_sample_set in random_sample_set_dict.items():
        print(f"{ds_name}: {len(random_sample_set)}")
        for i, sample in enumerate(random_sample_set[:1]):
            print(f"{i + 1}: {sample}")

    for ds_name, random_sample_set in random_sample_set_dict.items():
        ds_type = (
            DatasetInstances.SPP if "spp" in ds_name else DatasetInstances.MULTIPLT
        )
        ds_language = (
            "R" if "multiplt" in ds_type == DatasetInstances.MULTIPLT else "Python"
        )

        target_original_path = f"{ds_name}-original{args.title}.jsonl"
        target_original_path = target_original_path.replace("-", "")
        target_modified_path = f"{ds_name}-modified{args.title}.jsonl"
        target_modified_path = target_modified_path.replace("-", "")
        if os.path.exists(target_original_path):
            os.remove(target_original_path)
        if os.path.exists(target_modified_path):
            os.remove(target_modified_path)

        for i, sample in enumerate(random_sample_set):
            if ds_type == DatasetInstances.MULTIPLT:
                example = sample["prompt"]
            else:
                example = sample["code"]
            prompt, target = split_column(example, ds_type)
            modified_prompt = gpt_call(client, ds_language, prompt)
            if modified_prompt is None:
                raise ValueError("GPT-4o failed to generate a response.")

            if ds_type == DatasetInstances.MULTIPLT:
                modified_prompt = modified_prompt.lstrip()
            else:
                modified_prompt = modified_prompt.strip()
            print(f"Sample {i + 1}/{sample_count} " + "-" * 50)
            print(f"Orig: {repr(prompt)}")
            print(f"New: {repr(modified_prompt)}")
            modified_example = join_columns(modified_prompt, target, ds_type)
            if ds_type == DatasetInstances.MULTIPLT:
                modified_sample = {**sample, "prompt": modified_example}
            else:
                modified_sample = {**sample, "code": modified_example}

            write_jsonl(target_original_path, [sample], append=True)
            write_jsonl(target_modified_path, [modified_sample], append=True)

    pass


if __name__ == "__main__":
    main()
