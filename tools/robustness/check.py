import pathlib
from gpt import stream_jsonl

for dsname in ["spp30k", "multipltr"]:
    previous_original_samples = []
    previous_original_samples_files = pathlib.Path(".").glob(f"{dsname}original*.jsonl")
    print(f"Checking {dsname} with {f"{dsname}original*.jsonl"} glob...")
    for file in previous_original_samples_files:
        print(f"\tChecking {file}...")
        samples = stream_jsonl(str(file))
        previous_original_samples.extend(samples)

    previous_original_samples = [sample["index"] for sample in previous_original_samples]
    unique_original_samples = set(previous_original_samples)

    print(f"Unique original samples: {len(unique_original_samples)}")
    print(f"Total original samples: {len(previous_original_samples)}")

    duplicate_samples = [sample for sample in previous_original_samples if previous_original_samples.count(sample) > 1]
    for sample in duplicate_samples:
        print(f"\t{sample}")
