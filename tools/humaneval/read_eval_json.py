import argparse
import json
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    with open(args.input, "r") as json_file:
        json_list = list(json_file)

    samples = []
    for json_str in json_list:
        result = json.loads(json_str)
        samples.append(result)
        # print(f"result: {result}")
        # print(isinstance(result, dict))

    passed_ids = []
    passed_results_counter = Counter()
    failed_results_counter = Counter()
    for sample in samples:
        print(f"\n=====> task_id: {sample['task_id']}")
        if sample["passed"]:
            passed_ids.append(sample["task_id"])
            passed_results_counter.update([sample["result"]])
            print(f"PASS:\nresult: {sample['result']}\n{sample['completion']}")
        else:
            failed_results_counter.update([sample["result"]])
            print(f"FAIL:\nresult: {sample['result']}\n{sample['completion']}")

    print(f"Total passed: {len(passed_ids)}")
    print(f"Passed ids: {passed_ids}")

    print(f"Passed results: {passed_results_counter.most_common(10)}")
    print(f"Failed results: {failed_results_counter.most_common(10)}")


if __name__ == "__main__":
    main()
