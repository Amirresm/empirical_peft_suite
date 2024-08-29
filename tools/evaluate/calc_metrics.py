import argparse
from typing import Dict
import tqdm
from bleu2.calc_bleu2 import calculate_bleu2
import evaluate

keys = [
    ("->Prompt:", "prompt"),
    ("->Target:", "target"),
    ("->Pred:", "pred"),
    ("->Output:", "output"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    metric_bleu = "/home/amirreza/projects/ubc/peft_suite/suite/lib/bleu/bleu.py"
    metric_rouge = "/home/amirreza/projects/ubc/peft_suite/suite/lib/rouge/rouge.py"

    metric_bleu = evaluate.load(path=metric_bleu)
    metric_rouge = evaluate.load(path=metric_rouge)
    with open(args.input, "r") as file:
        preds, targets = read_generations_from_file2(file)

        results = calc_all_metrics(
            preds, targets, metric_rouge, metric_bleu, metric_bleu
        )
        
        # save results next to args.input
        with open(args.input + ".results", "w") as file:
            for key, value in results.items():
                file.write(f"{key}: {value}\n")

        print(f"Generation Results:\n{results}")


def read_generations_from_file(file, line_limit=1000000):
    stuff = []
    bar = tqdm.tqdm()
    lines = file.readlines()
    buffer_dict: Dict | None = None
    cursor = None
    for line in lines:
        line_limit -= 1
        if line_limit == 0:
            break
        if "=========" in line:
            if buffer_dict is None:
                buffer_dict = {}
                continue
            else:
                stuff.append(buffer_dict.copy())
                bar.update(1)
                bar.set_description(f"Processed {len(stuff)}")
                continue
        for key, value, idx in keys:
            if line.startswith(key):
                cursor = value
                continue
        if line.startswith("->"):
            continue
        if buffer_dict is not None and cursor is not None:
            if cursor in buffer_dict:
                buffer_dict[cursor] += line
            else:
                buffer_dict[cursor] = line


def read_generations_from_file2(file, line_limit=1000000):
    bar = tqdm.tqdm()
    lines = file.readlines()
    buffer_dict: Dict | None = None
    cursor = None
    preds = []
    refs = []
    for line in lines:
        line_limit -= 1
        if line_limit == 0:
            break
        if "=========" in line:
            if buffer_dict is None:
                buffer_dict = {}
                continue
            else:
                preds.append(buffer_dict["pred"])
                refs.append(buffer_dict["target"])
                bar.update(1)
                bar.set_description(f"Processed {len(preds)}")
                continue
        for key, value in keys:
            if line.startswith(key):
                cursor = value
                continue
        if line.startswith("->"):
            continue
        if buffer_dict is not None and cursor is not None:
            if cursor in buffer_dict:
                buffer_dict[cursor] += line
            else:
                buffer_dict[cursor] = line

    return preds, refs


def calc_all_metrics(
    preds,
    labels,
    metric_rouge,
    metric_bleu,
    metric_path,
):
    result = {}
    # print(f"Calculating rouge for {len(preds)} predictions")
    # result = metric_rouge.compute(
    #     predictions=preds,
    #     references=labels,
    #     use_stemmer=True,
    # )
    # result = {f"ROUGE_{k}": round(v * 100, 4) for k, v in result.items()}
    # prediction_lens = [np.count_nonzero(pred != pad_token_id) for pred in preds]
    # result["gen_len"] = np.mean(prediction_lens)

    print(f"Calculating bleu2 for {len(preds)} predictions")
    bleu2, _ = calculate_bleu2(preds, labels, smooth=True)
    bleu2 = {
        f"BLEU2_{k}": str(v) if isinstance(v, list) else v for k, v in bleu2.items()
    }
    result = {**result, **bleu2}
    result_bleu = {}
    if metric_path is not None:
        if any([len(pred) > 0 for pred in preds]) and any([
            len(label) > 0 for label in labels
        ]):
            print(f"Calculating bleu for {len(preds)} predictions")
            result_bleu = metric_bleu.compute(
                predictions=preds,
                references=labels,
                smooth=True,
            )
            result_bleu["bleuP"] = round(result_bleu["bleu"] * 100, 4)
            result_bleu = {
                f"BLEU_{k}": str(v) if isinstance(v, list) else v
                for k, v in result_bleu.items()
            }
        else:
            print(
                f"Skipping BLEU computation as preds is empty: \n {preds[:20]} \n labels: \n {labels[:20]}"
            )
            result_bleu = {
                "BLEU_bleu": -1.0,
                "BLEU_bleuP": -1.0,
                "BLEU_brevity_penalty": -1.0,
                "BLEU_length_ratio": -1.0,
                "BLEU_precisions": -1.0,
                "BLEU_reference_length": -1.0,
                "BLEU_translation_length": -1.0,
            }
    if metric_path is not None:
        result = {**result, **result_bleu}

    # if has_codebleu:
    #     cb_results = calc_codebleu(
    #         [[l] for l in labels], preds, lang="python"
    #     )
    #     cb_results["codebleuP"] = cb_results["codebleu"] * 100
    #     result = {**result, **cb_results}

    return result


if __name__ == "__main__":
    main()
