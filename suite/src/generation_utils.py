import json
import os

import numpy as np
import torch
from accelerate.utils import tqdm
from transformers.trainer_utils import Dict

from constants import COMPLETION_COL, PROMPT_COL, DatasetInstances
from lib.codeeval.core.data import stream_jsonl
from lib.codeeval.human_eval.evaluation import evaluate_functional_correctness
from lib.codeeval.core import run_eval

from src.evaluation_utils import calc_all_metrics
from src.general_utits import ensure_path_exists
from src.logging_utils import logger
from src.text_utils import (
    fix_indents,
    get_humaneval_prompter,
    join_columns,
    split_column,
)


def generation_from_predict_encoder_decoder(
    tokenizer,
    preds,
    labels,
    raw_dataset,
    tokenized_dataset,
    save_path,
    ds_instance,
):
    samples = raw_dataset.select(range(len(tokenized_dataset)))
    prompts = []
    targets = []
    examples = []
    for sample in samples:
        prompt = sample[PROMPT_COL]
        target = sample[COMPLETION_COL]
        example = join_columns(prompt, target, ds_instance)
        prompts.append(input)
        targets.append(target)
        examples.append(example)

    raw_inputs = np.where(
        tokenized_dataset["input_ids"] != -100,
        tokenized_dataset["input_ids"],
        tokenizer.pad_token_id,
    )
    raw_inputs_token_count = []
    for i in range(len(raw_inputs)):
        raw_inputs_token_count.append(len(raw_inputs[i]))

    raw_inputs = tokenizer.batch_decode(
        raw_inputs,
        skip_special_tokens=True,
    )
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    preds_token_count = []
    for i in range(len(preds)):
        preds_token_count.append(len(preds[i]))

    preds = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True,
    )
    outputs = []
    token_counts = []
    for i in range(len(preds)):
        token_counts.append(
            (raw_inputs_token_count[i], preds_token_count[i], -1)
        )
        outputs.append(join_columns(raw_inputs[i], preds[i], ds_instance))

    pairs = [
        f"{index + 1}=========\n\
->Example:\n{example}\n\
->Prompt:\n{raw_input}\n\
->Target:\n{label}\n\
->Pred:\n{pred}\n\
->Output:\n{output}\n\
->Inp_Tokens:\n{token_count[0]}\n\
->Out_Tokens:\n{token_count[1]}\n\
->New_Tokens:\n{token_count[2]}\n\
--\n\n"
        for example, raw_input, label, pred, output, token_count, index in zip(
            examples,
            raw_inputs,
            labels,
            preds,
            outputs,
            token_counts,
            range(len(preds)),
        )
    ]
    output_prediction_file = os.path.join(
        save_path,
        "generated_predictions.txt",
    )
    ensure_path_exists(save_path)
    with open(output_prediction_file, "w") as writer:
        writer.write("\n".join(pairs))

    logger.info(f"{len(pairs)} predictions saved to {output_prediction_file}")


def generation_decoder_only(
    model,
    tokenizer,
    raw_dataset,
    max_predict_samples,
    max_source_length,
    max_new_tokens,
    padding,
    save_path,
    metric_rouge,
    metric_bleu,
    metric_path,
    batch_size,
    ds_instance,
    is_decoder_only,
    save_path_prefix=None,
):
    samples = raw_dataset.select(range(max_predict_samples))
    prompts = []
    targets = []
    examples = []
    for sample in samples:
        prompt = sample[PROMPT_COL]
        target = sample[COMPLETION_COL]
        example = join_columns(prompt, target, ds_instance)
        prompts.append(prompt)
        targets.append(target)
        examples.append(example)

    outputs = []
    preds = []
    token_counts = []
    loop_range = (
        len(prompts) // batch_size
        if len(prompts) % batch_size == 0
        else (len(prompts) // batch_size) + 1
    )
    model.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast():
        for i in range(loop_range):
            logger.info(
                f"Generation progress (batch_size={batch_size}): {i + 1}/{loop_range}"
            )
            index = i * batch_size
            end_index = min(index + batch_size, len(prompts))
            if index >= end_index:
                break
            prompts_encoded = tokenizer(
                prompts[index:end_index],
                return_tensors="pt",
                max_length=max_source_length,
                padding=padding,
                truncation=True,
            )
            prompts_encoded = prompts_encoded.to(model.device)
            batch_outputs = model.generate(
                **prompts_encoded,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            for i, bo in enumerate(batch_outputs):
                inp_tokens = len(prompts_encoded[i])
                out_tokens = len(bo)
                new_tokens = out_tokens
                if is_decoder_only:
                    new_tokens = out_tokens - inp_tokens
                else:
                    new_tokens = -1
                token_counts.append((inp_tokens, out_tokens, new_tokens))

            batch_outputs = tokenizer.batch_decode(
                batch_outputs, skip_special_tokens=True
            )
            b_preds, b_outputs = process_decoder_only_generation(
                prompts[index:end_index],
                batch_outputs,
                ds_instance=ds_instance,
                is_decoder_only=is_decoder_only,
            )
            for i, btch in enumerate(zip(b_preds, b_outputs)):
                preds.append(btch[0])
                outputs.append(btch[1])
                logger.info(
                    f"{index + i}===\nInput:\n{prompts[index + i]}\nPred:\n{btch[0]}\nGold:\n{targets[index + i]}"
                )

    pairs = [
        f"{index + 1}=========\n\
->Example:\n{example}\n\
->Prompt:\n{prompt}\n\
->Target:\n{target}\n\
->Pred:\n{pred}\n\
->Output:\n{output}\n\
->Inp_Tokens:\n{token_count[0]}\n\
->Out_Tokens:\n{token_count[1]}\n\
->New_Tokens:\n{token_count[2]}\n\
--\n\n"
        for example, prompt, target, pred, output, token_count, index in zip(
            examples,
            prompts,
            targets,
            preds,
            outputs,
            token_counts,
            range(len(preds)),
        )
    ]

    output_prediction_file = os.path.join(
        save_path,
        (
            f"{save_path_prefix}_generations.txt"
            if save_path_prefix is not None
            else "generated_generations.txt"
        ),
    )
    ensure_path_exists(save_path)
    with open(output_prediction_file, "w") as writer:
        writer.write("\n".join(pairs))

    logger.info(f"{len(pairs)} generations saved to {output_prediction_file}")
    results = calc_all_metrics(
        preds,
        targets,
        metric_rouge,
        metric_bleu,
        metric_path,
        tokenizer.pad_token_id,
    )
    # add generate prefix to keys in results
    res = {}
    for key in results.keys():
        res[f"generate_{key}"] = results[key]

    results = res
    logger.info(f"Generation Results:\n{results}")

    return results


def process_decoder_only_generation(
    prompts, outputs, ds_instance, is_decoder_only
):
    preds = []
    outs = []
    if is_decoder_only:
        for i in range(len(outputs)):
            input = prompts[i]
            output = outputs[i]

            outs.append(output)

            pred = output.split(input)
            if len(pred) == 2:
                pred = pred[1]
            else:
                _, pred = split_column(output, ds_instance)
            preds.append(pred)
    else:
        preds = outputs
        for i in range(len(outputs)):
            outs.append(join_columns(prompts[i], outputs[i], ds_instance))

    return preds, outs


def get_generate_batch_completion(prompter, join_prompt_and_completion):
    @torch.inference_mode()
    def generate_batch_completion(
        model, tokenizer, prompt, do_padding
    ) -> list[str]:
        input_batch = [prompter(p) for p in prompt]
        if do_padding:
            inputs = tokenizer(
                input_batch,
                return_tensors="pt",
                max_length=256,
                padding="max_length",
                truncation=True,
            ).to(model.device)
        else:
            inputs = tokenizer(input_batch, return_tensors="pt").to(
                model.device
            )

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        batch_completions = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        res = [fix_indents(completion) for completion in batch_completions]
        res = batch_completions
        if join_prompt_and_completion:
            res = [f"{input_batch[i]}\n{res[i]}" for i in range(len(res))]
        for i in range(len(res)):
            logger.info(
                f"=================\n->Prompt:\n{prompt[i]}\n->Completion:\n{res[i]}"
            )
        return res

    return generate_batch_completion


ROOT = os.path.dirname(os.path.abspath(__file__))
HUMAN_EVAL = os.path.join(ROOT, "lib", "codeeval", "data", "HumanEval.jsonl.gz")
HUMAN_EVAL_R = os.path.join(
    ROOT, "lib", "codeeval", "data", "humaneval-r.jsonl"
)


def read_humaneval_r_problems(
    evalset_file: str = HUMAN_EVAL_R,
) -> Dict[str, Dict]:
    return {task["name"]: task for task in stream_jsonl(evalset_file)}


def read_humaneval_problems(evalset_file: str = HUMAN_EVAL) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def run_humaneval(
    model,
    tokenizer,
    num_samples_per_task,
    max_new_tokens,
    output_dir,
    is_decoder_only,
    save_path,
    batch_size,
    prompt_mode,
    ds_instance,
    calc_passk=True,
):
    if num_samples_per_task > 0:
        out_path = os.path.join(output_dir, f"humaneval_{num_samples_per_task}")
        os.makedirs(out_path, exist_ok=True)
        out_path = f"{out_path}/eval.jsonl"
        prompter = get_humaneval_prompter(prompt_mode)
        generate_batch_completion = get_generate_batch_completion(
            prompter=prompter,
            join_prompt_and_completion=not is_decoder_only
            and ds_instance != DatasetInstances.MULTIPLT,
        )

        if ds_instance == DatasetInstances.MULTIPLT:
            problems = read_humaneval_r_problems()
            logger.info(
                f"Running humaneval-R-{num_samples_per_task}, output to {out_path}"
            )
        else:
            problems = read_humaneval_problems()
            logger.info(
                f"Running humaneval-{num_samples_per_task}, output to {out_path}"
            )

        samples = run_eval(
            model,
            tokenizer,
            num_samples_per_task,
            out_path,
            generate_batch_completion,
            problems=problems,
            batch_size=batch_size,
        )

        pairs = [
            f"{index + 1}=========\n\
->Task:\n{sample['task_id']}\n\
->Completion:\n{sample['completion']}\n\
--\n\n"
            for sample, index in zip(
                samples,
                range(len(samples)),
            )
        ]

        output_prediction_file = os.path.join(
            save_path,
            "generated_humaneval.txt",
        )
        ensure_path_exists(save_path)
        with open(output_prediction_file, "w") as writer:
            writer.write("\n".join(pairs))

        logger.info(
            f"{len(pairs)} generations saved to {output_prediction_file}"
        )

        if ds_instance == DatasetInstances.MULTIPLT:
            logger.info(ds_instance)
            pbar = tqdm(total=len(samples))
            for sample in samples:
                task_id = sample["task_id"]
                problem = problems[task_id]
                if is_decoder_only:
                    _, completion = split_column(
                        sample["completion"], ds_instance
                    )
                else:
                    completion = sample["completion"]
                out_dict = {
                    "name": problem["name"],
                    "language": problem["language"],
                    "prompt": problem["prompt"],
                    "tests": problem["tests"],
                    "completions": [completion],
                    "stop_tokens": problem["stop_tokens"],
                }
                output_problems_dir = os.path.join(
                    save_path,
                    "humaneval_r_problems",
                )
                output_problems_file = os.path.join(
                    output_problems_dir,
                    f"{task_id}.json",
                )
                ensure_path_exists(output_problems_dir)
                out_dict = json.dumps(out_dict)
                with open(output_problems_file, "w") as writer:
                    writer.write(out_dict)

                pbar.update(1)
                pbar.set_description(f"Saving {task_id}")

        if ds_instance == DatasetInstances.SPP and calc_passk:
            results, details = evaluate_functional_correctness(
                sample_file=out_path,
            )
            pairs = [
                f"{index + 1}=========\n\
->Task:\n{sample['task_id']}\n\
->Passed:\n{details[sample['task_id']][0]}\n\
->Result:\n{details[sample['task_id']][1]}\n\
->Completion:\n{sample['completion']}\n\
--\n\n"
                for sample, index in zip(
                    samples,
                    range(len(samples)),
                )
            ]

            output_prediction_file = os.path.join(
                save_path,
                "generated_humaneval.txt",
            )
            ensure_path_exists(save_path)
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join(pairs))

            logger.info(
                f"{len(pairs)} generations saved to {output_prediction_file}"
            )
            results["pass@1_count"] = results["pass@1"] * 164

            # add humaneval prefix to keys in results
            res = {}
            for key in results.keys():
                res[f"humaneval_{key}"] = results[key]

            results = res
            logger.info(f"Pass@k: {results}")
            return results

        return None
