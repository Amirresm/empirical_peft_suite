import os
import numpy as np
import torch

from lib.codeeval.human_eval.evaluation import evaluate_functional_correctness
from lib.codeeval.core import run_eval

from evaluation_utils import calc_all_metrics
from general_utits import ensure_path_exists
from logging_utils import logger
from text_utils import (
    clean_whitespaces_generations,
    create_llama_prompt,
    csn_split,
    fix_indents,
    spp_split,
)


def generation_from_predict_encoder_decoder(
    tokenizer,
    preds,
    labels,
    raw_dataset,
    tokenized_dataset,
    text_column,
    summary_column,
    save_path,
):
    source = raw_dataset.select(range(len(tokenized_dataset)))
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    raw_inputs = tokenized_dataset["input_ids"]
    raw_inputs = np.where(
        raw_inputs != -100,
        raw_inputs,
        tokenizer.pad_token_id,
    )
    raw_labels = tokenized_dataset["labels"]
    raw_labels = [
        [tokenizer.pad_token_id if t == -100 else t for t in rl] for rl in raw_labels
    ]
    raw_inputs = tokenizer.batch_decode(
        raw_inputs,
        skip_special_tokens=True,
    )
    raw_labels = tokenizer.batch_decode(
        raw_labels,
        skip_special_tokens=True,
    )
    labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )
    preds = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True,
    )
    pairs = [
        f"{index + 1}=========\n->Original Input:\n{or_inp}\n->Original Target:\n{or_tgt}\n->Reconstructed Target:\n{orig}\n->Reconstructed Predication:\n{pred}\n->Raw Input:\n{raw_input}\n->Raw Target:\n{raw_label}\n--\n"
        for pred, orig, or_inp, or_tgt, raw_input, raw_label, index in zip(
            preds,
            labels,
            source[text_column],
            source[text_column],
            # source[summary_column],
            raw_inputs,
            raw_labels,
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
    text_column,
    summary_column,
    max_predict_samples,
    max_source_length,
    max_new_tokens,
    padding,
    save_path,
    metric_rouge,
    metric_bleu,
    metric_path,
):
    samples = raw_dataset.select(range(max_predict_samples))
    prompts = []
    targets = []
    for i, sample in enumerate(samples):
        input = sample[text_column]
        target = (
            sample[text_column] if summary_column == "NONE" else sample[summary_column]
        )
        if isinstance(input, list):
            input = " ".join(input)
        if isinstance(target, list):
            target = " ".join(target)

        if summary_column == "NONE":
            input, target = spp_split(input)
        else:
            input = f"# code:\n{input}\n# summarize:\n"
            target = target

        prompts.append(input)
        targets.append(target)

    outputs = []
    token_counts = []
    batch_size = 4
    loop_range = (
        len(prompts) // batch_size
        if len(prompts) % batch_size == 0
        else (len(prompts) // batch_size) + 1
    )
    for i in range(loop_range):
        logger.info(f"Generation progress: {i + 1}/{len(prompts) // batch_size}")
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
        model.eval()
        with torch.inference_mode(), torch.cuda.amp.autocast():
            batch_outputs = model.generate(
                **prompts_encoded,
                max_new_tokens=max_new_tokens,
                # do_sample=True,
                # top_k=50,
                # top_p=0.95,
            )
            for i, bo in enumerate(batch_outputs):
                token_counts.append((len(prompts_encoded[i]), len(bo)))

            batch_outputs = tokenizer.batch_decode(
                batch_outputs, skip_special_tokens=True
            )
            for i, bo in enumerate(batch_outputs):
                outputs.append(bo)
                logger.info(
                    f"{index + i}===\nInput:\n{prompts[index + i]}\nPred:\n{bo}\nGold:\n{targets[index + i]}"
                )

    preds = process_decoder_only_generation(prompts, outputs)
    pairs = [
        f"{index + 1}=========\n\
->Prompt:\n{prompt}\n\
->Target:\n{target}\n\
->Pred:\n{pred}\n\
->Output:\n{output}\n\
->In TCount: {token_count[0]}\n\
->Out TCount: {token_count[1]}\n\
--\n\n"
        for prompt, target, pred, output, token_count, index in zip(
            prompts,
            targets,
            preds,
            outputs,
            token_counts,
            range(len(outputs)),
        )
    ]

    output_prediction_file = os.path.join(
        save_path,
        "generated_generations.txt",
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
    logger.info(f"Generation Results:\n{results}")

    return results


def process_decoder_only_generation(inputs, outputs):
    preds = []
    for i in range(len(outputs)):
        input = inputs[i]
        output = outputs[i]

        pred = output.split(input)
        if len(pred) == 1:
            pred = ""
        else:
            pred = pred[1]
        preds.append(pred)

    return preds


@torch.inference_mode()
def generate_batch_completion(model, tokenizer, prompt, batch_size) -> list[str]:
    prompt_input = create_llama_prompt(prompt)
    input_batch = [prompt_input for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **inputs,
        # use_cache=True,
        max_new_tokens=200,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    batch_completions = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    # res = [filter_code(fix_indents(extract_code(completion))) for completion in batch_completions]
    res = [fix_indents(completion) for completion in batch_completions]
    res = batch_completions
    logger.info(f"Generated completions prompt:\n {prompt}")
    # logger.info(f"Generated completions raw:\n {batch_completions[0]}")
    logger.info(f"Generated completions example:\n {res[0]}")
    return res


def run_humaneval(model, tokenizer, num_samples_per_task, output_dir, calc_passk=True):
    if num_samples_per_task > 0:
        out_path = os.path.join(output_dir, f"humaneval_{num_samples_per_task}")
        os.makedirs(out_path, exist_ok=True)
        out_path = f"{out_path}/eval.jsonl"
        logger.info(f"Running humaneval-{num_samples_per_task}, output to {out_path}")
        run_eval(
            model,
            tokenizer,
            num_samples_per_task,
            out_path,
            generate_batch_completion,
            True,
            # limit=10,
        )

        if calc_passk:
            pass_at_k = evaluate_functional_correctness(
                sample_file=out_path,
                # k=[1, 10, 100],
                # n_workers=4,
                # timeout=3.0,
            )
            pass_at_k["pass@1_count"] = pass_at_k["pass@1"] * 164

            logger.info(f"Pass@k: {pass_at_k}")
            logger.info(f"Correct count {pass_at_k["pass@1"] * 164}")

            return pass_at_k

        return None
