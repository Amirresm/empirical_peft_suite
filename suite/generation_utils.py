import os
import numpy as np
import torch

from codeeval.core import run_eval

from evaluation_utils import calc_all_metrics
from general_utits import ensure_path_exists
from logging_utils import logger
from text_utils import clean_whitespaces_generations, create_llama_prompt, fix_indents, spp_split


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
            source[summary_column],
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
    expected = []
    prompts = []
    descs = []
    targets = []
    for i, sample in enumerate(samples):
        input = sample[text_column]
        target = sample[text_column]
        input = '"""'.join(input.split('"""')[:2]) + '"""\n'
        sample = create_llama_prompt(
            input, is_training=False, eos_token=tokenizer.eos_token
        )
        expect = create_llama_prompt(
            target, is_training=False, eos_token=tokenizer.eos_token
        )
        prompts.append(sample)
        expected.append(expect)
        descs.append(input)
        targets.append(target)

    outputs = []
    token_counts = []
    batch_size = 4
    for i in range(len(prompts) // batch_size):
        logger.info(f"Generation progress: {i + 1}/{len(prompts) // batch_size}")
        index = i * batch_size
        prompts_encoded = tokenizer(
            prompts[index : index + batch_size],
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

    # logger.info("Cleaning outputs...")
    # outputs = list(map(clean_whitespaces_generations, outputs))

    # logger.info("Extracting code predictions...")
    # pred_codes = []
    # for output in outputs:
    #     # pred_code = output.split("[/INST] CODE:")[1].strip()
    #     pred_code = output.strip()
    #     try:
    #         pred_code = output.split("[/INST]")[1].strip()
    #     except:
    #         pass
    #     pred_codes.append(pred_code)
    # logger.info(f"Lenghts: outputs={len(outputs)} expecteds={len(expected)}")
    # pairs = [
    #     f"{index + 1}=========\n->Pred Code:\n{pcode}\n->Target Code:\n{tcode}\n->Instruction:\n{tdesc}\n->Reconstructed Predication:\n{pred}\n->Raw Input:\n{raw_input}\n--\n\n"
    #     for pcode, tcode, tdesc, pred, raw_input, index in zip(
    #         pred_codes,
    #         targets,
    #         descs,
    #         outputs,
    #         expected,
    #         range(len(outputs)),
    #     )
    # ]
    in_prompts, out_prompts, in_completions, out_completions = (
        process_decoder_only_generation(targets, outputs)
    )
    pairs = [
        f"{index + 1}=========\n->Gold input:\n{sample}\n->Gold prompt:\n{in_prompt}\n->Gold completion:\n{in_completion}\n\
                    ->Pred output:\n{output}\n->Pred prompt:\n{out_prompt}\n->Pred completion:\n{out_completion}\n\
                    ->In TCount: {token_count[0]}\n->Out TCount: {token_count[1]}\n Total TCount: {token_count[0] + token_count[1]}\n--\n\n"
        for sample, output, in_prompt, in_completion, out_prompt, out_completion, token_count, index in zip(
            targets,
            outputs,
            in_prompts,
            in_completions,
            out_prompts,
            out_completions,
            token_counts,
            range(len(outputs)),
        )
    ]

    generation_sample_size = 50
    generation_sample = pairs[:]
    # random.shuffle(generation_sample)
    generation_sample = "\n".join(generation_sample[:generation_sample_size])
    logger.info(
        f"Generation sample limit {generation_sample_size}:\n{generation_sample}"
    )
    output_prediction_file = os.path.join(
        save_path,
        "generated_predictions.txt",
    )
    ensure_path_exists(save_path)
    with open(output_prediction_file, "w") as writer:
        writer.write("\n".join(pairs))

    logger.info(f"{len(pairs)} generations saved to {output_prediction_file}")
    results = calc_all_metrics(
        out_completions,
        in_completions,
        metric_rouge,
        metric_bleu,
        metric_path,
        tokenizer.pad_token_id,
    )
    logger.info(f"Generation Results:\n{results}")


def process_decoder_only_generation(inputs, outputs):
    in_prompts, out_prompts, in_completions, out_completions = [], [], [], []
    for i in range(len(outputs)):
        input = inputs[i]
        output = outputs[i]
        in_prompt, in_completion = spp_split(input)
        out_prompt, out_completion = spp_split(output)
        in_prompts.append(in_prompt)
        in_completions.append(in_completion)
        out_prompts.append(out_prompt)
        out_completions.append(out_completion)

    return in_prompts, out_prompts, in_completions, out_completions

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


def run_humaneval(model, tokenizer, num_samples_per_task, output_dir):
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
        )

