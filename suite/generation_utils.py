import os
import numpy as np

from general_utits import ensure_path_exists
from logging_utils import logger


def generation_from_predict(
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
