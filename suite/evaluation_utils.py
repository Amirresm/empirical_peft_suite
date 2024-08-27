from torch import log
from bleu2.calc_bleu2 import calculate_bleu2
from logging_utils import logger

import nltk
import numpy as np

from text_utils import clean_whitespaces_generations

has_codebleu = False
# try:
#     from codebleu import calc_codebleu
#     has_codebleu = True
# except ImportError:
#     print("CodeBLEU not found", flush=True)


def compute_metrics_decoder_only_preprocess(preds, labels):
    new_preds = []
    new_labels = []
    for pred in preds:
        splits = pred.split('"""')
        if len(splits) == 3:
            new_preds.append(splits[2].strip())
        else:
            new_preds.append(pred)

    for label in labels:
        splits = label.split('"""')
        if len(splits) == 3:
            new_labels.append(splits[2].strip())
        else:
            new_labels.append(label)

    return new_preds, new_labels


def compute_metrics_encoder_decoder_preprocess(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def get_compute_metrics(
    tokenizer,
    ignore_pad_token_for_loss,
    metric_rouge,
    metric_bleu,
    is_decoder_only,
    metric_path,
):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        if is_decoder_only:
            labels = labels[:, 1:]
            preds = preds[:, :-1]
        if ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        if ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        inspect_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
        inspect_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        logger.info(
            f"\nCompute_metrics Preds example ({preds.shape}):\n{clean_whitespaces_generations(inspect_preds[0])}\n"
        )
        logger.info(
            f"\nCompute_metrics Labels example ({labels.shape}):\n{clean_whitespaces_generations(inspect_labels[0])}\n\n"
        )
        # for i in range(5):
        #     logger.info(
        #         f"\nCompute_metrics Preds example ({preds.shape}):\n{inspect_preds[i]}\n"
        #     )
        #     logger.info(
        #         f"\nCompute_metrics Labels example ({labels.shape}):\n{inspect_labels[i]}\n\n"
        #     )

        decoded_preds, decoded_labels = (
            compute_metrics_decoder_only_preprocess(decoded_preds, decoded_labels)
            if is_decoder_only
            else compute_metrics_encoder_decoder_preprocess(
                decoded_preds, decoded_labels
            )
        )

        result = metric_rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        result = {f"ROUGE_{k}": round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)

        # CodeBERT bleu metric
        bleu2, b2args = calculate_bleu2(decoded_preds, decoded_labels, smooth=True)
        bleu2 = {
            f"BLEU2_{k}": str(v) if isinstance(v, list) else v for k, v in bleu2.items()
        }
        result = {**result, **bleu2}
        result_bleu = {}
        if metric_path is not None:
            if any([len(decoded_pred) > 0 for decoded_pred in decoded_preds]) and any([
                len(decoded_label) > 0 for decoded_label in decoded_labels
            ]):
                result_bleu = metric_bleu.compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                    smooth=True,
                )
                result_bleu["bleuP"] = round(result_bleu["bleu"] * 100, 4)
                result_bleu = {
                    f"BLEU_{k}": str(v) if isinstance(v, list) else v
                    for k, v in result_bleu.items()
                }
            else:
                logger.info(
                    f"Skipping BLEU computation as decoded_preds is empty: \n {decoded_preds[:20]} \n decoded_labels: \n {decoded_labels[:20]}"
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
        #         [[l] for l in decoded_labels], decoded_preds, lang="python"
        #     )
        #     cb_results["codebleuP"] = cb_results["codebleu"] * 100
        #     result = {**result, **cb_results}

        return result

    return compute_metrics


def preprocess_logits_for_metrics(logits, _):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def calc_all_metrics(
    preds, labels, metric_rouge, metric_bleu, metric_path, pad_token_id
):
    result = metric_rouge.compute(
        predictions=preds,
        references=labels,
        use_stemmer=True,
    )
    result = {f"ROUGE_{k}": round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)

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
            logger.info(
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
