from .bleu import bleuFromMaps, normalize, computeMaps2


def prepare_for_bleu2(preds_list, refs_list):
    preds = {}
    refs = {}
    for idx in range(len(preds_list)):
        preds[str(idx)] = preds_list[idx]
        refs[str(idx)] = refs_list[idx]
    return preds, refs


def calculate_bleu2(preds, refs, tokenizer=None, smooth=False):
    tokenizer = normalize if tokenizer is None else tokenizer
    bleu2_args = prepare_for_bleu2(preds, refs)
    (goldMap, predictionMap) = computeMaps2(
        bleu2_args[0], bleu2_args[1], tokenizer=tokenizer
    )
    bleu2_outputs = bleuFromMaps(goldMap, predictionMap, smooth=smooth)
    bleu2_outputs = {
        "precisions": bleu2_outputs[1:],
        "bleu": bleu2_outputs[0],
        "bleuP": bleu2_outputs[0] * 100,
    }
    return bleu2_outputs, bleu2_args
