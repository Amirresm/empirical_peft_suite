from codebleu import calc_codebleu
from src.bleu2.calc_bleu2 import calculate_bleu2



def _custom_exact_match(preds, targets):
    strict_match = 0
    exact_match = 0
    loose_match = 0
    total = 0
    for pred, target in zip(preds, targets):
        total += 1
        if pred == target:
            strict_match += 1
        if pred.strip() == target.strip():
            exact_match += 1
        if pred.lower().strip() == target.lower().strip():
            loose_match += 1

    res = {
        "strict_match": strict_match / total,
        "exact_match": exact_match / total,
        "loose_match": loose_match / total,
    }
    
    return res["exact_match"]


def codebleu(preds, labels):
    result = {}
    weights = (0.25, 0.25, 0.25, 0.25)
    cb_results = calc_codebleu(
        [[label] for label in labels], preds, lang="python", weights=weights
    )
    custom_codebleu = (
        +weights[0] * cb_results["ngram_match_score"]
        + weights[1] * cb_results["weighted_ngram_match_score"]
        + weights[2] * cb_results["syntax_match_score"]
        + weights[3] * cb_results["dataflow_match_score"]
    )
    cb_results["codebleu"] = custom_codebleu
    cb_results["codebleuP"] = cb_results["codebleu"] * 100
    result = {**result, **cb_results}

    res = {}
    for k, v in result.items():
        res[f"CODEBLEU_{k}"] = v

    return cb_results["codebleuP"]


def bleu(preds, labels):
    results, _ = calculate_bleu2(preds, labels, smooth=True)
    results["bleuP"] = round(results["bleu"] * 100, 4)
    return results["bleuP"]


def pairwise_metrics(str1, str2):
    cb = codebleu([str1], [str2])
    bl = bleu([str1], [str2])
    em = _custom_exact_match([str1], [str2])
    return {"codebleu": cb, "bleu": bl, "exact_match": em}
