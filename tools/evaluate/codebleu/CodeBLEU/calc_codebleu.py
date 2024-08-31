# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding:utf-8 -*-
import os
from .bleu import corpus_bleu as bleu_corpus_bleu
from .weighted_ngram_match import corpus_bleu as weighted_ngram_match_corpus_bleu
from .syntax_match import corpus_syntax_match
from .dataflow_match import corpus_dataflow_match


def calc_codebleu(predictions, references, lang, weights=[0.25, 0.25, 0.25, 0.25]):
    alpha, beta, gamma, theta = weights

    # # preprocess inputs
    # pre_references = [
    #     [x.strip() for x in open(file, "r", encoding="utf-8").readlines()]
    #     for file in args.refs
    # ]
    # predictions = [x.strip() for x in open(args.hyp, "r", encoding="utf-8").readlines()]

    # for i in range(len(pre_references)):
    #     assert len(predictions) == len(pre_references[i])

    # references = []
    # for i in range(len(predictions)):
    #     ref_for_instance = []
    #     for j in range(len(pre_references)):
    #         ref_for_instance.append(pre_references[j][i])
    #     references.append(ref_for_instance)
    # assert len(references) == len(pre_references) * len(predictions)

    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in predictions]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu_corpus_bleu(tokenized_refs, tokenized_hyps)

    # calculate weighted ngram match
    keywords = [
        x.strip()
        for x in open(os.path.join(os.path.dirname(__file__), "keywords", lang + ".txt"), "r", encoding="utf-8").readlines()
    ]

    def make_weights(reference_tokens, key_word_list):
        return {
            token: 1 if token in key_word_list else 0.2 for token in reference_tokens
        }

    tokenized_refs_with_weights = [
        [
            [reference_tokens, make_weights(reference_tokens, keywords)]
            for reference_tokens in reference
        ]
        for reference in tokenized_refs
    ]

    weighted_ngram_match_score = weighted_ngram_match_corpus_bleu(
        tokenized_refs_with_weights, tokenized_hyps
    )

    # calculate syntax match
    syntax_match_score = corpus_syntax_match(references, predictions, lang)

    # calculate dataflow match
    dataflow_match_score = corpus_dataflow_match(
        references, predictions, lang
    )

    print(
        "ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}".format(
            ngram_match_score,
            weighted_ngram_match_score,
            syntax_match_score,
            dataflow_match_score,
        )
    )

    code_bleu_score = (
        alpha * ngram_match_score
        + beta * weighted_ngram_match_score
        + gamma * syntax_match_score
        + theta * dataflow_match_score
    )

    results = {
        "codebleu": code_bleu_score,
        "ngram_match": ngram_match_score,
        "weighted_ngram_match": weighted_ngram_match_score,
        "syntax_match": syntax_match_score,
        "dataflow_match": dataflow_match_score,
    }

    print("CodeBLEU score: ", code_bleu_score)

    return results
