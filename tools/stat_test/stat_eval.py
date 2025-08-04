from scipy.stats import shapiro
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from itertools import combinations


def check_normality(data, alpha=0.05):
    """
    Check if the data is normally distributed
    :param data: list of data
    :param alpha: significance level
    :return: True if data is normally distributed, False otherwise
    """
    stat, p = shapiro(data)
    return p > alpha


def visualize_normality(data):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(data, kde=True)
    plt.title("Histogram")

    plt.subplot(1, 2, 2)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title("Q-Q Plot")

    plt.tight_layout()
    plt.show()


def evaluate_anova(data_list):
    f_stat, p_value = stats.f_oneway(*data_list)

    if p_value < 0.05:
        return True
    else:
        return False


def evaluate_pairwise_tukeyhsd(data_list, method_names):
    baseline_method = [
        method_name for method_name in method_names if "none" in method_name
    ][0]
    is_pass = True
    data = []
    labels = []
    for i, d in enumerate(data_list):
        data.extend(d)
        labels.extend([method_names[i]] * len(d))
    tukey_results = pairwise_tukeyhsd(data, labels)
    groups = tukey_results.groupsunique
    for i in range(len(groups)):
        method1 = groups[i]
        method2 = groups[(i + 1) % len(groups)]
        if (
            method1 == baseline_method
            and method2 != baseline_method
            or method1 != baseline_method
            and method2 == baseline_method
        ):
            if tukey_results.pvalues[i] > 0.05:
                is_pass = False
    return is_pass


def evaluate_pairwise_ttest(bleu_list, method_name_list):
    baseline_method = [
        method_name for method_name in method_name_list if "none" in method_name
    ][0]
    is_pass = True
    p_values = []
    comparisons = []
    for i, j in combinations(range(len(method_name_list)), 2):
        method1 = method_name_list[i]
        method2 = method_name_list[j]
        scores1 = bleu_list[i]
        scores2 = bleu_list[j]

        t_stat, p_value = stats.ttest_ind(scores1, scores2)
        p_values.append(p_value)
        comparisons.append((method1, method2))

    adjusted_pvals_bonferroni = multipletests(p_values, method="bonferroni")[1]
    adjusted_pvals_bh = multipletests(p_values, method="fdr_bh")[1]

    failed = []
    p_value_map_baseline = {baseline_method: -1}
    for idx, (method1, method2) in enumerate(comparisons):
        if method1 == baseline_method and method2 != baseline_method:
            other_method = method2
        elif method2 == baseline_method and method1 != baseline_method:
            other_method = method1
        else:
            continue

        if (
            other_method in p_value_map_baseline
            and p_value_map_baseline[other_method]
            != adjusted_pvals_bonferroni[idx]
        ):
            print(
                f"NOT MATCH: p-value for the same comparison: baseline='{baseline_method}' vs other='{other_method}'"
            )
        p_value_map_baseline[other_method] = adjusted_pvals_bonferroni[idx]

        if adjusted_pvals_bonferroni[idx] > 0.05:
            is_pass = False
            failed.append(
                f"Comparison: baseline='{baseline_method}' vs other='{other_method}': Bonferroni-adjusted p-value: {adjusted_pvals_bonferroni[idx]:.8g}"
            )

    return is_pass, failed, p_value_map_baseline


def significance_test(results_list, method_name_list, metric):
    anova_sig = evaluate_anova(results_list)
    if anova_sig:
        ttest_pass, failed, p_value_map_baseline = evaluate_pairwise_ttest(
            results_list, method_name_list
        )

        return True, failed, p_value_map_baseline

    else:
        return False, [], {"anova_fail": -2}


def evaluate_results(grouped_results):
    method_name_list = []
    em_list = []
    bleu_list = []
    codebleu_list = []

    summaries = []
    for results in grouped_results:
        task = results["task"]
        metrics = results["metrics"]
        method_name = f"{task.remark}-{task.peft_lib}-{task.peft}"
        method_name_list.append(method_name)
        summary = {"name": method_name, "scores": {}}
        for metric, data in metrics.items():
            if len(data) == 0:
                continue

            if metric == "exact_match":
                em_list.append(data)
            if metric == "bleu":
                bleu_list.append(data)
            if metric == "codebleu":
                codebleu_list.append(data)

            normal = check_normality(data)

            summary["scores"][metric] = {
                "average": sum(data) / len(data),
                "normal": normal,
            }

        summaries.append(summary)

    stat_results = {}
    p_value_map_baseline = None
    for metric_data, metric in [
        (bleu_list, "bleu"),
    ]:
        if len(metric_data) > 1:
            anova_res, ttest_failed, p_value_map_baseline = significance_test(
                metric_data, method_name_list, metric
            )
            stat_results[metric] = {
                "anova": anova_res,
                "ttest_failed": ttest_failed,
            }

    return summaries, stat_results, p_value_map_baseline
