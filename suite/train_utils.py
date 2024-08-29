from suite.general_utits import log_metrics, save_metrics


def handle_metrics(prefix, metrics, output_dir, sample_count = None):
    if sample_count is not None:
        metrics[f"{prefix}_samples"] = sample_count
    log_metrics(prefix, metrics)
    save_metrics(prefix, metrics, output_dir)

