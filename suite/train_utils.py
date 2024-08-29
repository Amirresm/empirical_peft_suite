from general_utits import log_metrics, save_metrics


def handle_metrics(prefix, metrics, output_dir, sample_count = None, trainer = None):
    if sample_count is not None:
        metrics[f"{prefix}_samples"] = sample_count
    if trainer is not None:
        trainer.log_metrics(prefix, metrics)
        trainer.save_metrics(prefix, metrics, output_dir)
    else:
        log_metrics(prefix, metrics)
        save_metrics(prefix, metrics, output_dir)

