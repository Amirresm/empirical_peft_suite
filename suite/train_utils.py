def handle_metrics(trainer, prefix, metrics, sample_count = None):
    if sample_count is not None:
        metrics[f"{prefix}_samples"] = sample_count
    trainer.log_metrics(prefix, metrics)
    trainer.save_metrics(prefix, metrics)

