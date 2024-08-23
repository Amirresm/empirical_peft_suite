import logging
import sys
import transformers
import datasets

logger = logging.getLogger(__name__)

def setup_logging(logger, get_process_log_level):
    # Setup logging
    logging.basicConfig(
        format="==> %(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = get_process_log_level
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
