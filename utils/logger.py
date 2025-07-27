import logging
import os

def setup_logger(
    log_file: str = "experiment.log",
    log_dir: str = "./logs",
    log_level: int = logging.INFO,
    name: str = "multimodal"
):
    """
    Set up a logger that writes both to stdout and a file.

    Args:
        log_file: Log filename.
        log_dir: Directory to store log file.
        log_level: Logging level.
        name: Logger name.
    Returns:
        logger: Configured logger object.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    # File handler
    file_path = os.path.join(log_dir, log_file)
    if not logger.hasHandlers():
        fh = logging.FileHandler(file_path)
        fh.setLevel(log_level)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

# Example usage:
if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Logger initialized and ready.")
    logger.warning("This is a warning.")
    logger.error("This is an error.")
    logger.debug("This is a debug message.")
