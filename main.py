import argparse
import yaml
import os

from utils.config import load_yaml_config
from utils.seed import set_seed
from utils.logger import setup_logger
from training.train import train_model
from scripts.evaluate import evaluate_model
from scripts.extract_features import extract_all_features
from scripts.preprocess import preprocess_dataset

def main(config_path):
    # ===== 1. Load Config and Set Seed =====
    config = load_yaml_config(config_path)
    set_seed(config.get('random_seed', 42))

    # ===== 2. Initialize Logger =====
    logger = setup_logger(log_file="run.log", log_dir=config.get("logs_dir", "./logs"))
    logger.info(f"Loaded config from {config_path}")

    # ===== 3. Preprocess Dataset =====
    logger.info("Starting preprocessing...")
    preprocess_dataset(config)
    logger.info("Preprocessing completed.")

    # ===== 4. Feature Extraction =====
    logger.info("Extracting features for all modalities...")
    extract_all_features(config)
    logger.info("Feature extraction completed.")

    # ===== 5. Train Model =====
    logger.info(f"Training model: {config['model_type']} + {config.get('audio_branch', 'lstm')}")
    model, history = train_model(config, logger)
    logger.info("Training completed.")

    # ===== 6. Evaluate Model =====
    logger.info("Evaluating model on test set...")
    metrics = evaluate_model(model, config, logger)
    logger.info(f"Test Results: {metrics}")

    # ===== 7. Save Results =====
    results_dir = config.get("results_dir", "./results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "final_results.yaml")
    with open(results_file, "w") as f:
        yaml.safe_dump(metrics, f)
    logger.info(f"Saved final metrics to {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Information Fusion Pipeline")
    parser.add_argument('--config', type=str, default="configs/default.yaml", help="Path to experiment config YAML")
    args = parser.parse_args()
    main(args.config)
