import os
import yaml

def load_yaml_config(yaml_path="config.yaml"):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

class Config:
    # Paths
    DATASET_ROOT = './data/dataset'
    PREPROCESSED_ROOT = './data/preprocessed'
    FEATURES_ROOT = './data/features'
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'

    # Data
    IMG_SIZE = (128, 128)
    AUDIO_SR = 16000
    AUDIO_DURATION = 3.0
    NUM_CLASSES = 21

    # Training
    BATCH_SIZE = 64
    N_EPOCHS = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    VAL_SIZE = 0.05
    TEST_SIZE = 0.15
    RANDOM_SEED = 42

    # Model
    MODEL_TYPE = 'hybrid'  # hybrid, tensor, film
    USE_XLSTM = False      # Set True for xLSTM branch
    USE_RIDGELET = True

    # Hardware
    DEVICE = 'cuda' if (os.environ.get("CUDA_VISIBLE_DEVICES") is not None and os.system("nvidia-smi") == 0) else 'cpu'

config = Config()


