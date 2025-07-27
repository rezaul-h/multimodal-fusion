# Multimodal Information Fusion: Audio-Visual Object Recognition

This repository provides a modular and reproducible pipeline for multimodal object classification using **hybrid, tensor, and attention-based (FiLM) fusion** architectures, supporting advanced feature extraction and robust handling of real-world noisy data.  
**Supports: Xception, (x)LSTM, Ridgelet, SIFT, LBP, SC, PSSF, SMOTE, class weights, and more.**

## 📂 Project Structure

├── configs/ # YAML experiment configs (default, FiLM, xLSTM)

├── data/ # Raw and processed datasets (not included here)

├── features/ # Audio/image feature extraction scripts

├── fusion/ # Fusion layers (tensor, hybrid, FiLM)

├── models/ # Model architectures (vision, audio, fusion)

├── preprocessing/ # Preprocessing and augmentation scripts

├── scripts/ # Scripts for running experiments, evaluation, etc.

├── training/ # Training pipeline

├── utils/ # Utilities: config, logger, seed, SMOTE, class weights

├── logs/ # Log files (auto-generated)

├── results/ # Result tables, plots, YAMLs (auto-generated)

├── main.py # Main runner script (configurable)

├── requirements.txt

└── README.md


## 🚀 Getting Started

### 1. Clone the repository

git clone https://github.com/your-username/multimodal-fusion.git
cd multimodal-fusion


### 2. Install dependencies

pip install -r requirements.txt

### 3. Prepare the dataset

Dataset is available in 'https://drive.google.com/drive/folders/1aqMjK1U9Oas4OD5Fb8dSjrMPTaBkq6fK?usp=sharing'

Structure:
/dataset/

├── class_1/

│    ├── audio/

│    └── image/

├── class_2/

│    ├── audio/

│    └── image/

└── ...

### 4. Configure your experiment
Edit or create YAML files in configs/.

### 5. Run the pipeline

python main.py --config configs/experiment_film.yaml

Or for xLSTM:

python main.py --config configs/experiment_xlstm.yaml

Logs, checkpoints, and results will be saved in the respective folders.

🧩 Key Features
- Fully modular: Plug in any backbone or fusion type by YAML config.

- Reproducible: All random seeds set; configs are versioned.

- Advanced augmentation: Image and audio pipelines, including noise, jitter, and more.

- Class imbalance handling: SMOTE and class-weighted loss.

- Comprehensive metrics: Accuracy, F1, PR AUC, MCC, confusion matrix, and more.

- Computational cost analysis: Compare model FLOPs, inference time, GPU memory.

- Visualization: Radar plots, learning curves, confusion matrices.

- Easy to extend: Add new fusion models, features, or datasets with minimal code changes.

📊 Results & Benchmarks
The framework achieves state-of-the-art performance on a 21-class audio-visual object dataset, surpassing classic tensor fusion and single-modality baselines.

🤝 Contributing
Pull requests are welcome!

Open issues for bug reports or feature suggestions.

Please follow PEP8 for Python code style.

🛠️ Troubleshooting
If you encounter memory issues, reduce batch_size or image size in config.

For reproducibility, ensure your CUDA/cuDNN, PyTorch, and all packages match requirements.txt.

For help with dataset formatting, see sample scripts in preprocessing/.

📧 Contact
Questions? Reach out by rezaulh603@gmail.com or open a GitHub issue.
