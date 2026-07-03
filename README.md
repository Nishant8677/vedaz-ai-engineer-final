# Vedaz AI Engineer Assessment: Qwen Fine-Tuning

This repository contains the complete end-to-end pipeline for fine-tuning a Qwen LLM on the Vedaz astrological chat dataset. It focuses on compassionate, non-fatalistic, and safe responses.

## 📂 Project Architecture

```
vedaz-ai-engineer-final/
│
├── data/                  # Contains raw dataset, cleaned jsonl, and train/val splits
├── docs/                  # Documentation (Hosting guide, etc.)
├── notebooks/             # Exploratory Data Analysis (EDA) Jupyter Notebooks
├── outputs/               # Saved models, checkpoints, metrics, and inference samples
├── training/              # Unsloth SFT training, inference, and evaluation scripts
├── utils/                 # Data cleaning and formatting scripts
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🚀 Quickstart

### 1. Environment Setup
Install the necessary dependencies. We highly recommend using a virtual environment.
```bash
pip install -r requirements.txt
```
*Note: The `requirements.txt` installs Unsloth which is highly optimized for GPU training. It requires an NVIDIA GPU and Linux/WSL.*

### 2. Data Preparation
The raw data is messy and contains a mix of JSON formatting. The pipeline extracts valid conversations and splits them.
```bash
# Cleans raw JSON into strict JSONL conversational format
python utils/data_cleaning.py

# Splits dataset into 90% train and 10% validation
python utils/prepare_dataset.py
```

### 3. Exploratory Data Analysis (EDA)
Open the notebook in `notebooks/01_dataset_analysis.ipynb` to visualize conversation lengths, role distributions, and check for missing fields.

### 4. Fine-Tuning
The training script uses **Unsloth** and **QLoRA** to fine-tune `Qwen/Qwen2.5-3B-Instruct` efficiently.
```bash
python training/train.py
```
This script will:
- Load the model in 4-bit quantization.
- Apply LoRA adapters to all linear layers.
- Train the model using `SFTTrainer`.
- Merge the adapters back into the base model and save it to `outputs/merged_model/`.

### 5. Inference & Evaluation
Test the base model vs the fine-tuned model side by side on edge-case prompts.
```bash
python training/inference.py
python training/evaluate.py
```
Outputs and metrics are stored in the `outputs/samples/` directory.

## 🌩️ Production Deployment (vLLM)
For a comprehensive guide on hosting this fine-tuned model on a VPS using vLLM, please refer to:
👉 [docs/hosting.md](docs/hosting.md)

---
*Built for the Vedaz AI Engineer Assessment.*
