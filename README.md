Translation with Seq2Seq and Transformer
This project implements neural machine translation models—Seq2Seq, Seq2Seq with Attention, and Transformer—for translating Korean to English using a parallel corpus of 50,000 sentence pairs. Built with PyTorch, it uses Mecab for Korean tokenization and NLTK for English tokenization.
Features

Models: Seq2Seq (LSTM-based), Seq2Seq with Attention, and Transformer
Dataset: 50,000 Korean-English sentence pairs
Preprocessing: Tokenization with Mecab (Korean) and NLTK (English), vocabulary with min_freq=5
Training: Hyperparameter tuning, progress tracking with tqdm, loss visualization
Optimization: Data caching, gradient clipping, early stopping

Prerequisites

Python 3.8+
PyTorch 2.0+ (CUDA recommended for GPU)
Mecab (via konlpy)
NLTK
tqdm
matplotlib

Installation

Clone the repository:
git clone https://github.com/JJU09/Translation-with-Seq2Seq-and-transformer.git
cd Translation-with-Seq2Seq-and-transformer


Create a virtual environment (optional):
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


Install dependencies:
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install konlpy nltk tqdm matplotlib


Install Mecab (platform-specific):

Ubuntu:sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8
pip install python-mecab-ko


Windows/Mac: See Mecab installation guide


Download NLTK data:
import nltk
nltk.download('punkt')



Dataset

Source: Custom Korean-English parallel corpus (50,000 sentence pairs)
Format: JSON/CSV with ko (Korean) and en (English) fields
Preprocessing:
Korean: Tokenized into morphemes using Mecab
English: Tokenized into words using NLTK
Vocabulary: Words appearing ≥5 times (min_freq=5) included; others mapped to <UNK>



Place your dataset in data/ as data.json or data.csv. Example:
[
  {"ko": "안녕하세요", "en": "Hello"},
  {"ko": "저는 학생입니다", "en": "I am a student"}
]

Usage

Prepare the dataset:

Place dataset in data/ (e.g., data/data.json)
Update main.py to load data:import json
with open("data/data.json") as f:
    data = json.load(f)
src_sentences = [item["ko"] for item in data]
tgt_sentences = [item["en"] for item in data]




Run training:
python main.py


Trains Seq2Seq, Seq2Seq with Attention, and Transformer
Shows progress with tqdm
Saves loss plots as {model}_loss_plot.png


Hyperparameters:

Defined in main.py:hyperparams_list = [
    {"embed_size": 128, "hidden_size": 256, "num_layers": 2, "dropout": 0.3, "num_epochs": 20, "batch_size": 32, "learning_rate": 0.001, "heads": 4},
    {"embed_size": 256, "hidden_size": 512, "num_layers": 3, "dropout": 0.5, "num_epochs": 20, "batch_size": 32, "learning_rate": 0.0005, "heads": 4}
]


Adjust batch_size or num_epochs for your hardware


Outputs:

Vocabulary: src_vocab.pkl, tgt_vocab.pkl
Loss plots: Seq2Seq_loss_plot.png, etc.
Test loss: Printed after each experiment



Project Structure
Translation-with-Seq2Seq-and-transformer/
├── data/
│   └── data.json
├── main.py
├── src_vocab.pkl
├── tgt_vocab.pkl
├── Seq2Seq_loss_plot.png
├── README.md

Performance

Dataset: 50,000 sentence pairs
Hardware: NVIDIA RTX 3060 (12GB) or CPU
Training time (per epoch, GPU):
Seq2Seq: ~1-2 minutes
Seq2Seq with Attention: ~1.5-2.5 minutes
Transformer: ~2-3 minutes


Optimizations:
Data caching
Gradient clipping (max_norm=1.0)
Early stopping (patience=3)



Notes

Check GPU availability:import torch
print(torch.cuda.is_available())


Reduce batch_size to 16 if CUDA out of memory occurs
Preprocess Korean data with Mecab:from konlpy.tag import Mecab
mecab = Mecab()
src_sentences = [" ".join(mecab.morphs(s)) for s in src_sentences]


Vocabulary uses min_freq=5 for efficiency

Contributing
Submit issues or pull requests for:

New models
Additional languages
Better preprocessing
