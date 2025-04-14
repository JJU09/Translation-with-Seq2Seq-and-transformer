# Translation with Seq2Seq and Transformer

This project implements neural machine translation models—Seq2Seq, Seq2Seq with Attention, and Transformer—for translating Korean to English using a parallel corpus of 50,000 sentence pairs. Built with PyTorch, it uses Mecab for Korean tokenization and NLTK for English tokenization.

## Features
- **Models**: Seq2Seq (LSTM-based), Seq2Seq with Attention, Transformer
- **Dataset**: 50,000 Korean-English sentence pairs
- **Preprocessing**: Mecab (Korean), NLTK (English), `min_freq=5` vocabulary
- **Training**: Hyperparameter tuning, `tqdm` progress bars, loss plots
- **Optimization**: Data caching, gradient clipping, early stopping

## Prerequisites
- Python 3.8+
- PyTorch 2.0+ (CUDA recommended)
- Mecab (via konlpy)
- NLTK
- tqdm
- matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JJU09/Translation-with-Seq2Seq-and-transformer.git
   cd Translation-with-Seq2Seq-and-transformer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
   pip install konlpy nltk tqdm matplotlib
   ```

4. Install Mecab:
   - Ubuntu:
     ```bash
     sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8
     pip install python-mecab-ko
     ```
   - Windows/Mac:
     ```bash
     # Follow: https://github.com/konlpy/konlpy
     ```

5. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   ```

## Dataset
- **Source**: 50,000 Korean-English sentence pairs
- **Format**: JSON/CSV (`ko` and `en` fields)
- **Preprocessing**:
  - Korean: Mecab morpheme tokenization
  - English: NLTK word tokenization
  - Vocabulary: `min_freq=5`, others mapped to `<UNK>`

Place dataset in `data/` as `data.json` or `data.csv`. Example:
```json
[
  {"ko": "안녕하세요", "en": "Hello"},
  {"ko": "저는 학생입니다", "en": "I am a student"}
]
```

## Usage

1. Prepare dataset:
   - Place in `data/data.json`
   - Update `main.py`:
     ```python
     import json
     with open("data/data.json") as f:
         data = json.load(f)
     src_sentences = [item["ko"] for item in data]
     tgt_sentences = [item["en"] for item in data]
     ```

2. Run training:
   ```bash
   python main.py
   ```
   - Trains Seq2Seq, Seq2Seq with Attention, Transformer
   - Shows `tqdm` progress
   - Saves plots: `{model}_loss_plot.png`

3. Hyperparameters in `main.py`:
   ```python
   hyperparams_list = [
       {"embed_size": 128, "hidden_size": 256, "num_layers": 2, "dropout": 0.3, "num_epochs": 20, "batch_size": 32, "learning_rate": 0.001, "heads": 4},
       {"embed_size": 256, "hidden_size": 512, "num_layers": 3, "dropout": 0.5, "num_epochs": 20, "batch_size": 32, "learning_rate": 0.0005, "heads": 4}
   ]
   ```

4. Outputs:
   - Vocabulary: `src_vocab.pkl`, `tgt_vocab.pkl`
   - Plots: `Seq2Seq_loss_plot.png`, etc.
   - Test loss: Printed

## Project Structure
```
Translation-with-Seq2Seq-and-transformer/
├── data/
│   └── data.json
├── main.py
├── src_vocab.pkl
├── tgt_vocab.pkl
├── Seq2Seq_loss_plot.png
├── README.md
```

## Performance
- **Dataset**: 50,000 pairs
- **Hardware**: NVIDIA RTX 3060 or CPU
- **Training time (GPU)**:
  - Seq2Seq: ~1-2 min/epoch
  - Seq2Seq with Attention: ~1.5-2.5 min/epoch
  - Transformer: ~2-3 min/epoch
- **Optimizations**:
  - Data caching
  - Gradient clipping (`max_norm=1.0`)
  - Early stopping (patience=3)

## Notes
- Check GPU:
  ```python
  import torch
  print(torch.cuda.is_available())
  ```
- Memory errors: Set `batch_size=16`
- Preprocess Korean:
  ```python
  from konlpy.tag import Mecab
  mecab = Mecab()
  src_sentences = [" ".join(mecab.morphs(s)) for s in src_sentences]
  ```
- Vocabulary: `min_freq=5`

## Contributing
Submit issues/PRs for:
- New models
- More languages
- Optimization

## License
MIT License