# Translation with Seq2Seq and Transformer

This project implements neural machine translation models—Seq2Seq, Seq2Seq with Attention, and Transformer—for translating Korean to English using a parallel corpus of 50,000 sentence pairs. Built with PyTorch, it uses Konlpy's Okt for Korean tokenization and NLTK for English tokenization. Optimized for Windows compatibility with JSON data loading.

## Features
- **Models**: Seq2Seq (LSTM-based), Seq2Seq with Attention, Transformer
- **Dataset**: 50,000 Korean-English sentence pairs (JSON)
- **Preprocessing**: Okt (Korean morphemes), NLTK (English words), `min_freq=5`
- **Training**: Hyperparameter tuning, `tqdm` progress, loss plots
- **Optimization**: GPU support, data caching, gradient clipping, early stopping
- **Testing**: Model saving and translation testing

## Prerequisites
- Python 3.8+
- PyTorch 2.0+ (CUDA recommended)
- Konlpy (Okt)
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
   venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
   pip install konlpy nltk tqdm matplotlib
   ```

4. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   ```

## Dataset
- **Source**: 50,000 Korean-English sentence pairs
- **Format**: JSON files (`train_set.json`, `valid_set.json`)
- **Structure**:
   ```json
   {
       "data": [
           {"ko": "안녕하세요", "en": "Hello"},
           ...
       ]
   }
   ```
- **Files**:
   - Train: `data/일상생활및구어체_한영_train_set.json` (50,000 pairs)
   - Valid: `data/일상생활및구어체_한영_valid_set.json` (1,000 pairs)
- **Preprocessing**:
   - Korean: Okt morpheme tokenization
   - English: NLTK word tokenization, lowercase
   - Vocabulary: `min_freq=5`, `<UNK>` for rare words

Place JSON files in the `data/` directory and update paths in `main.py`:
```python
train_set_path = 'data/일상생활및구어체_한영_train_set.json'
valid_set_path = 'data/일상생활및구어체_한영_valid_set.json'
```

## Usage

1. Prepare dataset:
   - Ensure JSON files are in `data/`.
   - Update `train_set_path` and `valid_set_path` in `main.py` if needed.

2. Run training:
   ```bash
   python main.py
   ```
   - Trains Seq2Seq, Seq2SeqAttention, Transformer
   - Saves models to `models/`
   - Saves plots: `{model}_loss_plot.png`

3. Hyperparameters:
   ```python
   hyperparams_list = [
       {'embed_size': 128, 'hidden_size': 256, 'num_layers': 2, 'dropout': 0.3, 'num_epochs': 20, 'batch_size': 32, 'learning_rate': 0.001, 'heads': 4},
       {'embed_size': 256, 'hidden_size': 512, 'num_layers': 3, 'dropout': 0.5, 'num_epochs': 20, 'batch_size': 32, 'learning_rate': 0.0005, 'heads': 4}
   ]
   ```

4. Test translation:
   ```python
   from main import load_and_translate
   print(load_and_translate('Seq2SeqAttention', 'models/Seq2SeqAttention_embed128_batch32_lr0.001.pt', 'src_vocab.pkl', 'tgt_vocab.pkl', '안녕하세요'))
   ```

## Outputs
- **Vocabulary**: `src_vocab.pkl`, `tgt_vocab.pkl`
- **Models**: `models/{model}_embed{size}_batch{size}_lr{rate}.pt`
- **Plots**: `Seq2Seq_loss_plot.png`, etc.
- **Test Loss**: Printed per experiment
- **Translations**: Printed for sample sentences

## Project Structure
```
Translation-with-Seq2Seq-and-transformer/
├── data/
│   ├── 일상생활및구어체_한영_train_set.json
│   ├── 일상생활및구어체_한영_valid_set.json
├── models/
│   ├── Seq2Seq_embed128_batch32_lr0.001.pt
│   └── ...
├── main.py
├── src_vocab.pkl
├── tgt_vocab.pkl
├── Seq2Seq_loss_plot.png
├── README.md
```

## Performance
- **Dataset**: 50,000 train + 1,000 valid pairs
- **Hardware**: Windows, NVIDIA GPU or CPU
- **Training Time (GPU)**:
  - Seq2Seq: ~1-2 min/epoch
  - Seq2SeqAttention: ~1.5-2.5 min/epoch
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
- Memory errors: Reduce `batch_size` to 16
- Preprocess Korean:
  ```python
  from konlpy.tag import Okt
  okt = Okt()
  sentence = " ".join(okt.morphs("안녕하세요"))
  ```
- Okt is slower but Windows-friendly.

## Contributing
Submit issues/PRs for:
- New models
- Additional languages
- Optimizations

## License
MIT License