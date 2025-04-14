import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict
from tqdm import tqdm
import pickle
import os
from konlpy.tag import Okt
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')


# 실험 함수
def run_experiment(model_type: str, src_vocab: Vocab, tgt_vocab: Vocab, train_dataset: TranslationDataset, val_dataset: TranslationDataset, test_dataset: TranslationDataset, hyperparams: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    embed_size = hyperparams.get("embed_size", 256)
    hidden_size = hyperparams.get("hidden_size", 512)
    num_layers = hyperparams.get("num_layers", 2)
    dropout = hyperparams.get("dropout", 0.3)
    num_epochs = hyperparams.get("num_epochs", 20)
    batch_size = hyperparams.get("batch_size", 32)
    learning_rate = hyperparams.get("learning_rate", 0.001)
    heads = hyperparams.get("heads", 4)

    print(f"Starting {model_type} with embed_size={embed_size}, batch_size={batch_size}, learning_rate={learning_rate}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)  # Windows
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    if model_type == "Seq2Seq":
        model = Seq2Seq(len(src_vocab.word2idx), len(tgt_vocab.word2idx), embed_size, hidden_size, num_layers, dropout).to(device)
    elif model_type == "Seq2SeqAttention":
        model = Seq2SeqAttention(len(src_vocab.word2idx), len(tgt_vocab.word2idx), embed_size, hidden_size, num_layers, dropout).to(device)
    else:
        model = Transformer(len(src_vocab.word2idx), len(tgt_vocab.word2idx), embed_size, num_layers, heads, dropout, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.word2idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_type}_embed{embed_size}_batch{batch_size}_lr{learning_rate}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparams': hyperparams,
        'src_vocab_size': len(src_vocab.word2idx),
        'tgt_vocab_size': len(tgt_vocab.word2idx)
    }, model_path)
    print(f"Model saved to {model_path}")

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            output = model(src, tgt[:, :-1])
            output = output.contiguous().view(-1, output.size(-1))
            tgt = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output, tgt)
            total_loss += loss.item()

    test_loss = total_loss / len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    plot_losses(train_losses, val_losses, model_type)
    return test_loss

# 손실 시각화
def plot_losses(train_losses: List[float], val_losses: List[float], model_name: str):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"{model_name} Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{model_name}_loss_plot.png")
    plt.show()
    # plt.close()

# 번역 함수
def translate_sentence(model: nn.Module, sentence: str, src_vocab: Vocab, tgt_vocab: Vocab, max_len: int, device: torch.device, max_output_len: int = 50) -> str:
    okt = Okt()
    tokenized_sentence = " ".join(okt.morphs(sentence))
    model.eval()
    src_tensor = src_vocab.sentence_to_tensor(tokenized_sentence, max_len).unsqueeze(0).to(device)
    tgt_indices = [tgt_vocab.word2idx["<SOS>"]]
    
    with torch.no_grad():
        for _ in range(max_output_len):
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)
            output = model(src_tensor, tgt_tensor, teacher_forcing_ratio=0.0)
            pred_token = output[:, -1, :].argmax(-1).item()
            tgt_indices.append(pred_token)
            if pred_token == tgt_vocab.word2idx["<EOS>"]:
                break
    
    return tgt_vocab.tensor_to_sentence(torch.tensor(tgt_indices))

# 모델 로드 및 번역
def load_and_translate(model_type: str, model_path: str, src_vocab_path: str, tgt_vocab_path: str, sentence: str, max_len: int = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    src_vocab = Vocab.load(src_vocab_path)
    tgt_vocab = Vocab.load(tgt_vocab_path)
    
    checkpoint = torch.load(model_path, map_location=device)
    hyperparams = checkpoint['hyperparams']
    embed_size = hyperparams['embed_size']
    hidden_size = hyperparams.get('hidden_size', 512)
    num_layers = hyperparams['num_layers']
    dropout = hyperparams['dropout']
    heads = hyperparams.get('heads', 4)
    
    if model_type == "Seq2Seq":
        model = Seq2Seq(
            src_vocab_size=checkpoint['src_vocab_size'],
            tgt_vocab_size=checkpoint['tgt_vocab_size'],
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type == "Seq2SeqAttention":
        model = Seq2SeqAttention(
            src_vocab_size=checkpoint['src_vocab_size'],
            tgt_vocab_size=checkpoint['tgt_vocab_size'],
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        model = Transformer(
            src_vocab_size=checkpoint['src_vocab_size'],
            tgt_vocab_size=checkpoint['tgt_vocab_size'],
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            device=device
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    translated = translate_sentence(model, sentence, src_vocab, tgt_vocab, max_len, device)
    return translated
