import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

from data import Vocab

# 평가 및 시각화
def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device, tgt_vocab: Vocab) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            output = output.contiguous().view(-1, output.size(-1))
            tgt = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output, tgt)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    return avg_loss

def plot_losses(train_losses: List[float], val_losses: List[float], model_name: str):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"{model_name} Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{model_name}_loss_plot.png")
    plt.close()