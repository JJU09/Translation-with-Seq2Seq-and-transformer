import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
from tqdm import tqdm

# 학습 함수
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int, device: torch.device) -> Tuple[List[float], List[float]]:
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 3
    counter = 0

    for epoch in tqdm(range(num_epochs), desc="Training Epochs", leave=True):
        model.train()
        epoch_train_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(src, tgt[:, :-1])
            output = output.contiguous().view(-1, output.size(-1))
            tgt = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
                output = model(src, tgt[:, :-1])
                output = output.contiguous().view(-1, output.size(-1))
                tgt = tgt[:, 1:].contiguous().view(-1)
                loss = criterion(output, tgt)
                epoch_val_loss += loss.item()

        val_losses.append(epoch_val_loss / len(val_loader))
        tqdm.write(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                tqdm.write("Early stopping triggered!")
                break

    return train_losses, val_losses