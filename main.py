import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

from data import Vocab, TranslationDataset
from models import Seq2Seq, Seq2SeqAttention, Transformer
from train import train_model
from evaluate import evaluate_model, plot_losses


# 하이퍼파라미터 실험
def run_experiment(model_type: str, src_vocab: Vocab, tgt_vocab: Vocab, train_dataset: TranslationDataset, val_dataset: TranslationDataset, test_dataset: TranslationDataset, hyperparams: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 하이퍼파라미터 설정
    embed_size = hyperparams.get("embed_size", 256)
    hidden_size = hyperparams.get("hidden_size", 512)
    num_layers = hyperparams.get("num_layers", 2)
    dropout = hyperparams.get("dropout", 0.3)
    num_epochs = hyperparams.get("num_epochs", 20)
    batch_size = hyperparams.get("batch_size", 64)
    learning_rate = hyperparams.get("learning_rate", 0.001)
    heads = hyperparams.get("heads", 8)

    print(f"Starting {model_type} with embed_size={embed_size}, batch_size={batch_size}, learning_rate={learning_rate}")

    # 데이터 로더
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 모델 초기화
    if model_type == "Seq2Seq":
        model = Seq2Seq(len(src_vocab.word2idx), len(tgt_vocab.word2idx), embed_size, hidden_size, num_layers, dropout).to(device)
    elif model_type == "Seq2SeqAttention":
        model = Seq2SeqAttention(len(src_vocab.word2idx), len(tgt_vocab.word2idx), embed_size, hidden_size, num_layers, dropout).to(device)
    else:  # Transformer
        model = Transformer(len(src_vocab.word2idx), len(tgt_vocab.word2idx), embed_size, num_layers, heads, dropout, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.word2idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 (tqdm은 train_model에서 처리)
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # 평가
    print("Evaluating model...")
    test_loss = evaluate_model(model, test_loader, criterion, device, tgt_vocab)
    print(f"Evaluation complete. Test Loss: {test_loss:.4f}")

    # 시각화
    plot_losses(train_losses, val_losses, model_type)

    return test_loss

# 메인 실행 코드
if __name__ == "__main__":
    # 예시 데이터 (실제로는 대규모 병렬 코퍼스 사용)
    src_sentences = ["안녕하세요", "저는 학생입니다", "오늘은 좋은 날입니다"]
    tgt_sentences = ["Hello", "I am a student", "Today is a good day"]

    # 어휘사전 구축
    src_vocab = Vocab(min_freq=1)
    tgt_vocab = Vocab(min_freq=1)
    src_vocab.build_vocab(src_sentences)
    tgt_vocab.build_vocab(tgt_sentences)

    # 데이터셋
    max_len = 20
    dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # 하이퍼파라미터 설정
    hyperparams_list = [
        {"embed_size": 256, "hidden_size": 512, "num_layers": 2, "dropout": 0.3, "num_epochs": 20, "batch_size": 64, "learning_rate": 0.001, "heads": 8},
        {"embed_size": 512, "hidden_size": 1024, "num_layers": 3, "dropout": 0.5, "num_epochs": 20, "batch_size": 32, "learning_rate": 0.0005, "heads": 8},
    ]

    # 모델별 실험
    for model_type in ["Seq2Seq", "Seq2SeqAttention", "Transformer"]:
        print(f"\nRunning experiments for {model_type}")
        for i, hyperparams in enumerate(hyperparams_list):
            print(f"Experiment {i+1} with hyperparams: {hyperparams}")
            test_loss = run_experiment(model_type, src_vocab, tgt_vocab, train_dataset, val_dataset, test_dataset, hyperparams)
            print(f"Test Loss: {test_loss:.4f}")