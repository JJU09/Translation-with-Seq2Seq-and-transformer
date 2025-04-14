import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

from data import Vocab, TranslationDataset
from models import Seq2Seq, Seq2SeqAttention, Transformer
from train import train_model
from evaluate import evaluate_model, plot_losses


# 메인 실행
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 경로 (로컬 경로로 수정 필요)
    train_set_path = '/content/drive/MyDrive/코드잇/스프린트 미션/data/mission11/일상생활및구어체_한영_train_set.json'
    valid_set_path = '/content/drive/MyDrive/코드잇/스프린트 미션/data/mission11/일상생활및구어체_한영_valid_set.json'

    # 데이터 로드
    train_set = load_json(train_set_path, max_samples=1000)
    valid_set = load_json(valid_set_path, max_samples=1000)

    ko_sentences_train = [item["ko"] for item in train_set]
    en_sentences_train = [item["en"] for item in train_set]
    ko_sentences_valid = [item["ko"] for item in valid_set]
    en_sentences_valid = [item["en"] for item in valid_set]

    # 데이터 전처리
    src_sentences_train, tgt_sentences_train = preprocess_data(ko_sentences_train, en_sentences_train)
    src_sentences_valid, tgt_sentences_valid = preprocess_data(ko_sentences_valid, en_sentences_valid)

    # 어휘 사전 구축
    src_vocab = Vocab(min_freq=1)
    tgt_vocab = Vocab(min_freq=1)
    src_vocab.build_vocab(src_sentences_train)
    tgt_vocab.build_vocab(tgt_sentences_train)

    src_vocab.save("src_vocab.pkl")
    tgt_vocab.save("tgt_vocab.pkl")

    # 데이터셋 준비
    max_len = 50
    train_dataset = TranslationDataset(src_sentences_train, tgt_sentences_train, src_vocab, tgt_vocab, max_len)
    val_dataset = TranslationDataset(src_sentences_valid, tgt_sentences_valid, src_vocab, tgt_vocab, max_len)

    # 테스트 데이터는 훈련 데이터에서 분할
    train_size = int(0.9 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    # 하이퍼파라미터
    hyperparams_list = [
        {"embed_size": 128, "hidden_size": 256, "num_layers": 2, "dropout": 0.3, "num_epochs": 20, "batch_size": 32, "learning_rate": 0.001, "heads": 4},
        {"embed_size": 256, "hidden_size": 512, "num_layers": 3, "dropout": 0.5, "num_epochs": 20, "batch_size": 32, "learning_rate": 0.0005, "heads": 4}
    ]

    # 학습
    for model_type in ["Seq2Seq", "Seq2SeqAttention", "Transformer"]:
        print(f"\nRunning experiments for {model_type}")
        for i, hyperparams in enumerate(hyperparams_list):
            print(f"Experiment {i+1} with hyperparams: {hyperparams}")
            test_loss = run_experiment(model_type, src_vocab, tgt_vocab, train_dataset, val_dataset, test_dataset, hyperparams)
            print(f"Test Loss: {test_loss:.4f}")

    # 번역 테스트
    test_sentences = ["안녕하세요", "저는 학생입니다", "돈은 어디에 투자하셨나요?"]
    for model_type in ["Seq2Seq", "Seq2SeqAttention", "Transformer"]:
        for hyperparams in hyperparams_list:
            embed_size = hyperparams['embed_size']
            batch_size = hyperparams['batch_size']
            learning_rate = hyperparams['learning_rate']
            model_path = f"models/{model_type}_embed{embed_size}_batch{batch_size}_lr{learning_rate}.pt"
            if os.path.exists(model_path):
                print(f"\nTesting {model_type} model: embed={embed_size}, batch={batch_size}, lr={learning_rate}")
                for sentence in test_sentences:
                    translated = load_and_translate(model_type, model_path, "src_vocab.pkl", "tgt_vocab.pkl", sentence, max_len=50)
                    print(f"Input: {sentence}")
                    print(f"Translated: {translated}")
