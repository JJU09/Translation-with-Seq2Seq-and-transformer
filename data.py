import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import List, Tuple, Dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set_path = 'data/일상생활및구어체_한영_train_set.json'
valid_set_path = 'data/일상생활및구어체_한영_valid_set.json'

def load_json(file_path, max_samples=1000):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"][:max_samples]

train_set = load_json(train_set_path, max_samples=50000)
valid_set = load_json(valid_set_path, max_samples=1000)

ko_sentences_train = [item["ko"] for item in train_set]
en_sentences_train = [item["en"] for item in train_set]
ko_sentences_valid = [item["ko"] for item in valid_set]
en_sentences_valid = [item["en"] for item in valid_set]

# 데이터 전처리 및 어휘사전 구축
class Vocab:
    def __init__(self, min_freq: int = 2):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.min_freq = min_freq

    def build_vocab(self, sentences: List[str]) -> None:
        counter = Counter()
        for sentence in sentences:
            words = sentence.split()
            counter.update(words)
        
        # 기존 특수 토큰 유지하고 새로운 단어 추가
        vocab_size = len(self.word2idx)  # <PAD>, <SOS>, <EOS>, <UNK> 이후부터 시작
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = vocab_size
                self.idx2word[vocab_size] = word
                vocab_size += 1

    def sentence_to_tensor(self, sentence: str, max_len: int) -> torch.Tensor:
        words = sentence.split()
        indices = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words]
        indices = [self.word2idx["<SOS>"]] + indices + [self.word2idx["<EOS>"]]
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices += [self.word2idx["<PAD>"]] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)


# 데이터셋 및 데이터 로더
class TranslationDataset(Dataset):
    def __init__(self, src_sentences: List[str], tgt_sentences: List[str], src_vocab: Vocab, tgt_vocab: Vocab, max_len: int):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.src_sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self.src_vocab.sentence_to_tensor(self.src_sentences[idx], self.max_len)
        tgt = self.tgt_vocab.sentence_to_tensor(self.tgt_sentences[idx], self.max_len)
        return src, tgt