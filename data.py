import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import List, Tuple, Dict


# 데이터 로딩
def load_json(file_path: str, max_samples: int = 1000) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"][:max_samples]

# 데이터 전처리
def preprocess_data(ko_sentences: List[str], en_sentences: List[str]) -> Tuple[List[str], List[str]]:
    okt = Okt()
    ko_tokenized = [" ".join(okt.morphs(s)) for s in tqdm(ko_sentences, desc="Tokenizing Korean")]
    en_tokenized = [" ".join(word_tokenize(s.lower())) for s in tqdm(en_sentences, desc="Tokenizing English")]
    return ko_tokenized, en_tokenized

# 어휘 사전
class Vocab:
    def __init__(self, min_freq: int = 5):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.min_freq = min_freq

    def build_vocab(self, sentences: List[str]) -> None:
        counter = Counter()
        for sentence in tqdm(sentences, desc="Building vocab"):
            words = sentence.split()
            counter.update(words)
        
        vocab_size = len(self.word2idx)
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

    def tensor_to_sentence(self, tensor: torch.Tensor) -> str:
        indices = tensor.tolist()
        words = [self.idx2word.get(idx, "<UNK>") for idx in indices if idx not in [0, 1, 2]]
        return " ".join(words)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'Vocab':
        with open(path, 'rb') as f:
            return pickle.load(f)

# 데이터셋
class TranslationDataset(Dataset):
    def __init__(self, src_sentences: List[str], tgt_sentences: List[str], src_vocab: Vocab, tgt_vocab: Vocab, max_len: int):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.src_tensors = [self.src_vocab.sentence_to_tensor(s, max_len) for s in tqdm(src_sentences, desc="Preprocessing src")]
        self.tgt_tensors = [self.tgt_vocab.sentence_to_tensor(s, max_len) for s in tqdm(tgt_sentences, desc="Preprocessing tgt")]

    def __len__(self) -> int:
        return len(self.src_sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.src_tensors[idx], self.tgt_tensors[idx]
