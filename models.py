import torch
import torch.nn as nn
import random
import numpy as np

# Seq2Seq 모델
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_size: int, hidden_size: int, num_layers: int, dropout: float):
        super(Seq2Seq, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        batch_size = src.size(0)
        max_len = tgt.size(1)
        tgt_vocab_size = self.fc.out_features

        outputs = torch.zeros(batch_size, max_len, tgt_vocab_size).to(src.device)
        enc_embedded = self.dropout(self.encoder_embedding(src))
        _, (hidden, cell) = self.encoder(enc_embedded)

        dec_input = tgt[:, 0].unsqueeze(1)
        for t in range(1, max_len):
            dec_embedded = self.dropout(self.decoder_embedding(dec_input))
            output, (hidden, cell) = self.decoder(dec_embedded, (hidden, cell))
            output = self.fc(output.squeeze(1))
            outputs[:, t, :] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(-1)
            dec_input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs

# Seq2Seq with Attention 모델
class Seq2SeqAttention(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_size: int, hidden_size: int, num_layers: int, dropout: float):
        super(Seq2SeqAttention, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        batch_size = src.size(0)
        max_len = tgt.size(1)
        tgt_vocab_size = self.fc.out_features

        outputs = torch.zeros(batch_size, max_len, tgt_vocab_size).to(src.device)
        enc_embedded = self.dropout(self.encoder_embedding(src))
        enc_output, (hidden, cell) = self.encoder(enc_embedded)

        dec_input = tgt[:, 0].unsqueeze(1)
        for t in range(1, max_len):
            dec_embedded = self.dropout(self.decoder_embedding(dec_input))
            attn_weights = torch.softmax(self.attention(torch.cat((dec_embedded.squeeze(1), hidden[-1]), dim=1)), dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_output)
            dec_input_combined = torch.cat((dec_embedded, context), dim=2)
            output, (hidden, cell) = self.decoder(dec_input_combined, (hidden, cell))
            output = self.fc(output.squeeze(1))
            outputs[:, t, :] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(-1)
            dec_input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs

# Transformer 모델
class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_size: int, num_layers: int, heads: int, dropout: float, device: torch.device):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.positional_encoding = self.create_positional_encoding(1000, embed_size).to(device)
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=embed_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def create_positional_encoding(self, max_len: int, embed_size: int) -> torch.Tensor:
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-np.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        src_mask = self.transformer.generate_square_subsequent_mask(src.size(1)).to(self.device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.device)

        src_embedded = self.dropout(self.encoder_embedding(src) + self.positional_encoding[:src.size(1), :])
        tgt_embedded = self.dropout(self.decoder_embedding(tgt) + self.positional_encoding[:tgt.size(1), :])

        output = self.transformer(src_embedded, tgt_embedded, src_mask=src_mask, tgt_mask=tgt_mask)
        output = self.fc(output)
        return output