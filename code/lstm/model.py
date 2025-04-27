import torch
import torch.nn as nn
from torch.optim import AdamW


class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.model.vocab_size,
            config.model.hidden_size,
            padding_idx=config.model.pad_token_id,
        )
        self.lstm = nn.LSTM(
            config.model.hidden_size,
            config.model.lstm_hidden_size,
            num_layers=config.model.lstm_num_layers,
            batch_first=True,
            dropout=(
                config.model.lstm_dropout if config.model.lstm_num_layers > 1 else 0
            ),
            bidirectional=config.model.lstm_bidirectional,
        )
        lstm_output_dim = config.model.lstm_hidden_size * (
            2 if config.model.lstm_bidirectional else 1
        )
        self.classifier = nn.Linear(lstm_output_dim, config.model.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        embedded = self.embedding(input_ids)
        _, (h_n, _) = self.lstm(embedded)

        if self.config.model.lstm_bidirectional:
            hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            hidden = h_n[-1, :, :]

        logits = self.classifier(hidden)

        loss = None
        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, self.config.model.num_labels), labels.view(-1)
            )
        return (logits, loss)

    def configure_optim(self, device):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )
        return optimizer
