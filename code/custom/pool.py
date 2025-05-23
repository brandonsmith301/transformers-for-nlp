import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


def get_last_hidden_state(backbone_outputs):
    last_hidden_state = backbone_outputs[0]
    return last_hidden_state


def get_all_hidden_states(backbone_outputs):
    all_hidden_states = torch.stack(backbone_outputs[1])
    return all_hidden_states


def get_input_ids(inputs):
    return inputs["input_ids"]


def get_attention_mask(inputs):
    return inputs["attention_mask"]


class MeanPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(MeanPooling, self).__init__()
        self.output_dim = backbone_config.model.n_embd

    def forward(self, inputs, backbone_outputs):
        attention_mask = get_attention_mask(inputs)
        last_hidden_state = get_last_hidden_state(backbone_outputs)

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class LSTMPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config, is_lstm=True):
        super(LSTMPooling, self).__init__()

        self.num_hidden_layers = backbone_config.model.n_layer
        self.hidden_size = backbone_config.model.n_embd
        self.hidden_lstm_size = pooling_config.hidden_size
        self.dropout_rate = pooling_config.dropout_rate
        self.bidirectional = pooling_config.bidirectional

        self.is_lstm = is_lstm
        self.output_dim = (
            pooling_config.hidden_size * 2
            if self.bidirectional
            else pooling_config.hidden_size
        )

        if self.is_lstm:
            self.lstm = nn.LSTM(
                self.hidden_size,
                self.hidden_lstm_size,
                bidirectional=self.bidirectional,
                batch_first=True,
            )
        else:
            self.lstm = nn.GRU(
                self.hidden_size,
                self.hidden_lstm_size,
                bidirectional=self.bidirectional,
                batch_first=True,
            )

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        hidden_states = torch.stack(
            [
                all_hidden_states[layer_i][:, 0].squeeze()
                for layer_i in range(1, self.num_hidden_layers + 1)
            ],
            dim=-1,
        )
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out


class WeightedLayerPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(WeightedLayerPooling, self).__init__()

        self.num_hidden_layers = backbone_config.model.n_layer
        self.layer_start = pooling_config.layer_start
        self.layer_weights = None

        self.output_dim = backbone_config.model.n_embd

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        all_layer_embedding = all_hidden_states[self.layer_start :]

        num_layers = all_layer_embedding.shape[0]
        if self.layer_weights is None or self.layer_weights.size(0) != num_layers:
            self.layer_weights = nn.Parameter(
                torch.ones(
                    num_layers,
                    dtype=all_layer_embedding.dtype,
                    device=all_layer_embedding.device,
                )
            )

        weight_factor = self.layer_weights.view(-1, 1, 1, 1)

        weighted_average = (weight_factor * all_layer_embedding).sum(
            dim=0
        ) / self.layer_weights.sum()

        return weighted_average[:, 0]


class ConcatPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(
            ConcatPooling,
            self,
        ).__init__()

        self.n_layers = pooling_config.n_layers
        self.output_dim = backbone_config.model.n_embd * pooling_config.n_layers

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        concatenate_pooling = torch.cat(
            [all_hidden_states[-(i + 1)] for i in range(self.n_layers)], -1
        )
        concatenate_pooling = concatenate_pooling[:, 0]
        return concatenate_pooling


class AttentionPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(AttentionPooling, self).__init__()
        self.num_hidden_layers = backbone_config.model.n_layer
        self.hidden_size = backbone_config.model.n_embd
        self.hiddendim_fc = pooling_config.hiddendim_fc
        self.dropout = nn.Dropout(pooling_config.dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float().to(self.device)
        w_ht = np.random.normal(
            loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc)
        )
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().to(self.device)

        self.output_dim = self.hiddendim_fc

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        hidden_states = torch.stack(
            [
                all_hidden_states[layer_i][:, 0].squeeze()
                for layer_i in range(1, self.num_hidden_layers + 1)
            ],
            dim=-1,
        )
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v


class MeanMaxPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(MeanMaxPooling, self).__init__()
        self.output_dim = backbone_config.model.n_embd * 2  

    def forward(self, inputs, backbone_outputs):
        attention_mask = get_attention_mask(inputs)
        last_hidden_state = get_last_hidden_state(backbone_outputs)

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # mean
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        # max
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)

        mean_max_embeddings = torch.cat((mean_embeddings, max_embeddings), 1)
        return mean_max_embeddings


class MaxPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(MaxPooling, self).__init__()
        self.output_dim = backbone_config.model.n_embd

    def forward(self, inputs, backbone_outputs):
        attention_mask = get_attention_mask(inputs)
        last_hidden_state = get_last_hidden_state(backbone_outputs)

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings


class MinPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(MinPooling, self).__init__()
        self.output_dim = backbone_config.model.n_embd

    def forward(self, inputs, backbone_outputs):
        attention_mask = get_attention_mask(inputs)
        last_hidden_state = get_last_hidden_state(backbone_outputs)

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e4
        min_embeddings, _ = torch.min(embeddings, dim=1)
        return min_embeddings


class GeMText(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(GeMText, self).__init__()
        self.dim = 1
        self.eps = pooling_config.eps
        self.p = Parameter(torch.ones(1) * pooling_config.p)
        self.output_dim = backbone_config.model.n_embd

    def forward(self, inputs, backbone_outputs):
        attention_mask = get_attention_mask(inputs)
        last_hidden_state = get_last_hidden_state(backbone_outputs)

        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.shape
        )
        x = (
            (last_hidden_state.clamp(min=self.eps) * attention_mask_expanded)
            .pow(self.p)
            .sum(self.dim)
        )
        ret = x / attention_mask_expanded.sum(self.dim).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret


def get_pooling_layer(config, backbone_config):
    if config.model.pooling_type == "MeanPooling":
        return MeanPooling(backbone_config, config.model.gru_pooling)

    elif config.model.pooling_type == "GRUPooling":
        return LSTMPooling(backbone_config, config.model.gru_pooling, is_lstm=False)

    elif config.model.pooling_type == "LSTMPooling":
        return LSTMPooling(backbone_config, config.model.lstm_pooling, is_lstm=True)

    elif config.model.pooling_type == "WeightedLayerPooling":
        return WeightedLayerPooling(backbone_config, config.model.weighted_pooling)

    elif config.model.pooling_type == "ConcatPooling":
        return ConcatPooling(backbone_config, config.model.concat_pooling)

    elif config.model.pooling_type == "AttentionPooling":
        return AttentionPooling(backbone_config, config.model.attention_pooling)

    elif config.model.pooling_type == "MeanMaxPooling":
        return MeanMaxPooling(backbone_config, config.model.mean_max_pooling)

    elif config.model.pooling_type == "MaxPooling":
        return MaxPooling(backbone_config, config.model.max_pooling)

    elif config.model.pooling_type == "MinPooling":
        return MinPooling(backbone_config, config.model.min_pooling)

    elif config.model.pooling_type == "GeMText":
        return GeMText(backbone_config, config.model.gem_text_pooling)

    else:
        raise ValueError(f"Invalid pooling type: {config.model.pooling_type}")
