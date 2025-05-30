import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import  Mamba2
from pytorchcrf import CRF


class ProteinModel(nn.Module):
    def __init__(self, d_model=1280, num_classes=5):
        super(ProteinModel, self).__init__()
        self.mamba_model = Mamba2Wrapper(d_model=d_model)
        self.attention = AttentionWrapper2(d_model=d_model, num_heads=8, batch_first=True)

        self.feedforward = FeedforwardWrapper(d_model=d_model)

        self.classification = ComplexMLP(d_model, num_classes, [256])

        self.crf_turn = ComplexMLP(d_model, 2, [256])
        self.crf = CRF(num_tags=2, batch_first=True)

        self._initialize_transition_matrix()

    def _initialize_transition_matrix(self):
        transition_matrix = self.crf.transitions.data.clone()
        transition_matrix[0, 1] = -1e4
        self.crf.transitions.data.copy_(transition_matrix)
    def forward(self, x, tags=None):
        mamba_output = self.mamba_model(x)
        attn_output = self.attention(mamba_output)
        feed_output = self.feedforward(attn_output)
        class_input = feed_output[:, 0, :]
        class_output = self.classification(class_input)

        crf_input = feed_output[:, 1:-1 :]
        crf_input = self.crf_turn(crf_input)
        if tags is not None:
            crf_output = -self.crf(crf_input, tags, reduction='mean')
        else:
            crf_output = self.crf.decode(crf_input)
        return class_output,crf_output

class Mamba2Wrapper(nn.Module):
    def __init__(self, d_model=256):
        super(Mamba2Wrapper, self,).__init__()
        self.mamba_model = Mamba2(d_model=d_model)
        self.dropout = nn.Dropout(p=0.05)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        out = self.mamba_model(x)
        return self.norm(x + self.dropout(out))

class AttentionWrapper2(nn.Module):
    def __init__(self, d_model=256, num_heads=8, dropout=0.05, batch_first=True):
        super(AttentionWrapper2, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=batch_first)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x,key_padding_mask=None):
        attn_output, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        return self.norm(x + self.dropout(attn_output))

class FeedforwardWrapper(nn.Module):
    def __init__(self, d_model=256, d_ff=512, dropout=0.05):
        super(FeedforwardWrapper, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.dropout(out)
        return self.norm(x + self.linear2(out))

class ComplexMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout=0.05, activation='relu'):
        super(ComplexMLP, self).__init__()

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(self._get_activation(activation))
        layers.append(nn.LayerNorm(hidden_dims[0]))
        layers.append(nn.Dropout(dropout))

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(self._get_activation(activation))
            layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.layers = nn.Sequential(*layers)

    def _get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x):
        return self.layers(x)

