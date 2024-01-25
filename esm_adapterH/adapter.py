import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging
import os
import argparse


class ResMLP(torch.nn.Module):
  def __init__(self,
                bottleneck_size,
                module_type='MLP1',
                emb_dimension=512,
                nonlinearity='relu',  # activation function
                layer_norm=True,
                dropout=0.0,
                residual=True,
                ):
    """MLP class for soft prompt re-parameterization. MLP can have a Residual connection.
    Args:
        bottleneck_size (int): Dimension of the MLP bottlenack.
        module_type (str, optional): Type of MLP to be used.
            Currently supports 1-layer and 2-layer MLPs, and simple transformer layer ('MLP1'/'MLP2'/'transformer').
            Defaults to 'MLP1'.
        emb_dimension (int, optional): Dimension of T5 model embeddings. Defaults to 512 (T5-small embedding dimension).
        residual (bool, optional): Whether to use residual connection in MLP. Defaults to True.
    """
    super().__init__()
    assert module_type in ['MLP1', 'MLP2',
                            'transformer', 'LSTM', 'LSTM1', 'LSTM2']
    assert nonlinearity in ['relu', 'tanh', 'sigm']

    self.module_type = module_type

    if module_type not in ['LSTM', 'LSTM1', 'LSTM2', 'transformer']:
        layers = [nn.Linear(emb_dimension, bottleneck_size)]

        if nonlinearity == 'relu':
            layers.append(nn.ReLU())
        elif nonlinearity == 'tanh':
            layers.append(nn.Tanh())
        elif nonlinearity == 'sigm':
            layers.append(nn.Sigmoid())

        layers.append(nn.Linear(bottleneck_size, emb_dimension))

        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        if layer_norm:
            layers.append(nn.LayerNorm(emb_dimension))

        if module_type == 'MLP2':
            layers = layers + layers  # repeat twice
        
        self.module = torch.nn.Sequential(*layers)

    elif module_type in ['LSTM1', 'LSTM2', 'LSTM']:
        self.lstm_head = torch.nn.LSTM(
            input_size=emb_dimension, hidden_size=emb_dimension // 2,
            num_layers=1 if module_type == 'LSTM1' else 2, dropout=0.05,
            bidirectional=True, batch_first=True)
        self.mlp_head = nn.Sequential(
            nn.Linear(emb_dimension, emb_dimension),
            nn.ReLU(),
            nn.Linear(emb_dimension, emb_dimension))

    elif module_type == 'transformer':
        device = 'cuda'
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dimension, nhead=2, dropout=0.05).to(device)
        self.module = nn.TransformerEncoder(
            self.encoder_layer, num_layers=2).to(device)

    self.residual = residual
    #if self.residual:
    #    print('Using skip connection in MLP')

  def forward(self, inputs):
    if self.module_type == 'LSTM':
        output_embeds = self.mlp_head(self.lstm_head(inputs)[0]).squeeze()
    elif self.module_type in ['LSTM1', 'LSTM2']:
        output_embeds = self.lstm_head(inputs)[0].squeeze()
        if self.residual:
            output_embeds += inputs
        return output_embeds

    if self.residual:
        return self.module(inputs) + inputs
    else:
        return self.module(inputs)
