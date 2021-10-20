import pickle
import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score, precision_score, recall_score

from collections import defaultdict
from itertools import chain
from torch.optim.optimizer import Optimizer
import warnings
import math

class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gat = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gat)])
        self.rnn = nn.LSTM(dim,dim,1,bidirectional=True)
        self.W_attention_compounds = nn.Linear(dim, dim)
        self.W_attention_proteins = nn.Linear(2*dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2*dim, 2)
    def gat(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gat[i](xs))
            weights = F.linear(hs, hs)
            attn = weights.mul(A)
            attn_n = F.normalize(attn, p=1, dim=1)
            xs = xs + torch.matmul(attn_n, hs)
        return xs
    def attention_rnn(self, x, xs):
        xs = torch.unsqueeze(xs, 1)
        xs, _ = self.rnn(xs)
        xs = torch.relu(xs)
        xs = torch.squeeze(xs, 1)
        h = torch.relu(self.W_attention_compounds(x))
        hs = torch.relu(self.W_attention_proteins(xs))
        weights = F.linear(h,hs)
        weights_compounds = torch.mean(torch.t(weights),0)
        weights_compounds = torch.unsqueeze(weights_compounds,0)
#        w_compounds = torch.tanh(weights_compounds)
        weights_proteins = torch.mean(weights,0)
        weights_proteins = torch.unsqueeze(weights_proteins,0)
#        w_proteins = torch.tanh(weights_proteins)
        y = torch.t(weights_compounds) * h
        ys = torch.t(weights_proteins) * hs
        return torch.unsqueeze(torch.mean(ys, 0), 0), torch.unsqueeze(torch.mean(y, 0), 0), weights
    def forward(self, inputs):
        fingerprints, adjacency, words = inputs
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_temp_vector = self.gat(fingerprint_vectors, adjacency, layer_gat)
        word_vectors = self.embed_word(words)
        protein_vector, compound_vector, weights = self.attention_rnn(compound_temp_vector,
                                            word_vectors)
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)
        return interaction, weights
    def __call__(self, data, train=True):
        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction, weights = self.forward(inputs)
        if train:
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores, weights


class Trainer(object):
    def __init__(self, model):
        self.model = model
        weight_p, bias_p = [], []

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        self.optimizer_inner = RAdam(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': weight_decay}], lr=lr)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)
    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total

class Tester(object):
    def __init__(self, model):
        self.model = model
    def test(self, dataset):
        N = len(dataset)
        T, Y, S = [], [], []
        for data in dataset:
            (correct_labels, predicted_labels,
             predicted_scores, _) = self.model(data, train=False)
            T.append(correct_labels)
            Y.append(predicted_labels)
            S.append(predicted_scores)
        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return AUC, precision, recall
    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')
    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

