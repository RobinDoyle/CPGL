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

class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class PlainRAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    step_size = group['lr'] * math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup=warmup)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss

class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        #self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn1 = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_gnn2 = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_gnn3 = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])  
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=3,
                     stride=1, padding=1) for _ in range(layer_gnn)])
        self.w = nn.Linear(101, dim)
        self.rnn = nn.LSTM(dim,dim,1,bidirectional=True)
        self.W_attention_compounds = nn.Linear(dim, dim)
        self.W_attention_proteins = nn.Linear(2*dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2*dim, 2)
    def gat(self, xs, A, layer):
        for i in range(layer):
            hs1 = torch.relu(self.W_gnn1[i](xs))
            hs2 = torch.relu(self.W_gnn2[i](xs))
            weights = F.linear(hs1, hs2)
            attn = weights.mul(A)
            hs3 = torch.relu(self.W_gnn3[i](xs))
            xs = xs + torch.matmul(attn, hs3)
        return xs
    def attention_rnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of RNN."""
        xs = self.w(xs)
        xs = torch.unsqueeze(xs, 1)
        xs, _ = self.rnn(xs)
        xs = torch.relu(xs)
        xs = torch.squeeze(xs, 1)
        h = torch.relu(self.W_attention_compounds(x))
        hs = torch.relu(self.W_attention_proteins(xs))
        weights = F.linear(h,hs)
        weights_compounds = torch.mean(torch.t(weights),0)
        weights_compounds = torch.unsqueeze(weights_compounds,0)
        w_compounds = torch.tanh(weights_compounds)
        weights_proteins = torch.mean(weights,0)
        weights_proteins = torch.unsqueeze(weights_proteins,0)
        w_proteins = torch.tanh(weights_proteins)
        y = torch.t(weights_compounds) * h
        ys = torch.t(weights_proteins) * hs
        return torch.unsqueeze(torch.mean(ys, 0), 0), torch.unsqueeze(torch.mean(y, 0), 0), weights
    def forward(self, inputs):
        fingerprints, adjacency, words = inputs
        """Compound vector with GAT."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_temp_vector = self.gat(fingerprint_vectors, adjacency, layer_gnn)
        """Protein vector with attention-RNN."""
        #word_vectors = self.embed_word(words)
        protein_vector, compound_vector, weights = self.attention_rnn(compound_temp_vector,
                                            words, layer_cnn)
        """Concatenate the above two vectors and output the interaction."""
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
        return AUC, precision, recall, T, Y, S
    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')
    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy',allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def val(data, n_val, i) :
    l=int(len(data)/n_val)
    if i==n_val-1:
        val=data[(l*i):]
        train=data[:(l*i)]
    elif i==0:
        val=data[:l]
        train=data[l:]
    else :
        val=data[(l*i):(l*(i+1))]
        train=data[:(l*i)]+data[l*(i+1):]
    return val, train


if __name__ == "__main__":
    
    """Hyperparameters."""
    DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_output, lr, lr_decay, decay_interval, weight_decay, iteration = 'celegans',2,3,128,3,11,3,3,1e-3,0.5,10,1e-6,31
    n_val=5
    setting = (DATASET + '_' + str(n_val) + '_' + str(dim) +  '_' + str(layer_gnn) + '_' + str(layer_output) + '_' + str(decay_interval) + '_' + str(weight_decay) )
    (dim, layer_gnn, window, layer_cnn, layer_output, decay_interval,
         iteration) = map(int, [dim, layer_gnn, window, layer_cnn, layer_output,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Output files."""
    file_AUCs = '../output/result/'+ DATASET +'/AUCs_' + setting + '.txt'

    AUCs = ('n_radius\tn_val\tEpoch\tTime(sec)\tLoss_train\tAUC_dev\t'
                'AUC_test\tPrecision_test\tRecall_test')
    with open(file_AUCs, 'w') as f:
            f.write(AUCs + '\n')
    """Start training."""
    print(AUCs)


    """Load preprocessed data."""
    dir_input = ('../input/' + DATASET + '/radius' + str(radius) + '_ngram' + str(ngram) + '/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.FloatTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    #word_dict = load_pickle(dir_input + 'word_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    #n_word = len(word_dict)
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_test, dataset_ = split_dataset(dataset, 0.1)
    T_v, Y_v, S_v=torch.tensor([]), torch.tensor([]), torch.tensor([])
    for i in range(n_val):
        file_model = '../output/model/'+ DATASET + '/'  + str(i) + '_' +str(radius) + '_' + setting
        dataset_dev, dataset_train = val(dataset_, n_val, i)
        torch.manual_seed(0)
        model = CompoundProteinInteractionPrediction().to(device)
        trainer = Trainer(model)
        tester = Tester(model)
        start = timeit.default_timer()
        max_auc=0
        temp=0
        trainer.optimizer.param_groups[0]['lr'] = lr
        for epoch in range(1, iteration):
            loss_train = trainer.train(dataset_train)
            AUC_dev = tester.test(dataset_dev)[0]
            AUC_test, precision_test, recall_test, _, _, _ = tester.test(dataset_test)
            if temp > AUC_dev:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay
            end = timeit.default_timer()
            time = end - start
            AUCs = [radius, i, epoch, time, loss_train, AUC_dev, AUC_test, precision_test, recall_test]
            tester.save_AUCs(AUCs, file_AUCs)
            if AUC_dev > max_auc:
                max_auc = AUC_dev
                tester.save_model(model, file_model)
            print('\t'.join(map(str, AUCs)))
            if np.abs(temp - AUC_dev) < 1e-4:
                break
            temp = AUC_dev
        best_model = CompoundProteinInteractionPrediction().to(device)
        best_model.load_state_dict(torch.load(file_model))
        tester = Tester(best_model)
        _, _, _, T, Y, S = tester.test(dataset_test)
        T = torch.tensor(T)
        T = torch.unsqueeze(T,0)
        Y = torch.tensor(Y)
        Y = torch.unsqueeze(Y,0)
        S = torch.tensor(S)
        S = torch.unsqueeze(S,0)
        T_v = torch.cat((T_v, T), dim=0)
        Y_v = torch.cat((Y_v, Y), dim=0)
        S_v = torch.cat((S_v, S), dim=0)
    T_v = torch.squeeze(T_v, 2)
    Y_v = torch.squeeze(Y_v, 2)
    S_v = torch.squeeze(S_v, 2)
    T = torch.mean(T_v,0)
    Y = torch.mean(S_v,0)
    Y = Y.round()
    S = torch.mean(S_v,0)
    AUC = roc_auc_score(T, S)
    precision = precision_score(T, Y)
    recall = recall_score(T, Y)
    AUCs=[AUC, precision, recall]
    with open(file_AUCs, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')
    print('\t'.join(map(str, AUCs)))
