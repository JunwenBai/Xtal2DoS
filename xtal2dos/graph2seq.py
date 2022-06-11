import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy

import torch_scatter
from torch_scatter.composite import scatter_softmax
from torch_scatter import scatter_add, scatter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    #seq_range = torch.range(0, max_len - 1).long()
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()                                                                                                                                
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    length = Variable(torch.LongTensor(length)).cuda()

    """ 
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        #print("hid:", hidden.shape) # [1, 50, 500]
        #print("enc_out:", encoder_outputs.shape) # [8, 50, 500]
        max_len = encoder_outputs.size(0) # [8]
        this_batch_size = encoder_outputs.size(1) # [50]

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S -> [50, 8]

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                #print(hidden[:, b].shape, encoder_outputs[i, b].shape)
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        #print(attn_energies.shape) # [50, 8]

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=-1).unsqueeze(1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy =torch.dot(hidden.view(-1), encoder_output.view(-1))
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.dot(hidden.view(-1), energy.view(-1))
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = torch.dot(self.v.view(-1), energy.view(-1))
        return energy

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, input_size, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        #self.embedding = nn.Embedding(output_size, hidden_size)
        #self.embedding_dropout = nn.Dropout(dropout)
        self.enc = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.proj = nn.Linear(hidden_size, hidden_size)
        #self.bn = nn.BatchNorm1d(hidden_size)
        self.compress = nn.Linear(hidden_size * 3, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        #if attn_model != 'none':
        #    self.attn = Attn(attn_model, hidden_size)i

    def attn(self, rnn_output, nodes, batch):
        # rnn_output: [1, 96, 128]
        # nodes: [224, 128]
        # batch: len 224
        #?#attn_mtx = rnn_output.transpose(0, 1) * nodes.unsqueeze(0).contiguous() # [8, 48, 128]
        attn_mtx = rnn_output.transpose(0, 1) * nodes.view(1, nodes.shape[0], nodes.shape[1]) # [32, 224, 128]
        attn_mtx = attn_mtx.sum(-1) # [32, 224]
        bs = attn_mtx.shape[0] # 32
        attn_weights = scatter_softmax(attn_mtx, batch, dim=-1) # [32, 224]

        #?#context_mtx = attn_weights.unsqueeze(-1) * nodes.unsqueeze(0) # [8, 48, 128]
        context_mtx = attn_weights.view(attn_weights.shape[0], attn_weights.shape[1], 1) * nodes.unsqueeze(0) # [32, 224, 128]
        context = scatter_add(context_mtx, batch, dim=1) # [32, 32, 128]
        #torch.manual_seed(time.time())
        #context = torch.randn(8, 8, 128).to(device)
        #context = context.view(self.output_size, bs//self.output_size, bs//self.output_size, self.hidden_size) # [3, 32, 32, 128]
        #print(context.diagonal(dim1=1, dim2=2).shape) # [3, 128, 32]
        context = context.diagonal(dim1=0, dim2=1).transpose(0, 1).contiguous() # [32, 128]
        #context = context.view(-1, self.hidden_size)
        #indices = torch.arange(bs).to(device).view(-1, 1, 1).repeat(1, 1, 128)
        #context2 = context.gather(dim=1, index=indices).squeeze(1)
        #print(context1.shape, context2.shape)
        #print(torch.sum(context1))
        #print(torch.sum(context1-context2))
        return context, attn_weights

    def forward(self, input_seq, last_hidden, nodes, batch):
        # Note: we run this one step at a time
        #print("input_seq:", input_seq.shape) # [1, 8, 256*3]
        #print("last_hid:", last_hidden.shape) # [2, 8,128]
        #print("nodes:", nodes.shape) # [48, 128]
        #print("batch", len(batch)) # 48

        # Get the embedding of the current input word (last output word)
        embedded = F.relu(self.enc(input_seq)) # [1, 32, 128]

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)
        #print("rnn_out:", rnn_output.shape) # [1, 32, 128]
        #print("hid:", hidden.shape) # [2, 32, 128]
        rnn_output = self.proj(F.relu(rnn_output))
        #print("rnn_output:", rnn_output.shape) # [1, 32, 128]
        bs = rnn_output.shape[1]
        rnn_output = rnn_output.view(-1, bs, self.hidden_size) # [1, 32, 128]

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        # rnn_output: [1, 8, 128]
        # nodes: [48, 128]
        context, attn_weights = self.attn(rnn_output, nodes, batch) # [32, 128]
        #print("context:", context.shape) # [32, 128]
        #print("attn:", attn_weights.shape) # [32, 224]
        
        #print(encoder_outputs.transpose(0, 1).shape) # 
        #context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N -> [50, 1, 500]

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N -> [32, 128]
        embedded = embedded.squeeze(0)
        concat_input = torch.cat((rnn_output, context, embedded), 1) # [32, 128*3]
        concat_output = F.relu(self.compress(concat_input)) # [32, 128]
        output = self.out(concat_output) # [32, 3]

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

