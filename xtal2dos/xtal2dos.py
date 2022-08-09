import torch, numpy as np
import torch.optim as optim
from   torch.optim import lr_scheduler 
from   torch.nn import Linear, Dropout, Parameter
import torch.nn.functional as F 
import torch.nn as nn

from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.utils    import softmax, to_dense_batch
from torch_geometric.nn       import global_add_pool, global_mean_pool
from torch_geometric.nn       import GATConv, GATv2Conv, TransformerConv, ResGatedGraphConv
from torch_geometric.nn       import GATConv
from torch_scatter            import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch.autograd import Variable

from random import sample
from copy import copy, deepcopy
from xtal2dos.utils import *
from xtal2dos.SinkhornDistance import SinkhornDistance
from xtal2dos.pytorch_stats_loss import torch_wasserstein_loss
from xtal2dos.graph2seq import LuongAttnDecoderRNN

import copy
from xtal2dos.transformer import Decoder, Embeddings, MultiHeadedAttention, PositionwiseFeedForward, Generator, DecoderLayer, rate

import e3nn
from xtal2dos.e3nn_utils import Network

#device = set_device()
torch.cuda.empty_cache()
kl_loss_fn = torch.nn.KLDivLoss()
sinkhorn = SinkhornDistance(eps=0.1, max_iter=50, reduction='mean')
huber_fn = nn.SmoothL1Loss()


class COMPOSITION_Attention(torch.nn.Module):
    def __init__(self,neurons):
        super(COMPOSITION_Attention, self).__init__()
        self.node_layer1    = Linear(neurons+103,32)
        self.atten_layer    = Linear(32,1)

    def forward(self,x,batch,global_feat):
        #torch.set_printoptions(threshold=10_000)
        # global_feat, [bs*103], rach row is an atom composition vector
        # x: [num_atom * atom_emb_len]

        #print(x.shape) # [3667, 128]
        #print(batch.shape) # [3667]
        #print(global_feat.shape) # [256, 103]

        counts      = torch.unique(batch,return_counts=True)[-1]   # return the number of atoms per crystal
        # batch includes all of the atoms from the Batch of crystals, each atom indexed by its Batch index.

        graph_embed = global_feat
        graph_embed = torch.repeat_interleave(graph_embed, counts, dim=0)  # repeat rows according to counts
        chunk       = torch.cat([x,graph_embed],dim=-1) # [3667, 231]
        
        x           = F.softplus(self.node_layer1(chunk))  # [num_atom * 32] -> [3667, 32]
        x           = self.atten_layer(x) # [num_atom * 1] -> [3667, 1]
        weights     = softmax(x,batch) # [num_atom * 1] -> [3667, 1]
        
        return weights


class GAT_Crystal(MessagePassing):
    def __init__(self, in_features, out_features, edge_dim, heads, concat=False,
                 dropout=0.0, bias=True, has_edge_attr=True, **kwargs):
        super(GAT_Crystal, self).__init__(aggr='add',flow='target_to_source', node_dim=0, **kwargs)
        self.in_features       = in_features
        self.out_features      = out_features
        self.heads             = heads
        self.concat            = concat
        #self.dropout          = dropout
        self.dropout           = nn.Dropout(p=dropout)
        self.neg_slope         = 0.2
        self.prelu             = nn.PReLU()
        self.bn1               = nn.BatchNorm1d(heads)
        if has_edge_attr:
            self.W             = Parameter(torch.Tensor(in_features+edge_dim,heads*out_features))
        else:
            self.W = Parameter(torch.Tensor(in_features, heads * out_features))
        self.att               = Parameter(torch.Tensor(1,heads,2*out_features))

        if bias and concat       : self.bias = Parameter(torch.Tensor(heads * out_features))
        elif bias and not concat : self.bias = Parameter(torch.Tensor(out_features))
        else                     : self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        # x: [num_node, emb_len]
        # edge_index: [2, num_edge]
        # edge_attr: [num_edge, emb_len]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        # edge_index_i: [num_edge]
        # x_i: [num_edge, emb_len]
        # x_j: [num_edge, emb_len]
        # size_i: num_node
        # edge_attr: [num_edge, emb_len]
        if edge_attr is not None:
            x_i   = torch.cat([x_i,edge_attr],dim=-1)
            x_j   = torch.cat([x_j,edge_attr],dim=-1)

        x_i   = F.softplus(torch.matmul(x_i,self.W))
        x_j   = F.softplus(torch.matmul(x_j,self.W))

        x_i   = x_i.view(-1, self.heads, self.out_features) # [num_edge, num_head, emb_len]
        x_j   = x_j.view(-1, self.heads, self.out_features) # [num_edge, num_head, emb_len]

        alpha = F.softplus((torch.cat([x_i, x_j], dim=-1)*self.att).sum(dim=-1))  # [num_edge, num_head]

        # self.att: (1,heads,2*out_features)

        alpha = F.softplus(self.bn1(alpha))
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i) # [num_edge, num_head]
        #alpha = softmax(alpha, edge_index_i) # [num_edge, num_head]
        alpha = self.dropout(alpha)

        return x_j * alpha.view(-1, self.heads, 1) # [num_edge, num_head, emb_len]

    def update(self, aggr_out):
        # aggr_out: [num_node, num_head, emb_len]
        if self.concat is True:    aggr_out = aggr_out.view(-1, self.heads * self.out_features)
        else:                      aggr_out = aggr_out.mean(dim=1)
        if self.bias is not None:  aggr_out = aggr_out + self.bias
        return aggr_out # [num_node, emb_len]

class FractionalEncoder(nn.Module):
    """
    Encoding element fractional amount using a "fractional encoding" inspired
    by the positional encoder discussed by Vaswani.
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self,
                 d_model,
                 resolution=100,
                 log10=False,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model//2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        x = torch.linspace(0, self.resolution - 1,
                           self.resolution,
                           requires_grad=False) \
            .view(self.resolution, 1)   # (resolution, 1)
        fraction = torch.linspace(0, self.d_model - 1,
                                  self.d_model,
                                  requires_grad=False) \
            .view(1, self.d_model).repeat(self.resolution, 1) # (resolution, d_model)

        pe = torch.zeros(self.resolution, self.d_model) # (resolution, d_model)
        pe[:, 0::2] = torch.sin(x /torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        pe = self.register_buffer('pe', pe) # (resolution, d_model)

    def forward(self, x):
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x))**2
            x[x > 1] = 1
            # x = 1 - x  # for sinusoidal encoding at x=0
        x[x < 1/self.resolution] = 1/self.resolution
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1 # (bs, n_elem)
        out = self.pe[frac_idx] # (bs, n_elem, d_model)
        return out

class GNN(torch.nn.Module):
    def __init__(self,heads,neurons=64,nl=3,concat_comp=False):
        super(GNN, self).__init__()

        self.n_heads        = heads
        self.number_layers  = nl
        self.concat_comp    = concat_comp

        n_h, n_hX2          = neurons, neurons*2
        self.neurons        = neurons
        self.neg_slope      = 0.2  

        self.embed_n        = Linear(92,n_h)
        self.embed_e        = Linear(41,n_h)
        self.embed_comp     = Linear(103,n_h)
 
        self.node_att       = nn.ModuleList([GAT_Crystal(n_h,n_h,n_h,self.n_heads) for i in range(nl)])
        self.batch_norm     = nn.ModuleList([nn.BatchNorm1d(n_h) for i in range(nl)])

        self.comp_atten     = COMPOSITION_Attention(n_h)

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.]))
        self.pe = FractionalEncoder(n_h, resolution=5000, log10=False)
        self.ple = FractionalEncoder(n_h, resolution=5000, log10=True)
        self.pe_linear = nn.Linear(103, 1)
        self.ple_linear = nn.Linear(103, 1)

        if self.concat_comp : reg_h   = n_hX2
        else                : reg_h   = n_h

        self.linear1    = nn.Linear(reg_h,reg_h)
        self.linear2    = nn.Linear(reg_h,reg_h)

    def forward(self,data):
        x, edge_index, edge_attr   = data.x, data.edge_index, data.edge_attr
        #print("x:", x.shape) # [56, 92]
        #print("edge_idx:", edge_index.shape) # [2, 672]
        #print("edge_att:", edge_attr.shape) # [672, 41]

        batch, global_feat, cluster = data.batch, data.global_feature, data.cluster
        #print("batch:", batch.shape)
        #print("glb_fea:", global_feat.shape)
        #print("cluster:", cluster.shape)

        x           = self.embed_n(x) # [num_atom, emb_len] -> [56, 128]

        edge_attr   = F.leaky_relu(self.embed_e(edge_attr),self.neg_slope) # [num_edges, emb_len] -> [672, 128]

        for a_idx in range(len(self.node_att)):
            x     = self.node_att[a_idx](x,edge_index,edge_attr) # [num_atom, emb_len] -> [56, 128]
            x     = self.batch_norm[a_idx](x)
            x     = F.softplus(x)

        ag        = self.comp_atten(x,batch,global_feat) # [num_atom * 1]
        x         = (x)*ag  # [num_atom, emb_len] -> [57, 128]
        
        # CRYSTAL FEATURE-AGGREGATION 
        y         = global_mean_pool(x,batch)#*2**self.emb_scaler#.unsqueeze(1).squeeze() # [bs, emb_len]
        #y         = F.relu(self.linear1(y))  # [bs, emb_len]
        #y         = F.relu(self.linear2(y))  # [bs, emb_len]

        if self.concat_comp: # False
            pe = torch.zeros([global_feat.shape[0], global_feat.shape[1], y.shape[1]]).cuda()
            ple = torch.zeros([global_feat.shape[0], global_feat.shape[1], y.shape[1]]).cuda()
            pe_scaler = 2 ** (1 - self.pos_scaler) ** 2
            ple_scaler = 2 ** (1 - self.pos_scaler_log) ** 2
            pe[:, :, :y.shape[1] // 2] = self.pe(global_feat)# * pe_scaler
            ple[:, :, y.shape[1] // 2:] = self.ple(global_feat)# * ple_scaler
            pe = self.pe_linear(torch.transpose(pe, 1,2)).squeeze()* pe_scaler
            ple = self.ple_linear(torch.transpose(ple, 1,2)).squeeze()* ple_scaler
            y = y + pe + ple
            #y = torch.cat([y, pe+ple], dim=-1)
            #y     = torch.cat([y, F.leaky_relu(self.embed_comp(global_feat), self.neg_slope)], dim=-1)

        return y, x, batch

class E3NN(Network):
    def __init__(self, neurons, **kwargs):
        super().__init__(**kwargs)

        n_h = neurons
        self.embed_n        = Linear(92,n_h)
        self.embed_e        = Linear(41,n_h)
        self.embed_comp     = Linear(103,n_h)

        self.neg_slope      = 0.2

    def forward(self, data):
        x, edge_index, edge_attr   = data.x, data.edge_index, data.edge_attr

        batch, global_feat, cluster = data.batch, data.global_feature, data.cluster

        #print(x.shape)
        data.x = F.leaky_relu(self.embed_n(x)) # [num_atom, emb_len]
        #print(x.shape)
        data.edge_attr = F.leaky_relu(self.embed_e(edge_attr), self.neg_slope) # [num_edges, emb_len]
        global_feat = F.leaky_relu(self.embed_comp(global_feat), self.neg_slope)

        x = super().forward(data)
        
        # y = scatter(x, batch, dim=0).div(self.num_nodes**0.5)
        y = global_mean_pool(x, batch)
        z = torch.cat([y, global_feat], dim=-1)
        return z, x, batch

class GAT(nn.Module):
    def __init__(self,heads=4,neurons=64,nl=3,concat_comp=False,dropout=0.5):
        super(GAT, self).__init__()

        n_h, n_hX2          = neurons, neurons*2
        self.neurons        = neurons
        self.neg_slope      = 0.2

        self.embed_n        = Linear(92,n_h)
        self.embed_e        = Linear(41,n_h)
        self.embed_comp     = Linear(103,n_h)

        #self.conv1 = GATv2Conv(n_h, 16, heads=4, concat=True, edge_dim=n_h)
        #self.conv2 = GATv2Conv(64, 16, 4, concat=True, edge_dim=n_h)
        #self.conv3 = GATv2Conv(64, 16, 4, concat=True, edge_dim=n_h)
        #self.conv4 = GATv2Conv(64, n_h, 4, concat=False, edge_dim=n_h)

        self.conv1 = TransformerConv(n_h, 16, heads=4, concat=True, edge_dim=n_h)
        self.conv2 = TransformerConv(64, 16, 4, concat=True, edge_dim=n_h)
        self.conv3 = TransformerConv(64, 16, 4, concat=True, edge_dim=n_h)
        self.conv4 = TransformerConv(64, n_h, 4, concat=False, edge_dim=n_h)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        #self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(n_h)

        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self,data):
        x, edge_index, edge_attr   = data.x, data.edge_index, data.edge_attr

        batch, global_feat, cluster = data.batch, data.global_feature, data.cluster
        # batch indicates each atom/node in x belongs to which batch

        #print(x.shape)
        x = F.leaky_relu(self.embed_n(x)) # [num_atom, emb_len]
        #print(x.shape)
        edge_attr = F.leaky_relu(self.embed_e(edge_attr), self.neg_slope) # [num_edges, emb_len]
        global_feat = F.leaky_relu(self.embed_comp(global_feat), self.neg_slope)
        #edge_attr = self.embed_e(edge_attr)
        #global_feat = self.embed_comp(global_feat)

        # run GNN to update node embedding
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        #x = self.conv3(x, edge_index, edge_attr)
        #x = self.bn3(x)
        #x = F.leaky_relu(x)
        x = self.conv4(x, edge_index, edge_attr)
        x = self.bn4(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        y = global_mean_pool(x, batch)
        z = torch.cat([y, global_feat], dim=-1)
        return z, x, batch


class Xtal2DoS(nn.Module):
    def __init__(self, args):
        super(Xtal2DoS, self).__init__()
        n_heads = args.num_heads
        number_neurons = args.d_model
        number_layers = args.num_layers
        concat_comp = args.concat_comp
        #self.graph_encoder = GNN(n_heads, neurons=number_neurons, nl=number_layers, concat_comp=concat_comp)
        if args.model == 'gat':
            self.graph_encoder = GAT(n_heads, neurons=number_neurons, nl=number_layers, concat_comp=concat_comp, dropout=args.graph_dropout)
        elif args.model == 'e3nn':
            self.graph_encoder = E3NN(
                                    neurons=number_neurons,
                                    irreps_in=None, # TODO: atomic mass     
                                    irreps_out=str(number_neurons)+"x0e",   
                                    irreps_node_attr=str(number_neurons)+"x0e",    
                                    layers=number_layers,
                                    mul=32,
                                    lmax=1, 
                                    max_radius=4.,
                                    num_neighbors=12)

        self.loss_type = args.xtal2dos_loss_type
        #self.input_dim = args.xtal2dos_input_dim # 128
        #self.latent_dim = args.xtal2dos_latent_dim # 128
        #self.emb_size = args.xtal2dos_emb_size # 512
        self.label_dim = args.xtal2dos_label_dim # 51
        #self.chunk_dim = args.chunk_dim # 3
        #self.seq_len = self.label_dim // self.chunk_dim

        self.scale_coeff = args.xtal2dos_scale_coeff
        self.K = args.xtal2dos_K
        self.dec_layers = args.dec_layers
        self.dec_dropout = args.dec_dropout
        self.dec_in_dim = args.dec_in_dim
        self.min_gap = 1./self.dec_in_dim
        self.bs = args.batch_size
        self.use_bin = args.use_bin
        self.d_model = args.d_model
        self.h = args.h
        self.d_ff = args.d_ff
        self.args = args

        #self.decoder = LuongAttnDecoderRNN('dot', self.chunk_dim+self.latent_dim, self.latent_dim, self.chunk_dim, self.dec_layers, dropout = self.dec_dropout)
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.h, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dec_dropout)
        self.tgt_embed = Embeddings(self.d_model, self.label_dim)
        self.fc = Linear(number_neurons*2+self.d_model, self.d_model)
        self.decoder = Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dec_dropout), self.dec_layers)
        self.generator = Linear(self.d_model, 1)
        #self.generator = Generator(self.d_model, 1)

        #self.fx = Linear(self.d_model, self.label_dim)
 

    def forward(self, data):
        label = data.y # [256, 128]
        bs = label.shape[0]
        seq_len = self.label_dim
        crystal_ctx, nodes, batch = self.graph_encoder(data)
        #print(crystal_ctx.shape) # [128, 1024]
        #print(nodes.shape) # [1940, 512]
        #print(len(batch)) # [1940]
        
        nodes, nd_mask = to_dense_batch(nodes, batch)
        #print(nodes.shape) # [128, 72, 512]
        #out = torch.mean(self.fx(F.relu(nodes)), dim=1)
        #return out
        
        pos = Variable(torch.arange(self.label_dim).repeat(bs, 1)).cuda() # [256, 128]
        tgt_embs = self.tgt_embed(pos)
        #print(pos.shape) # [128, 128]
        #print(tgt_embs.shape) # [128, 128, 512]
        #print(nodes.shape) # [128, 72, 512]
        dec_input = torch.cat((tgt_embs, crystal_ctx.unsqueeze(1).repeat(1, seq_len, 1)), dim=-1)
        #print(dec_input.shape) # [128, 128, 1536]
        dec_input = F.leaky_relu(self.fc(dec_input))
        #print(dec_input.shape) # [128, 128, 512]
        
        #print(nd_mask.unsqueeze(1).shape) # [256, 72]
        out = self.decoder(dec_input, nodes, nd_mask.unsqueeze(1), None) # [256, 128, 512]
        out = self.generator(out).squeeze(-1) # [256, 128]

        return out


def compute_loss(input_label, input_label_sum, fx_out, loss_fn, args, verbose=False):

    if args.label_scaling == 'normalized_sum':

        loss_type = args.xtal2dos_loss_type
        pred = F.softmax(fx_out)

        if loss_type == 'KL':
            log_target = torch.log(input_label+1e-10)
            log_pred = F.log_softmax(fx_out)
            label_weights = F.softmax(log_target*args.temp)
            #kl_loss = loss_fn(log_pred, log_target)
            if args.sum_weighted:
                loss = torch.sum(label_weights*(log_target-log_pred),dim=1)
                loss = loss * input_label_sum / args.sum_scale
            else:
                loss = torch.sum(label_weights*(log_target-log_pred),dim=1)

        elif loss_type == 'MAE':
            target = input_label
            err = target - pred
            if args.sum_weighted:
                loss = torch.sum(torch.abs(err), dim=1)
                loss = loss * input_label_sum / args.sum_scale
            else:
                loss = torch.mean(torch.sum(torch.abs(err), dim=1))
        elif loss_type == "MSE":
            target = input_label * input_label_sum.unsqueeze(-1)
            pred = pred * input_label_sum.unsqueeze(-1)
            err = target - pred
            loss = torch.mean(torch.square(err), dim=1)

        loss_lst = loss
        total_loss = torch.mean(loss_lst)

        if verbose:
            return total_loss, pred, loss_lst
        else:
            return total_loss, pred
