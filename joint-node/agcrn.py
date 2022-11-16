import torch
import torch.nn as nn
import torch.nn.functional as F

class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    
    def forward(self, x, node_embeddings, em_t=None):
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        
        ###
        if em_t is not None:
            rand = torch.rand(supports.shape[0], supports.shape[1]).to(supports.device)
            supports = supports * (rand >= em_t)
        ###
        
        support_set = [torch.eye(node_num).to(supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings, em_t=None):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, em_t))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings, em_t))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings, em_t=None):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings, em_t)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)


class AGCRN(nn.Module):
    def __init__(self, num_nodes, embed_dim, in_dim, out_dim, rnn_units, num_layers, cheb_k, horizon):
        super(AGCRN, self).__init__()
        self.num_nodes = num_nodes
        self.output_dim = out_dim
        self.hidden_dim = rnn_units
        self.horizon = horizon

        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(num_nodes, in_dim, rnn_units, cheb_k, embed_dim, num_layers)

        self.end_conv = nn.Conv2d(1, horizon * out_dim, kernel_size=(1, rnn_units), bias=True)
        
        self.project_linear1 = nn.Linear(rnn_units, rnn_units)
        self.project_bn = nn.BatchNorm1d(rnn_units)
        self.project_relu = nn.ReLU()
        self.project_linear2 = nn.Linear(rnn_units, rnn_units)

    def forward(self, source, em_t=None):
        # source (B, T, N, D)
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source[:,:,:,:1], init_state, self.node_embeddings, em_t)
        output = output[:, -1:, :, :]  # (B, 1, N, D)

        # regression head
        pred = self.end_conv(output)  # (B, T, N, 1)
        
        # project head
        rep = torch.squeeze(output) # (B, N, D)
        rep = self.project_linear1(rep)
        rep = rep.transpose(1,2)
        rep = self.project_bn(rep)
        rep = rep.transpose(1,2)
        rep = self.project_relu(rep)
        rep = self.project_linear2(rep)

        return pred, rep
