import util
import torch
import torch.optim as optim

def filter_negative(input_, thres):
    times = input_[:, 0, 0, 1]
        
    m = []
    cnt = 0
    c = thres / 288
    for t in times: 
        if t < c:
            st = times < 0
            gt = torch.logical_and(times <= (1 + t - c), times >= (t + c))
        elif t > (1 - c):
            st = torch.logical_and(times <= (t - c), times >= (c + t - 1))
            gt = times > 1
        else:
            st = times <= (t - c)
            gt = times >= (t + c)
        
        res = torch.logical_or(st, gt).view(1, -1)
#         res[0, cnt] = True
#         cnt += 1
        m.append(res)
    m = torch.cat(m)
    return m


class trainer():
    def __init__(self, device, model, adj_m, fn_t, im_t, tempe, lrate):
        self.device = device
        self.model = model
        self.model.to(device)
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate)
        
        self.nor_adj = util.asym_adj(adj_m)
        self.fn_t = fn_t
        self.im_t = im_t
        self.tempe = tempe 

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        bs = input.shape[0]
        frame = input.shape[1]
        num_node = input.shape[2]
        
        reps = []
        for _ in range(2):
            input_ = input.detach().clone()
            rand = torch.rand(bs, frame, num_node).to(self.device)
            input_[:,:,:,0] = input_[:,:,:,0] * (rand >= self.im_t)
            
            diff = torch.mean(torch.abs(input_[:,:,:,0] - input[:,:,:,0])).item()

            _, rep = self.model(input_)
            
            reps.append(rep)    
        
        ##### rep (bs, node, 256)
        # temporal contrast
        tempo_rep = reps[0].transpose(0,1) # (node, bs, 256)
        tempo_rep_aug = reps[1].transpose(0,1)
        tempo_norm = tempo_rep.norm(dim=2).unsqueeze(dim=2)
        tempo_norm_aug = tempo_rep_aug.norm(dim=2).unsqueeze(dim=2)
        tempo_matrix = torch.matmul(tempo_rep, tempo_rep_aug.transpose(1,2)) / torch.matmul(tempo_norm, tempo_norm_aug.transpose(1,2))
        tempo_matrix = torch.exp(tempo_matrix / self.tempe)

        # temporal negative filter
        if self.fn_t:
            m = filter_negative(input, self.fn_t)
            tempo_matrix = tempo_matrix * m
        tempo_neg = torch.sum(tempo_matrix, dim=2) # (node, bs)

        # spatial contrast
        spatial_norm = reps[0].norm(dim=2).unsqueeze(dim=2)
        spatial_norm_aug = reps[1].norm(dim=2).unsqueeze(dim=2)
        spatial_matrix = torch.matmul(reps[0], reps[1].transpose(1,2)) / torch.matmul(spatial_norm, spatial_norm_aug.transpose(1,2))
        spatial_matrix = torch.exp(spatial_matrix / self.tempe)

        diag = torch.eye(num_node, dtype=torch.bool).to(self.device)
        pos_sum = torch.sum(spatial_matrix * diag, dim=2) # (bs, node)

        # spatial negative filter
        if self.fn_t:
            adj = (self.nor_adj == 0)
            adj = torch.tensor(adj).to(self.device)
            adj = adj + diag
            spatial_matrix = spatial_matrix * adj
        spatial_neg = torch.sum(spatial_matrix, dim=2) # (bs, node)

        ratio = pos_sum / (spatial_neg + tempo_neg.transpose(0,1) - pos_sum)
        loss = torch.mean(-torch.log(ratio))
        #####
        
        loss.backward()
        self.optimizer.step()
        return loss.item(), 0, 0, diff
