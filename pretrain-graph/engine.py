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
        res[0, cnt] = True
        cnt += 1
        m.append(res)
    m = torch.cat(m)
    return m


class trainer():
    def __init__(self, device, model, fn_t, im_t, tempe, lrate):
        self.device = device
        self.model = model
        self.model.to(device)
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate)
        
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


        norm1 = reps[0].norm(dim=1)
        norm2 = reps[1].norm(dim=1)
        sim_matrix = torch.mm(reps[0], torch.transpose(reps[1], 0, 1)) / torch.mm(norm1.view(-1, 1), norm2.view(1, -1))
        sim_matrix = torch.exp(sim_matrix / self.tempe)

        diag = bs
        pos_sim = sim_matrix[range(diag), range(diag)]

        avg_neg = diag - 1
        if self.fn_t:
            m = filter_negative(input, self.fn_t)
            s = torch.sum(m, dim=1) - 1
            avg_neg = torch.mean(s * 1.0).cpu().item()
            sim_matrix = sim_matrix * m
                
        max_id = torch.argmax(sim_matrix, dim=1)
        labels = torch.arange(diag).to(self.device)
        corr_num = torch.sum(max_id==labels).item()
        avg_acc = corr_num / diag

        u_loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = torch.mean(-torch.log(u_loss))
        
        loss.backward()
        self.optimizer.step()
        return loss.item(), avg_neg, avg_acc, diff
