import util
import torch
import torch.optim as optim
import numpy as np
from scipy.fftpack import dct, idct

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
    def __init__(self, device, model, adj_m, scaler, method, fn_t, em_t, im_t, ts_t, ism_t, ism_e, tempe, lam, lrate):
        self.device = device
        self.model = model
        self.model.to(device)
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)
        self.scaler = scaler
        self.loss = util.masked_mae
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate)
        
        self.nor_adj = util.asym_adj(adj_m)
        self.method = method
        self.fn_t = fn_t
        self.em_t = em_t
        self.im_t = im_t
        self.ts_t = ts_t
        self.ism_t = ism_t
        self.ism_e = ism_e
        self.tempe = tempe
        self.lam = lam

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        bs = input.shape[0]
        frame = input.shape[1]
        num_node = input.shape[2]
        
        output, rep = self.model(input)
        predict = self.scaler.inverse_transform(output)
        s_loss = self.loss(predict, real_val, 0.0)
        mape = util.masked_mape(predict, real_val, 0.0).item()
        rmse = util.masked_rmse(predict, real_val, 0.0).item()
        
        if self.method == 'pure':
            s_loss.backward()
            self.optimizer.step()
            return s_loss.item(), mape, rmse, 0, 0, 0, 0, 0
        
        elif self.method == 'node':
            diff = 0
            if self.im_t or self.ts_t or self.ism_t:
                input_ = input.detach().clone()
                
                if self.im_t:
                    rand = torch.rand(bs, frame, num_node).to(self.device)
                    input_[:,:,:,0] = input_[:,:,:,0] * (rand >= self.im_t)
                
                if self.ts_t:
                    r = self.scaler.transform(real_val[:,:,:,0])
                    s = torch.cat((input_[:,:,:,0], r), dim=1)[:,:frame+1,:]
                    rand = (1 - self.ts_t) * torch.rand(bs, 1, 1) + self.ts_t
                    rand = rand.expand(bs, frame + 1, num_node).to(self.device)
                    input_[:,:,:,0] = (s * rand + torch.roll(s, -1, 1) * (1 - rand))[:,:frame,:]

                if self.ism_t:
                    r = self.scaler.transform(real_val[:,:,:,0])
                    s = torch.cat((input_[:,:,:,0], r), dim=1).cpu()
                    o = []
                    for i in range(bs):
                        t = np.transpose(np.array(s[i]))
                        m1 = np.ones((num_node, self.ism_e))
                        m2 = np.random.uniform(low=self.ism_t, high=1.0, size=(num_node, t.shape[1] - self.ism_e))
                        m2 = np.matmul(self.nor_adj, m2)
                        m2 = np.matmul(self.nor_adj, m2)
                        mall = np.concatenate((m1, m2), axis=1)
                        t = dct(t, norm='ortho')
                        t = np.multiply(t, mall)
                        t = idct(t, norm='ortho')
                        o.append(np.transpose(t))
                    o = np.stack(o)
                    input_[:,:,:,0] = torch.tensor(o[:,:frame,:]).to(self.device)
                
                diff = torch.mean(torch.abs(input_[:,:,:,0] - input[:,:,:,0])).item()
               
                if not self.em_t:
                    _, rep_aug = self.model(input_)
                else:
                    _, rep_aug = self.model(input_, self.em_t)

            else:
                _, rep_aug = self.model(input, self.em_t)


            ##### rep (bs, node, hid)
            # temporal contrast
            tempo_rep = rep.transpose(0,1) # (node, bs, 256)
            tempo_rep_aug = rep_aug.transpose(0,1)
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
            spatial_norm = rep.norm(dim=2).unsqueeze(dim=2)
            spatial_norm_aug = rep_aug.norm(dim=2).unsqueeze(dim=2)
            spatial_matrix = torch.matmul(rep, rep_aug.transpose(1,2)) / torch.matmul(spatial_norm, spatial_norm_aug.transpose(1,2))
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
            u_loss = torch.mean(-torch.log(ratio))
            #####     

            loss = s_loss + self.lam * u_loss
            loss.backward()
            self.optimizer.step()
            return loss.item(), mape, rmse, s_loss.item(), u_loss.item(), 0, 0, diff      

    def eval(self, input, real_val):
        self.model.eval()
        with torch.no_grad():
            output, _ = self.model(input)
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real_val, 0.0)
            mape = util.masked_mape(predict, real_val, 0.0).item()
            rmse = util.masked_rmse(predict, real_val, 0.0).item()
            return loss.item(), mape, rmse
