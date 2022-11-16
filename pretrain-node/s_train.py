import os
import time
import util
import random
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agcrn import AGCRN

torch.set_num_threads(3)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:7', help='device name')
parser.add_argument('--dataset', type=str, default='d4', help='dataset name')
parser.add_argument('--in_dim', type=int, default=1, help='input dimension')
parser.add_argument('--out_dim', type=int, default=1, help='output dimension')
parser.add_argument('--rnn_units', type=int, default=64, help='hidden dimension')
parser.add_argument('--num_layers', type=int, default=2, help='number of layer')
parser.add_argument('--cheb_k', type=int, default=2, help='cheb order')
parser.add_argument('--horizon', type=int, default=12, help='sequence length')

parser.add_argument('--ft', type=int, default=1, help='linear evaluation or fine tuning')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lrate_enc', type=float, default=1e-4, help='learning rate for encoder')
parser.add_argument('--lrate_dec', type=float, default=1e-3, help='learning rate for decoder')
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--checkpoint', type=str, default='save/xxx.pth', help='path for pre-trained encoder')
parser.add_argument('--seed',type=int, default=0, help='random seed')
args = parser.parse_args()
print(args)

if args.dataset == 'd4':
    adj_data = '../data/adj_mx_04.pkl'
    input_data = '../data/PEMS-04'
    num_nodes = 307
    embed_dim = 10
elif args.dataset == 'd8':
    adj_data = '../data/adj_mx_08.pkl'
    input_data = '../data/PEMS-08'
    num_nodes = 170
    embed_dim = 2
save = 'save'
if not os.path.exists(save):
    os.makedirs(save)
save += '/'


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] =str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


class decoder(nn.Module):
    def __init__(self, out_dim, rnn_units, horizon, ft):
        super(decoder, self).__init__()
        self.ft = ft
        if self.ft:
            self.end_conv = nn.Conv2d(1, horizon * out_dim, kernel_size=(1, rnn_units), bias=True)
        else:
            self.end_conv_1 = nn.Conv2d(rnn_units, rnn_units * 8, kernel_size=(1, 1), bias=True)
            self.end_conv_2 = nn.Conv2d(rnn_units * 8, horizon * out_dim, kernel_size=(1, 1), bias=True)
        
    def forward(self, input):
        if self.ft:
            x = self.end_conv(input)
        else:
            x = input.transpose(1,3)
            x = F.relu(self.end_conv_1(x))
            x = self.end_conv_2(x)
        return x


class trainer():
    def __init__(self, device, model, dec, scaler, ft, lrate_enc, lrate_dec):
        self.device = device
        self.encoder = model
        self.encoder.to(device)
        self.decoder = dec
        self.decoder.to(device)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)
        self.scaler = scaler
        self.loss = util.masked_mae
        self.ft = ft
        
        if ft:
            self.optimizer = optim.Adam([{'params': self.encoder.parameters(), 'lr': lrate_enc},
                                         {'params': self.decoder.parameters(), 'lr': lrate_dec}])
        else:
            self.optimizer = optim.Adam(self.decoder.parameters(), lr=lrate_dec)

    def train(self, input, real_val):
        if self.ft:
            self.encoder.train()
        else:
            self.encoder.eval()
        self.decoder.train()
        self.optimizer.zero_grad()
        
        output, _ = self.encoder(input)
        output = self.decoder(output)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real_val, 0.0)
        mape = util.masked_mape(predict, real_val, 0.0).item()
        rmse = util.masked_rmse(predict, real_val, 0.0).item()
        loss.backward()
        self.optimizer.step()
        return loss.item(), mape, rmse
    
    def eval(self, input, real_val):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            output, _ = self.encoder(input)
            output = self.decoder(output)
            predict = self.scaler.inverse_transform(output)
 
            loss = self.loss(predict, real_val, 0.0)
            mape = util.masked_mape(predict, real_val, 0.0).item()
            rmse = util.masked_rmse(predict, real_val, 0.0).item()
            return loss.item(), mape, rmse
    

set_seed(args.seed)
device = torch.device(args.device)
dataloader = util.load_dataset(input_data, args.batch_size, args.batch_size, args.batch_size)
scaler = dataloader['scaler']

model = AGCRN(num_nodes, embed_dim, args.in_dim, args.out_dim, args.rnn_units, args.num_layers, args.cheb_k, args.horizon)
dec = decoder(args.out_dim, args.rnn_units, args.horizon, args.ft)

engine = trainer(device, model, dec, scaler, args.ft, args.lrate_enc, args.lrate_dec)
engine.encoder.load_state_dict(torch.load(args.checkpoint))
print('Encoder load successfully')


print("Start training...")
his_loss =[]
train_time = []
val_time = []
min_loss = float('inf')
for i in range(1, args.epochs + 1):
    train_loss = []
    train_mape = []
    train_rmse = []
    t1 = time.time()
    dataloader['train_loader'].shuffle()
    for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
        trainx = torch.Tensor(x).to(device)
        trainy = torch.Tensor(y).to(device)
        metrics = engine.train(trainx, trainy[:,:,:,:1])

        train_loss.append(metrics[0])
        train_mape.append(metrics[1])
        train_rmse.append(metrics[2])
    t2 = time.time()
    train_time.append(t2-t1)

    valid_loss = []
    valid_mape = []
    valid_rmse = []
    s1 = time.time()
    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testy = torch.Tensor(y).to(device)
        metrics = engine.eval(testx, testy[:,:,:,:1])

        valid_loss.append(metrics[0])
        valid_mape.append(metrics[1])
        valid_rmse.append(metrics[2])
    s2 = time.time()
    val_time.append(s2-s1)

    mtrain_loss = np.mean(train_loss)
    mtrain_mape = np.mean(train_mape)
    mtrain_rmse = np.mean(train_rmse)

    mvalid_loss = np.mean(valid_loss)
    mvalid_mape = np.mean(valid_mape)
    mvalid_rmse = np.mean(valid_rmse)
    his_loss.append(mvalid_loss)

    log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Train Time: {:.4f}/epoch, Valid Time: {:.4f}/epoch'
    print(log.format(i, mtrain_loss, mtrain_rmse, mtrain_mape, mvalid_loss, mvalid_rmse, mvalid_mape, (t2 - t1), (s2 - s1)))

    if min_loss > mvalid_loss:
        torch.save(engine.encoder.state_dict(), save + 'epoch_' + str(i) + '_' + str(round(mvalid_loss, 2)) + '_encoder.pth')
        torch.save(engine.decoder.state_dict(), save + 'epoch_' + str(i) + '_' + str(round(mvalid_loss, 2)) + '_decoder.pth')
        min_loss = mvalid_loss


bestid = np.argmin(his_loss)
engine.encoder.load_state_dict(torch.load(save + 'epoch_' + str(bestid + 1) + '_' + str(round(his_loss[bestid], 2)) + '_encoder.pth'))
engine.decoder.load_state_dict(torch.load(save + 'epoch_' + str(bestid + 1) + '_' + str(round(his_loss[bestid], 2)) + '_decoder.pth'))
log = 'Best Valid MAE: {:.4f}'
print(log.format(round(his_loss[bestid], 4)))

valid_loss = []
valid_mape = []
valid_rmse = []
for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
    testx = torch.Tensor(x).to(device)
    testy = torch.Tensor(y).to(device)
    metrics = engine.eval(testx, testy[:,:,:,:1])

    valid_loss.append(metrics[0])
    valid_mape.append(metrics[1])
    valid_rmse.append(metrics[2])
mvalid_loss = np.mean(valid_loss)
mvalid_mape = np.mean(valid_mape)
mvalid_rmse = np.mean(valid_rmse)
log = 'Recheck Valid MAE: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}'
print(log.format(np.mean(mvalid_loss), np.mean(mvalid_rmse), np.mean(mvalid_mape)))

outputs = []
for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    testx = torch.Tensor(x).to(device)
    engine.encoder.eval()
    engine.decoder.eval()
    with torch.no_grad():
        output, _ = engine.encoder(testx)
        output = engine.decoder(output)
        output = output.transpose(1,3)
        outputs.append(torch.squeeze(output, dim=1))

realy = torch.Tensor(dataloader['y_test']).to(device)
realy = realy.transpose(1,3)[:,0,:,:]
preds = torch.cat(outputs, dim=0)
preds = preds[:realy.size(0),...]

test_loss = []
test_mape = []
test_rmse = []
res = []
for k in range(args.horizon):
    pred = scaler.inverse_transform(preds[:,:,k])
    real = realy[:,:,k]
    metrics = util.metric(pred, real)
    log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
    print(log.format(k + 1, metrics[0], metrics[2], metrics[1]))
    test_loss.append(metrics[0])
    test_mape.append(metrics[1])
    test_rmse.append(metrics[2])
    if k in [2, 5, 11]:
        res += [metrics[0], metrics[2], metrics[1]]
mtest_loss = np.mean(test_loss)
mtest_mape = np.mean(test_mape)
mtest_rmse = np.mean(test_rmse)

log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
print(log.format(mtest_loss, mtest_rmse, mtest_mape))
res += [mtest_loss, mtest_rmse, mtest_mape]
res = [round(r, 4) for r in res]
print(res)

print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
print("Average Inference Time: {:.4f} secs/epoch".format(np.mean(val_time)))
