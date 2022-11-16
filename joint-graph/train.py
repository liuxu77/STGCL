import os
import time
import util
import random
import torch
import argparse
import numpy as np
from agcrn import AGCRN
from engine import trainer

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

parser.add_argument('--method', type=str, default='graph', help='two choices: pure, graph')
parser.add_argument('--fn_t', type=int, default=12, help='filter negatives threshold, 12 means 1 hour')
parser.add_argument('--em_t', type=float, default=0, help='edge masking threshold')
parser.add_argument('--im_t', type=float, default=0.01, help='input masking threshold')
parser.add_argument('--ts_t', type=float, default=0, help='temporal shifting threshold')
parser.add_argument('--ism_t', type=float, default=0, help='input smoothing scale')
parser.add_argument('--ism_e', type=int, default=20, help='input smoothing entries, which means how much entries we keep untouch during scaling in the frequency domain')
parser.add_argument('--tempe', type=float, default=0.1, help='temperature parameter')
parser.add_argument('--lam', type=float, default=0.1, help='loss lambda')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lrate', type=float, default=0.003, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--seed',type=int, default=0, help='random seed')
args = parser.parse_args()
print(args) 

# sanity check
assert args.method in ['pure', 'graph'], 'Please specify the type of methods'
if args.method == 'graph':
    assert args.em_t or args.im_t or args.ts_t or args.ism_t, 'Please specify at least one data augmentations'

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

set_seed(args.seed)
device = torch.device(args.device)
_, _, adj_m = util.load_pickle(adj_data)
dataloader = util.load_dataset(input_data, args.batch_size, args.batch_size, args.batch_size)
scaler = dataloader['scaler'] 

model = AGCRN(num_nodes, embed_dim, args.in_dim, args.out_dim, args.rnn_units, args.num_layers, args.cheb_k, args.horizon)
nparam = sum([p.nelement() for p in model.parameters()])
print('Total parameters:', nparam)

engine = trainer(device, model, adj_m, scaler, args.method, args.fn_t, args.em_t, args.im_t, args.ts_t, args.ism_t, args.ism_e, args.tempe, args.lam, args.lrate)


print('Start training...')
his_loss =[]
train_time = []
val_time = []
min_loss = float('inf')
for i in range(1, args.epochs + 1):
    train_loss = []
    train_mape = []
    train_rmse = []
    train_sloss = []
    train_uloss = []
    train_neg = []
    train_acc = []
    input_diff = []
    t1 = time.time()
    dataloader['train_loader'].shuffle()
    for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
        trainx = torch.Tensor(x).to(device)
        trainy = torch.Tensor(y).to(device)            
        metrics = engine.train(trainx, trainy[:,:,:,:1])

        train_loss.append(metrics[0])
        train_mape.append(metrics[1])
        train_rmse.append(metrics[2])
        train_sloss.append(metrics[3])
        train_uloss.append(metrics[4])
        train_neg.append(metrics[5])
        train_acc.append(metrics[6])
        input_diff.append(metrics[7])
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
    mtrain_sloss = np.mean(train_sloss)
    mtrain_uloss = np.mean(train_uloss)
    mtrain_neg = np.mean(train_neg)
    mtrain_acc = np.mean(train_acc)
    minput_diff = np.mean(input_diff)

    mvalid_loss = np.mean(valid_loss)
    mvalid_mape = np.mean(valid_mape)
    mvalid_rmse = np.mean(valid_rmse)
    his_loss.append(mvalid_loss)
    
    log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train SupLoss: {:.4f}, Train UnsupLoss: {:.4f}, Train UnsupAcc: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Train Neg: {:.4f}, Input Diff: {:.4f}, Train Time: {:.4f}/epoch, Valid Time: {:.4f}/epoch'
    print(log.format(i, mtrain_loss, mtrain_sloss, mtrain_uloss, mtrain_acc, mtrain_rmse, mtrain_mape, mvalid_loss, mvalid_rmse, mvalid_mape, mtrain_neg, minput_diff, (t2 - t1), (s2 - s1)))

    if min_loss > mvalid_loss:
        torch.save(engine.model.state_dict(), save + 'epoch_' + str(i) + '_' + str(round(mvalid_loss, 2)) + '.pth')
        min_loss = mvalid_loss


bestid = np.argmin(his_loss)
engine.model.load_state_dict(torch.load(save + 'epoch_' + str(bestid + 1) + '_' + str(round(his_loss[bestid], 2)) + '.pth'))
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
    engine.model.eval()
    with torch.no_grad():
        output, _ = engine.model(testx)
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
