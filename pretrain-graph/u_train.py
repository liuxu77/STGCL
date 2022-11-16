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

parser.add_argument('--fn_t', type=int, default=12, help='filter negatives threshold, 12 means 1 hour')
parser.add_argument('--im_t', type=float, default=0.01, help='input masking threshold')
parser.add_argument('--tempe', type=float, default=0.1, help='temperature parameter')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lrate', type=float, default=0.003, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--patience', type=int, default=10, help='patience of early stopping')
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

set_seed(args.seed)
device = torch.device(args.device)
dataloader = util.load_dataset(input_data, args.batch_size, args.batch_size, args.batch_size)

model = AGCRN(num_nodes, embed_dim, args.in_dim, args.out_dim, args.rnn_units, args.num_layers, args.cheb_k, args.horizon)

engine = trainer(device, model, args.fn_t, args.im_t, args.tempe, args.lrate)


print('Start training...')
his_loss =[]
train_time = []
min_loss = float('inf')
wait = 0
for i in range(1, args.epochs + 1):
    train_loss = []
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
        train_neg.append(metrics[1])
        train_acc.append(metrics[2])
        input_diff.append(metrics[3])
    t2 = time.time()
    train_time.append(t2-t1)

    mtrain_loss = np.mean(train_loss)
    mtrain_neg = np.mean(train_neg)
    mtrain_acc = np.mean(train_acc)
    minput_diff = np.mean(input_diff)
    his_loss.append(mtrain_loss)

    log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train Acc: {:.4f}, Train Neg: {:.4f}, Input Diff: {:.4f}, Train Time: {:.4f}/epoch'
    print(log.format(i, mtrain_loss, mtrain_acc, mtrain_neg, minput_diff, (t2 - t1)))

    if min_loss > mtrain_loss:
        torch.save(engine.model.state_dict(), save + 'epoch_' + str(i) + '_' + str(round(mtrain_loss, 2)) + '.pth')
        min_loss = mtrain_loss
        wait = 0
    else:
        wait += 1
    
    if wait == args.patience:
        print('early stop')
        break

print('Best Train Loss', np.argmin(his_loss) + 1, np.min(his_loss))
print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
