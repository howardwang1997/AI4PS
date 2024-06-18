import os
import joblib
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from matbench.bench import MatbenchBenchmark

from data import MoleculeDataset, MoleculesDataset
from train import Trainer
from model.NN import CrysToGraphNet, SolutionNet
from model.bert_transformer import TransformerConvLayer
from gnn_utils import dataset_converter, split

# for debug
with open('../data/dataset_close_1.json') as f:
    d = json.load(f)
data = d['soqy']

mb = MatbenchBenchmark(autoload=False)
mb = mb.from_preset('matbench_v0.1', 'structure')

parser = argparse.ArgumentParser(description='Run CrysToGraph on matbench.')
parser.add_argument('--task', type=str, default='')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--atom_fea_len', type=int, default=92)
parser.add_argument('--nbr_fea_len', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_conv', type=int, default=3)
parser.add_argument('--n_fc', type=int, default=2)
parser.add_argument('--n_gt', type=int, default=0)
parser.add_argument('--epochs', type=int, default=-1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--grad_accum', type=int, default=1)
parser.add_argument('--milestone1', type=int, default=-1)
parser.add_argument('--milestone2', type=int, default=-1)
parser.add_argument('--rmtree', action='store_true')
parser.add_argument('--fold', type=int, default=-1)
parser.add_argument('--checkpoint0', type=str, default='')
parser.add_argument('--checkpoint1', type=str, default='')
parser.add_argument('--checkpoint2', type=str, default='')
parser.add_argument('--checkpoint3', type=str, default='')
parser.add_argument('--checkpoint4', type=str, default='')
parser.add_argument('--remarks', type=str, default='')
args = parser.parse_args()

map_checkpoint = {
    0: args.checkpoint0,
    1: args.checkpoint1,
    2: args.checkpoint2,
    3: args.checkpoint3,
    4: args.checkpoint4,
}

classification = False
name = 'soqy'
train_set, val_set = split(data)
fold = 0

# hyperparameters
atom_fea_len = args.atom_fea_len
nbr_fea_len = args.nbr_fea_len
batch_size = args.batch_size
epochs = args.epochs
weight_decay = args.weight_decay
lr = args.lr
grad_accum = args.grad_accum
pretrained = False if args.checkpoint == '' else True
separated_checkpoint = False
if args.checkpoint0 != '' and args.checkpoint1 != '' and args.checkpoint2 != '' and args.checkpoint3 != '' and args.checkpoint4 != '':
    pretrained = separated_checkpoint = True

embeddings_path = ''

# mkdir
try:
    os.mkdir(name)
except FileExistsError:
    pass

# for fold in task.folds:
#     if args.fold != -1 and fold != args.fold:
#         continue
#     if pretrained:
#         if separated_checkpoint:
#             checkpoint = map_checkpoint[fold]
#         else:
#             checkpoint = args.checkpoint
#     train_inputs, train_outputs = task.get_train_and_val_data(fold)

train_inputs, train_sols, train_outs = dataset_converter(train_set)

if epochs == -1:
    if len(train_outs) < 2000:
        epochs = 2000
    elif len(train_outs) < 10000:
        epochs = 1000
    elif len(train_outs) < 20000:
        epochs = 600
        grad_accum = 2
    else:
        epochs = 500
        grad_accum = 8

milestone2 = 99999
if args.milestone1 > 0:
    milestone1 = args.milestone1
    if args.milestone2 > milestone1:
        milestone2 = args.milestone2
else:
    milestone1 = int(epochs/3)

milestones = [milestone1, milestone2]

# define atom_vocab, dataset, model, trainer
embeddings = torch.load(embeddings_path).cuda()
atom_vocab = joblib.load('atom_vocab.jbl')
cd = MoleculesDataset(root=name,
                     atom_vocab=atom_vocab,
                     inputs=train_inputs,
                     solvents=train_sols,
                     outputs=train_outs)
module = nn.ModuleList([TransformerConvLayer(256, 32, 8, edge_dim=args.nbr_fea_len, dropout=0.0) for _ in range(args.n_conv)]), \
         nn.ModuleList([TransformerConvLayer(args.nbr_fea_len, 24, 8, edge_dim=30, dropout=0.0) for _ in range(args.n_conv)])
drop = 0.0 if not classification else 0.2
ctgn = SolutionNet(atom_fea_len, nbr_fea_len,
                   embeddings=embeddings, h_fea_len=256, n_conv=args.n_conv,
                   n_fc=args.n_fc, n_gt=args.n_gt, module=module, norm=True, drop=drop)

# if pretrained:
#     ctgn.load_state_dict(torch.load(checkpoint), strict=True)
optimizer = optim.AdamW(ctgn.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
trainer = Trainer(ctgn, name='%s_%d_%s' % (name, fold, args.remarks), classification=classification)

# train
train_loader = DataLoader(cd, batch_size=batch_size, shuffle=True, collate_fn=cd.collate_line_graph)
trainer.train(train_loader=train_loader,
              optimizer=optimizer,
              epochs=epochs,
              scheduler=scheduler,
              grad_accum=grad_accum)

# predict
# test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
test_inputs, test_sols, test_outs = dataset_converter(val_set)
cd = MoleculesDataset(root=name,
                     atom_vocab=atom_vocab,
                     inputs=test_inputs,
                     solvents=test_sols,
                     outputs=test_outs)
test_loader = DataLoader(cd, batch_size=2, shuffle=False, collate_fn=cd.collate_line_graph)
predictions, metrics = trainer.predict(test_loader=test_loader)
loss = metrics[1]
trainer.save_state_dict(f'config/{name}_checkpoint.pt', loss)
