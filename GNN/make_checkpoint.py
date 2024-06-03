import torch
from torch import nn
from model.NN import CrysToGraphNet
from model.bert_transformer import TransformerConvLayer


def make_checkpoint(checkpoint0, checkpoint1, n_conv=3, n_fc=1, n_gt=0):
    map_checkpoint = {
        0: checkpoint0,
        1: checkpoint1,
    }

    # hyperparameters
    atom_fea_len = 92
    nbr_fea_len = 42

    embeddings_path = '../GNN/config/embeddings_100_cgcnn.pt'

    # define atom_vocab, dataset, model, trainer
    embeddings = torch.load(embeddings_path).cuda()
    module_0 = nn.ModuleList([TransformerConvLayer(256, 32, 8, edge_dim=42, dropout=0.0) for _ in range(n_conv)]), \
        nn.ModuleList([TransformerConvLayer(42, 24, 8, edge_dim=30, dropout=0.0) for _ in range(n_conv)])

    module_1 = nn.ModuleList([TransformerConvLayer(256, 32, 8, edge_dim=42, dropout=0.0) for _ in range(n_conv)]), \
        nn.ModuleList([TransformerConvLayer(42, 24, 8, edge_dim=30, dropout=0.0) for _ in range(n_conv)])

    ckpt_0 = torch.load(map_checkpoint[0])
    ckpt_1 = torch.load(map_checkpoint[1])

    ctgn_0 = CrysToGraphNet(atom_fea_len, nbr_fea_len,
                            embeddings=embeddings, h_fea_len=256, n_conv=n_conv,
                            n_fc=n_fc, n_gt=n_gt, module=module_0, norm=True, drop=0)

    ctgn_0.load_state_dict(ckpt_0['state_dict'], strict=True)

    ctgn_1 = CrysToGraphNet(atom_fea_len, nbr_fea_len,
                            embeddings=embeddings, h_fea_len=256, n_conv=n_conv,
                            n_fc=n_fc, n_gt=n_gt, module=module_1, norm=True, drop=0)

    ctgn_1.load_state_dict(ckpt_1['state_dict'], strict=True)
    loss_0 = ckpt_0['loss']
    loss_1 = ckpt_1['loss']
    checkpoint = {
        'model_0': ctgn_0,
        'loss_0': loss_0,
        'model_1': ctgn_1,
        'loss_1': loss_1
    }

    return checkpoint