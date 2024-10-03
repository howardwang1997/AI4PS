import joblib
import torch
from torch import nn
from .model.NN import SolutionNet
from .model.bert_transformer import TransformerConvLayer


def make_checkpoint(checkpoints0:list, checkpoints1:list, embeddings_path='../GNN/config/embeddings_100_cgcnn.pt', n_conv=0, n_fc=1, n_gt=1, device='gpu'):
    map_checkpoint = {
        0: checkpoints0,
        1: checkpoints1,
    }
    n_ensembles = [len(map_checkpoint[0]), len(map_checkpoint[1])]

    # hyperparameters
    atom_fea_len = 92
    nbr_fea_len = 42

    # define atom_vocab, dataset, model, trainer
    embeddings = torch.load(embeddings_path)
    if device == 'gpu':
        embeddings = embeddings.cuda()
    module = nn.ModuleList([TransformerConvLayer(128, 32, 8, edge_dim=42, dropout=0.0) for _ in range(n_conv)]), \
        nn.ModuleList([TransformerConvLayer(42, 24, 8, edge_dim=30, dropout=0.0) for _ in range(n_conv)])

    # module_1 = nn.ModuleList([TransformerConvLayer(128, 32, 8, edge_dim=42, dropout=0.0) for _ in range(n_conv)]), \
    #     nn.ModuleList([TransformerConvLayer(42, 24, 8, edge_dim=30, dropout=0.0) for _ in range(n_conv)])

    ckpt_0 = [torch.load(m) for m in map_checkpoint[0]]
    ckpt_1 = [torch.load(m) for m in map_checkpoint[1]]

    ctgns_0 = [SolutionNet(atom_fea_len, nbr_fea_len,
                            embeddings=embeddings, h_fea_len=128, n_conv=n_conv,
                            n_fc=n_fc, n_gt=n_gt, module=module, norm=True, drop=0) for _ in range(n_ensembles[0])]

    for i in range(n_ensembles[0]):
        ctgns_0[i].load_state_dict(ckpt_0[i]['state_dict'], strict=True)

    ctgns_1 = [SolutionNet(atom_fea_len, nbr_fea_len,
                            embeddings=embeddings, h_fea_len=128, n_conv=n_conv,
                            n_fc=2, n_gt=n_gt, module=module, norm=True, drop=0) for _ in range(n_ensembles[1])]

    for i in range(n_ensembles[1]):
        ctgns_1[i].load_state_dict(ckpt_1[i]['state_dict'], strict=True)
    # loss_0 = ckpt_0['loss']
    # loss_1 = ckpt_1['loss']
    checkpoint = {
        'models_0': ctgns_0,
        # 'loss_0': loss_0,
        'models_1': ctgns_1,
        # 'loss_1': loss_1
    }
    # done
    return checkpoint


def get_atom_vocab(atom_vocab_path):
    atom_vocab = joblib.load(atom_vocab_path)
    return atom_vocab
