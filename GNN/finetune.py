if __name__ == '__main__':
    import os
    import json
    import joblib
    import argparse

    from rdkit import Chem

    # for debug
    with open('/mlx_devbox/users/howard.wang/playground/molllm/datasets/dataset_close_5_index_rmo3.json') as f:
        d = json.load(f)
    data = d['soqy']
    all_data = []
    aq_data = []
    for i in range(len(data)):
        d = data[i]
        try:
            mol = Chem.MolFromSmiles(d[1])
            sol = Chem.MolFromSmiles(d[2])
            if mol and sol:
                if '*' in d[1]:
                    print(f'MOLECULE ERROR in soqy {i}')
                    continue
                else:
                    all_data.append(d)
                    if d[2] == 'O':
                        aq_data.append(d)
            else:
                print(f'MOLECULE ERROR in soqy {i}')
        except TypeError:
                print(f'SOLVENT ERROR in soqy {i}, {d}')
            # print(d[0])
            # print(d[1])
    print(len(data), len(all_data), len(aq_data))

    # mb = MatbenchBenchmark(autoload=False)
    # mb = mb.from_preset('matbench_v0.1', 'structure')

    parser = argparse.ArgumentParser(description='Run CrysToGraph on matbench.')
    parser.add_argument('--task', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--atom_fea_len', type=int, default=92)
    parser.add_argument('--nbr_fea_len', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_conv', type=int, default=0)
    parser.add_argument('--n_fc', type=int, default=1)
    parser.add_argument('--n_gt', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=-1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--milestone1', type=int, default=-1)
    parser.add_argument('--milestone2', type=int, default=-1)
    parser.add_argument('--rmtree', action='store_true')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--remarks', type=str, default='')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import numpy as np
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader

    # torch.multiprocessing.set_start_method('spawn')
    # os.environ["MKL_SERVICE_FORCE_INTEL"] = 1

    # from matbench.bench import MatbenchBenchmark

    from data import MoleculesDataset
    from train import Trainer
    from model.NN import  SolutionNet
    from model.bert_transformer import TransformerConvLayer
    from gnn_utils import dataset_converter, split

    classification = False
    name = 'soqy_finetune_aqueous'
    _, val_set = split(all_data, seed=args.seed, fold=args.fold, val_ratio=0.1)
    train_set = aq_data

    # hyperparameters
    atom_fea_len = args.atom_fea_len
    nbr_fea_len = args.nbr_fea_len
    batch_size = args.batch_size
    epochs = args.epochs
    weight_decay = args.weight_decay
    lr = args.lr
    grad_accum = args.grad_accum
    pretrained = False if args.checkpoint == '' else True

    fold = args.fold

    embeddings_path = ''
    embeddings_path = '/mlx_devbox/users/howard.wang/playground/new_benchmark/CrysToGraph/CrysToGraph/config/embeddings_100_cgcnn.pt'
    atom_vocab = joblib.load('/mlx_devbox/users/howard.wang/playground/new_benchmark/CrysToGraph/CrysToGraph/config/100_vocab.jbl')

    # mkdir
    try:
        os.mkdir(name)
    except FileExistsError:
        pass

    train_names, train_inputs, train_sols, train_outs = dataset_converter(train_set)

    if epochs == -1:
        epochs = 10
    grad_accum = 1
    milestones = [99990, 99999]

    # define atom_vocab, dataset, model, trainer
    embeddings = torch.load(embeddings_path).cuda()
    atom_vocab = joblib.load('atom_vocab.jbl')
    cd = MoleculesDataset(root='/mnt/bn/ai4s-hl/bamboo/pyscf_data/hongyi/ps/soqy',
                        atom_vocab=atom_vocab,
                        inputs=train_inputs,
                        solvents=train_sols,
                        names=train_names,
                        outputs=train_outs)
    module = nn.ModuleList([TransformerConvLayer(128, 32, 8, edge_dim=args.nbr_fea_len, dropout=0.0) for _ in range(args.n_conv)]), \
            nn.ModuleList([TransformerConvLayer(args.nbr_fea_len, 24, 8, edge_dim=30, dropout=0.0) for _ in range(args.n_conv)])
    # module = None
    drop = 0.0 if not classification else 0.2
    ctgn = SolutionNet(atom_fea_len, nbr_fea_len,
                    embeddings=embeddings, h_fea_len=128, n_conv=args.n_conv,
                    n_fc=args.n_fc, n_gt=args.n_gt, module=module, norm=True, drop=drop)

    if pretrained:
        ctgn.load_state_dict(torch.load(args.checkpoint)['state_dict'], strict=True)
    optimizer = optim.AdamW(ctgn.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    trainer = Trainer(ctgn, name='%s_%d_%s' % (name, fold, args.remarks), classification=classification)

    # predict
    # test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
    test_names, test_inputs, test_sols, test_outs = dataset_converter(val_set)
    td = MoleculesDataset(root='/mnt/bn/ai4s-hl/bamboo/pyscf_data/hongyi/ps/soqy',
                        atom_vocab=atom_vocab,
                        inputs=test_inputs,
                        solvents=test_sols,
                        names=test_names,
                        outputs=test_outs)
    test_loader = DataLoader(td, batch_size=2, shuffle=False, collate_fn=cd.collate_line_graph, pin_memory=True, pin_memory_device='cuda')

    # train
    train_loader = DataLoader(cd, batch_size=batch_size, shuffle=True, collate_fn=cd.collate_line_graph, num_workers=0, pin_memory=True, pin_memory_device='cuda')
    trainer.train(train_loader=train_loader,
                optimizer=optimizer,
                epochs=epochs,
                scheduler=scheduler,
                grad_accum=grad_accum,
                val_freq=99999,
                test_loader=test_loader)

    # predict
    predictions, metrics = trainer.predict(test_loader=test_loader)
    loss = metrics[1]
    targets_p_t = {
        'predictions': torch.tensor(predictions).cpu(),
        'targets': torch.tensor(test_outs).cpu()
    }
    trainer.save_state_dict(f'/mnt/bn/ai4s-hl/bamboo/hongyi/debug/checkpoints/{name}_{args.remarks}_seed_{args.seed}_fold_{fold}_epoch_{epochs}_checkpoint.pt', loss, targets_p_t)
