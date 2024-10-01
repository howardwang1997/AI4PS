import sys
sys.path.append('..')

import joblib
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader

from GNN.make_checkpoint import make_checkpoint, get_atom_vocab
from GNN.data import MoleculesDataset
from GNN.gnn_utils import dataset_converter
from GNN.train import Trainer


class BayesianPredictor:
    def __init__(self,
                 checkpoints0,
                 checkpoints1,
                 device='gpu',
                 save_path='bayesian_sd.pt'):
        self.embeddings_path = '/mlx_devbox/users/howard.wang/playground/new_benchmark/CrysToGraph/CrysToGraph/config/embeddings_100_cgcnn.pt'
        self.atom_vocab = get_atom_vocab('/mlx_devbox/users/howard.wang/playground/new_benchmark/CrysToGraph/CrysToGraph/config/100_vocab.jbl')
        # self.embeddings = torch.load(self.embeddings_path).to(self.device)

        models = make_checkpoint(checkpoints0, checkpoints1, self.embeddings_path, device=device)
        # modify
        self.model_soqy = models['model_0']
        self.model_abs = models['model_1']
        # self.val_loss_soqy = models['loss_0']
        # self.val_loss_abs = models['loss_1']

        if device == 'gpu':
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.model_soqy = [m.to(self.device) for m in self.model_soqy]
        self.model_abs = [m.to(self.device) for m in self.model_abs]
        for m in self.model_soqy:
            m.eval()
        for m in self.model_abs:
            m.eval()

        # self.save_path = save_path
        
    def get_dataloader(self, data):
        """
        data: List[List[str, str]]
        """
        mols, sols = dataset_converter(data)
        td = MoleculesDataset(root='/mnt/bn/ai4s-hl/bamboo/pyscf_data/hongyi/ps',
                              atom_vocab=self.atom_vocab,
                              inputs=mols,
                              solvents=sols,
                              outputs=[0]*len(sols))
        
        return td
        
    def train(self, exp_data, epochs=100):
        pass

    def predict(self, parameters):
        test_inputs = parameters
        # possibly a data conversion step

        td = self.get_dataloader(test_inputs)

        with torch.no_grad():
            soqy = []
            abs = []
            for m in self.model_soqy:
                trainer_soqy = Trainer(m, name='soqy', classification=False, cuda=self.device=='gpu')
                pred_soqy, _ = trainer_soqy.predict(test_loader=td)
                soqy.append(pred_soqy.reshape(1, -1))
            soqy = torch.cat(soqy, dim=0).to('cpu')

            for m in self.model_abs:
                trainer_abs = Trainer(m, name='abs', classification=False, cuda=self.device=='gpu')
                pred_abs, _ = trainer_abs.predict(test_loader=td)
                abs.append(pred_abs.reshape(1, -1))
            abs = torch.cat(abs, dim=0).to('cpu')
        outputs = {
            'soqy': soqy,
            'soqy_mean': torch.mean(soqy, dim=0),
            'soqy_std': torch.std(soqy, dim=0),
            'abs': abs,
            'abs_mean': torch.mean(abs, dim=0),
            'abs_std': torch.std(abs, dim=0),
        }

        # outputs = (soqy.to("cpu"), absorption.to("cpu"))
        return outputs
