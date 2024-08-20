import sys
sys.path.append('..')

import joblib
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader

from GNN.make_checkpoint import make_checkpoint
from GNN.data import MoleculesDataset
from GNN.gnn_utils import dataset_converter
from GNN.train import Trainer


class BayesianPredictor:
    def __init__(self,
                 checkpoint0,
                 checkpoint1,
                 device='gpu',
                 save_path='bayesian_sd.pt'):
        models = make_checkpoint(checkpoint0, checkpoint1)
        self.model_soqy = models['model_0']
        self.model_abs = models['model_1']
        self.val_loss_soqy = models['loss_0']
        self.val_loss_abs = models['loss_1']

        if device == 'gpu':
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.model_soqy = self.model_soqy.to(self.device)
        self.model_abs = self.model_abs.to(self.device)
        self.model_soqy.eval()
        self.model_abs.eval()

        # self.save_path = save_path
        
        # init dataloader
        self.embeddings_path = '/mlx_devbox/users/howard.wang/playground/new_benchmark/CrysToGraph/CrysToGraph/config/embeddings_100_cgcnn.pt'
        self.atom_vocab = joblib.load('/mlx_devbox/users/howard.wang/playground/new_benchmark/CrysToGraph/CrysToGraph/config/100_vocab.jbl')
        self.embeddings = torch.load(self.embeddings_path).to(self.device)

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
            trainer_soqy = Trainer(self.model_soqy, name='soqy' , classification=False)
            trainer_abs = Trainer(self.model_abs, name='abs' , classification=False)
            pred_soqy, _ = trainer_soqy.predict(test_loader=td)
            pred_abs, _ = trainer_abs.predict(test_loader=td)

            # soqy = self.model_soqy(test_inputs).reshape(-1)
            # absorption = self.model_abs(test_inputs).reshape(-1)
            soqy = pred_soqy.reshape(-1)
            absorption = pred_abs.reshape(-1)
        outputs = (soqy.to("cpu"), absorption.to("cpu"))
        return outputs
