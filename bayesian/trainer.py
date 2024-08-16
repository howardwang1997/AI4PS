import sys
sys.path.append('..')

from matplotlib import pyplot as plt
import torch
from ..GNN.make_checkpoint import make_checkpoint


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

        self.save_path = save_path

    def train(self, exp_data, epochs=100):
        pass

    def predict(self, parameters):
        test_inputs = parameters
        """
        NEED IMPLEMENTATION, data conversion
        """
        test_inputs = test_inputs.to(self.device)

        with torch.no_grad():
            soqy = self.model_soqy(test_inputs).reshape(-1)
            absorption = self.model_abs(test_inputs).reshape(-1)
        outputs = (soqy.to("cpu"), absorption.to("cpu"))
        return outputs
