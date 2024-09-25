import time
import os
from math import sqrt
from sklearn.metrics import roc_auc_score, f1_score

import torch
from torch.autograd import Variable
from torch.nn import MSELoss, L1Loss, BCEWithLogitsLoss


class AverageRecorder(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer():
    def __init__(self, model, name='', cuda=True, classification: bool=False):
        self.batch_time = AverageRecorder()
        self.data_time = AverageRecorder()
        self.losses = AverageRecorder()

        self.cuda = cuda and torch.cuda.is_available()
        self.model = model
        self.name = name
        self.classification = classification
        if len(self.name) == 0:
            self.name = 'model'
        if self.cuda:
            self.model = model.cuda()

    def _step(self, inputs):
        if self.cuda:
            inputs = inputs.cuda()
        outputs = self.model(inputs)
        return outputs

    def train(self, train_loader, optimizer, epochs,
              scheduler=None, verbose_freq: int=100, checkpoint_freq: int=20,
              grad_accum: int=1, val_freq=0, test_loader=None):
        val = False
        self.val_loss_list = []
        if val_freq > 0 and test_loader:
            val = True

        lrs = True
        if scheduler is None:
            lrs = False

        if self.classification:
            self.criterion_task = BCEWithLogitsLoss()
        else:
            self.criterion_task = MSELoss()

        end = time.time()
        self.loss_list = []
        for epoch in range(epochs):
            self.model.train()
            loss_list = []
            self.losses.reset()
            self.data_time.reset()
            self.batch_time.reset()

            for i, data in enumerate(train_loader):
                self.data_time.update(time.time() - end)
                end = time.time()

                data_batch = tuple([d.to(torch.device('cuda:0')) for d in data[:-1]])
                output = self.model(data_batch)
                target = data[-1]

                target = Variable(target.float())
                if self.cuda:
                    target = target.cuda()
                loss = self.criterion_task(output, target)

                loss_list.append(loss.data.cpu().item())
                self.losses.update(loss.data.cpu().item(), len(target))

                loss /= grad_accum
                loss.backward()

                if i % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                self.batch_time.update(time.time() - end)
                end = time.time()

                if i % verbose_freq == 0:
                    self.verbose(epoch, i, len(train_loader))
                elif i == len(train_loader) - 1:
                    self.verbose(epoch, -1, len(train_loader))
            if lrs:
                scheduler.step()
            self.loss_list.append(loss_list)

            if val:
                if epoch % val_freq == 0:
                    outputs, metrics = self.predict(test_loader)
                    self.val_loss_list.append(float(metrics[0]))

            if epoch % checkpoint_freq == 0:
                self.save_checkpoints(epoch)
        self.save_state_dict()
        self.model.cuda()

    def predict(self, test_loader):
        self.model.eval()

        outputs = torch.Tensor()
        targets = torch.Tensor()

        end = time.time()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                self.data_time.update(time.time() - end)
                end = time.time()
                if self.cuda:
                    data_batch = tuple([d.to(torch.device('cuda:0')) for d in data[:-1]])
                else:
                    data_batch = tuple([d.to(torch.device('cpu')) for d in data[:-1]])
                output = self.model(data_batch).cpu()
                target = torch.tensor(data[-1]).reshape(-1)

                self.batch_time.update(time.time() - end)
                end = time.time()

                outputs = torch.cat((outputs, output), dim=0)
                targets = torch.cat((targets, target), dim=0)

        if self.classification:
            auc = roc_auc_score(targets, outputs>0)
            f1 = f1_score(targets, outputs>0)
            print('%s VALIDATION: ROC_AUC_SCORE= %.4f, F1_SCORE= %.4f' % (self.name, float(auc), float(f1)))
            metrics = (auc, f1)
        else:
            mae = L1Loss()(outputs, targets)
            rmse = sqrt(MSELoss()(outputs, targets))
            print('%s VALIDATION: MAE_SCORE= %.4f, RMSE_SCORE= %.4f' % (self.name, float(mae), float(rmse)))
            metrics = (mae, rmse)

        return outputs, metrics

    def verbose(self, epoch, i, total):
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            epoch, i, total, batch_time=self.batch_time,
            data_time=self.data_time, loss=self.losses)
        )

    def save_checkpoints(self, epoch=0):
        try:
            os.mkdir('checkpoints')
        except FileExistsError:
            pass

        save_model_index = epoch
        path = 'checkpoints/%s_ckpt_%d.pt' % (self.name, save_model_index)

        # torch.save(self.model.state_dict(), path)

    def save_state_dict(self, path='', loss=0., targets={}):
        try:
            os.mkdir('results')
        except FileExistsError:
            pass

        if path == '':
            path = 'results/%s.pt' % self.name
        if len(self.val_loss_list) > 0:
            loss = self.val_loss_list[-1]
        checkpoint = {'state_dict': self.model.cpu().state_dict(),
                      'loss': loss,
                      'val_loss_list': self.val_loss_list,
                      'targets': targets}
        torch.save(checkpoint, path)
