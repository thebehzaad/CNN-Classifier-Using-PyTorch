"""*******************************************************************************

                               CNN Trainer and Tester

*******************************************************************************"""
#%% Importing Libraries

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

#%% CNN Trainer

class CNNTrainer(nn.Module):
    def __init__(self, model, train_dataloader, valid_dataloader, config):
        super(CNNTrainer, self).__init__()
        self.model = model.to(config['device'])
        self.config = config
        self.train_dl = train_dataloader
        self.valid_dl = valid_dataloader
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.epochs = 0
        self.max_epochs = config['num_of_training_epochs']

    def train_epochs(self):
        start_time = time.time()
        for e in range(self.max_epochs):
            epoch_loss = self.train()
            epoch_val_loss = self.val()
            print('Validation Loss: %f' % epoch_val_loss)
        print('Total Training Mins: %5.2f' % ((time.time()-start_time)/60))

    def train(self):
        self.model.train()
        epoch_loss = 0
        for batch_num, (images, labels) in enumerate(self.train_dl):
            print("Train_batch: %d" % (batch_num + 1))
            loss = self.train_batch(images, labels)
            epoch_loss += loss
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1
        # LOG EPOCH LEVEL INFORMATION
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f' % (self.epochs, self.max_epochs, epoch_loss))
        return epoch_loss

    def train_batch(self, images, labels):
        self.optimizer.zero_grad()
        pred_probs = self.model(images)
        labels=labels.long()
        loss = self.loss(pred_probs, labels)
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), self.config['gradient_clipping'])
        self.optimizer.step()
        return float(loss)

    def val(self):
        self.model.eval()
        with torch.no_grad():
            hold_out_loss = 0
            for batch_num, (images, labels) in enumerate(self.valid_dl):
                pred_probs= self.model(images)
                labels=labels.long()
                hold_out_loss += self.loss(pred_probs, labels)
            hold_out_loss = hold_out_loss / (batch_num + 1)
        return float(hold_out_loss)

class CNNTester(nn.Module):
    def __init__(self, model, test_dataloader, config):
        super(CNNTester, self).__init__()
        self.model = model.to(config['device'])
        self.config = config
        self.test_dl = test_dataloader

    def val(self):
        self.model.eval()
        n_samples=0
        n_correct=0
        with torch.no_grad():
            for images, labels in self.test_dl:
                pred_probs= self.model(images)
                _,pred_labels=torch.max(pred_probs,1)
                n_samples+=labels.size(0)
                n_correct+=(pred_labels==labels).sum().item()
        accuracy_score=n_correct/n_samples
        return float(accuracy_score)
