import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader, SequentialSampler
from torchfitter.trainer import Trainer
from torchfitter.utils.data import DataWrapper
from torchfitter.callbacks import (
    EarlyStopping,
    RichProgressBar,
    LearningRateScheduler
)
import Models
import matplotlib.pyplot as plt
import pytorch_warmup as warmup



torch.manual_seed(0)
np.random.seed(0)

class Pipeline:
    """
    Class to ease the running of multiple experiments. Taken from https://github.com/Xylambda/attention/blob/main/utils.py
    """
    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.train_history = None
        self.val_history = None
        self.y_pred = None
        self.dataset_state = "empty"
        self.preds = None
        self.tests = None
        self.n_layers = 1

    def create_model(self, params):
        pass

    def input_data(self,dataset):
       

        X_train, y_train, X_val, y_val, X_test, y_test = dataset

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        
        self.dataset_state = "loaded"

    def check_dataset(self): 
        return self.dataset_state
        
    def train_model(self, params):
        learning_rate = params[6].item()
        n_epochs = int(params[5].item())
        batch_size = int(params[4].item())
        # ---------------------------------------------------------------------
        #              Construct the data loaders for training
        # ---------------------------------------------------------------------
        train_dataset = PanelDataset(
            self.X_train, self.y_train, nglobal = self.model.dglobal, window_len= self.model.T
        )
        val_dataset = PanelDataset(
            self.X_val, self.y_val,nglobal = self.model.dglobal, window_len= self.model.T
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, pin_memory=True
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)              
        # ---------------------------------------------------------------------
        #              Loss, Optimizer, Learning Rate Scheduler
        # ---------------------------------------------------------------------
                  
        loss_fn =  torch.nn.MSELoss()           
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        num_steps = len(train_loader) * n_epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
     
        
        # ---------------------------------------------------------------------
        #              Training Loop
        # ---------------------------------------------------------------------
        
        trainlossvec = []
        vallossvec = []
        
        
        for epoch in range(n_epochs):
            
            ### Training set
            self.model.train(True)
            running_loss = 0 
            count = 0
            for i,data in enumerate(train_loader): 
                count = count+1
                inputs, labels = data
                # Zero gradients 
                optimizer.zero_grad()
                #Compute model predictions
                outputs = self.model(inputs)
                #Compute loss
                loss = loss_fn(outputs, labels)
                #Backward propagation
                loss.backward() 
                optimizer.step()
                running_loss += loss.item()
                with warmup_scheduler.dampening():
                    lr_scheduler.step()
            avg_loss = running_loss/count
            trainlossvec.append(running_loss/count)
    
            ### Validation set
            self.model.train(False)
            with torch.no_grad():
                running_vloss = 0.0
                count = 0
                for i, vdata in enumerate(val_loader):
                    count = count+1
                    vinputs, vlabels = vdata
                    voutputs = self.model(vinputs)
                    vloss = loss_fn(voutputs, vlabels)
                    running_vloss += vloss.item()
                avg_vloss = running_vloss / count
                vallossvec.append(avg_vloss)
                
            ### Print training state
            if epoch % 10 == 0: 
                print('EPOCH {}:'.format(epoch + 1))
                print('training loss: {}'.format(avg_loss))
                print('validation loss: {}'.format(avg_vloss))
    
            ### Update learning_rate
           
    
        # ---------------------------------------------------------------------
        #                   Store training results
        # ---------------------------------------------------------------------
        self.train_history = trainlossvec
        self.val_history = vallossvec
        
    def plot_history(self):
        plt.plot(self.train_history, label = "Training loss")
        plt.plot(self.val_history, label = "Validation loss")
        plt.legend(loc='best')
        plt.show()
   
    def predict(self):   
        ### Returns predictions of the model on the test set
        self.model.train(False)
        with torch.no_grad():
            test_dataset = PanelDataset(
                    self.X_test, self.y_test,nglobal = self.model.dglobal, window_len= self.model.T
            )
                    
            sampler = SequentialSampler(test_dataset)
            test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            pin_memory=True,
            sampler=sampler,
            shuffle=False
            )
            t=0 
            for i,d in enumerate(test_loader):
                if t == 0 :
                    y_pred = self.model(d[0])
                    y_test = d[1]
                else : 
                    y_pred = torch.cat( (y_pred, self.model(d[0])))
                    y_test = torch.cat( (y_test, d[1]))
                t += 1
            
            y_pred = y_pred.squeeze() 
            y_test = y_test.squeeze()
        self.preds = y_pred 
        self.tests = y_test 
        return y_pred, y_test 

    def plot_predict(self,i):
        plt.plot(self.preds[:,i], label = "predicted")
        plt.plot(self.tests[:,i], label = "true value")
        plt.legend(loc='best')
        plt.show()       

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class DTMLPipeline(Pipeline):
    def __init__(self):
        super().__init__()

        

    def create_model(self, params):
        
        if self.dataset_state == "empty": 
            print("No dataset. Use input_data first")
        else : 
            X_train = self.X_train
            model = Models.DTMLModel(
            d = X_train.shape[1],
            T = int(params[3]),
            h = int(params[2]),
            n_features = X_train.shape[2] ,
            beta = params[0], 
            batch_size = int(params[4]),
            nglobal = int(params[1]),
            dropout = 0.2,
            n_layers = int(params[7])
            )
            self.model = model

    

class PanelDataset(torch.utils.data.Dataset):   
    """_summary_
    Custom class to sample from the data for training
    Args:
        torch (_type_): _dataset_
    """
    def __init__(self, X, y, nglobal, window_len=1):
        self.X = X
        self.y = y
        self.window_len = window_len
        self.nglobal = nglobal
    def __len__(self):
        return self.X.__len__() - self.window_len 

    def __getitem__(self, index):
        return (self.X[index:index+self.window_len,:], self.y[index+self.window_len,self.nglobal:])
    
