# Code for making a Two hidden layer multi-label classification model
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import recall_score, precision_score, accuracy_score

class MultiLabelTaskHead(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(MultiLabelTaskHead, self).__init__()
        self.fc1 = nn.Linear(input_size, 48)
        self.fc2 = nn.Linear(48, 48)
        self.fc3 = nn.Linear(48, output_size)
        ## Example in case of 5 w analysis output_size = 5 
        self.sigmoid = nn.Sigmoid()
        self.device = device
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
    def predict(self, x):
        x = self.forward(x)
        x = torch.round(x)
        return x
    
    def accuracy(self, prediction, target):
        prediction = torch.round(prediction)
        
        return torch.mean((prediction == target).float())  

    def recall(self, prediction, target):
        prediction = torch.round(prediction)

        tp = torch.sum(torch.logical_and(prediction == 1, target == 1), axis=0)
        fn = torch.sum(torch.logical_and(prediction == 0, target == 1), axis=0)

        recall = tp / (tp + fn)
        overall_recall = torch.mean(recall)

        return overall_recall

    def precision(self, prediction, target):
        prediction = torch.round(prediction)

        tp = torch.sum(torch.logical_and(prediction == 1, target == 1), axis=0)
        fp = torch.sum(torch.logical_and(prediction == 1, target == 0), axis=0)

        precision = tp / (tp + fp)
        overall_precision = torch.mean(precision)

        return overall_precision

