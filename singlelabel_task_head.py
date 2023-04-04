# In this case we shall use the same model as the one used in the previous task
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from sklearn.metrics import recall_score, precision_score, accuracy_score


class SingleLabelTaskHead(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(SingleLabelTaskHead, self).__init__()
        self.fc1 = nn.Linear(input_size, 48)
        self.fc2 = nn.Linear(48, 48)
        self.fc3 = nn.Linear(48, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        x = torch.argmax(x, dim=1)
        return x
    
    def accuracy(self, prediction, target):
        prediction = torch.argmax(prediction, dim=1)
        
        # ## save target and prediction to a txt file
        # with open('target.txt', 'w') as f:
        #     f.write(str(target.cpu().detach().numpy()))
        # with open('prediction.txt', 'w') as f:
        #     f.write(str(predictinumpy()))on.cpu().detach().

        print(accuracy_score(target.cpu().detach().numpy(), prediction.cpu().detach().numpy()))
        return accuracy_score(target.cpu().detach().numpy(), prediction.cpu().detach().numpy())
    
    def recall(self, prediction, target):
        prediction = torch.argmax(prediction, dim=1)
        print(recall_score(target.cpu().detach().numpy(), prediction.cpu().detach().numpy(), average='micro'))
        return recall_score(target.cpu().detach().numpy(), prediction.cpu().detach().numpy(), average='micro')
    
    def precision(self, prediction, target):
        prediction = torch.argmax(prediction, dim=1)
        print(precision_score(target.cpu().detach().numpy(), prediction.cpu().detach().numpy(), average='micro'))
        return precision_score(target.cpu().detach().numpy(), prediction.cpu().detach().numpy(), average='micro')

