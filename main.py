import os
import glob
import ast
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
from nltk.tokenize import WhitespaceTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, precision_score, accuracy_score
from natsort import natsorted
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from Multilabel_task_head import MultiLabelTaskHead
from singlelabel_task_head import SingleLabelTaskHead
from base_network import base_network
from multi_task import MultiTaskModel

np.random.seed(45)
torch.manual_seed(45)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA on", torch.cuda.get_device_name(device))

warnings.filterwarnings('ignore')

batch_size = 32
epoch = 1000
max_seq_length = 128
input_size = 128
# device = 'cpu'

# Load the data
with open('data/data_Xtrain.json', 'r') as file:
    X_train = np.array(json.load(file))
with open('data/data_Xval.json', 'r') as file:
    Xval = np.array(json.load(file))

y_train = pd.read_csv('data/data_ytrain.csv')
y_train_s = y_train['Intent Of Lie (Gaining Advantage/Gaining Esteem/Avoiding Punishment/Avoiding Embarrassment/Protecting Themselves)']
yval = pd.read_csv('data/data_yval.csv')
yval_s = yval['Intent Of Lie (Gaining Advantage/Gaining Esteem/Avoiding Punishment/Avoiding Embarrassment/Protecting Themselves)']

y_train_m = y_train[['ordered_list_1', 'ordered_list_3', 'ordered_list_4', 'ordered_list_7']].applymap(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
y_val_m = yval[['ordered_list_1', 'ordered_list_3', 'ordered_list_4', 'ordered_list_7']].applymap(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

print(type(y_train_m))

y_train_m1 = y_train_m['ordered_list_1'].apply(np.array).to_numpy()
y_val_m1 = y_val_m['ordered_list_1'].apply(np.array).to_numpy()
y_train_m3 = y_train_m['ordered_list_3'].apply(np.array).to_numpy()
y_val_m3 = y_val_m['ordered_list_3'].apply(np.array).to_numpy()
y_train_m4 = y_train_m['ordered_list_4'].apply(np.array).to_numpy()
y_val_m4 = y_val_m['ordered_list_4'].apply(np.array).to_numpy()
y_train_m7 = y_train_m['ordered_list_7'].apply(np.array).to_numpy()
y_val_m7 = y_val_m['ordered_list_7'].apply(np.array).to_numpy()

# Label Encoding of single label dataset.
le = LabelEncoder()
y_train_s = le.fit_transform(y_train_s)
yval_s = le.transform(yval_s)
y_train_s = np.array(y_train_s)
yval_s = np.array(yval_s)

# Tokenize and pad the data
tokenizer = WhitespaceTokenizer()
tokenized_sentences = [tokenizer.tokenize(
    sentence)[:max_seq_length] for sentence in X_train]
tokenized_sentences_val = [tokenizer.tokenize(
    sentence)[:max_seq_length] for sentence in Xval]
vocab = {token: i+1 for i,
         token in enumerate(set(token for sent in tokenized_sentences for token in sent))}
indexed_sequences = [torch.tensor([vocab.get(token, 0) for token in sent] + [
                                  0] * (max_seq_length - len(sent))) for sent in tokenized_sentences]
indexed_sequences_val = [torch.tensor([vocab.get(token, 0) for token in sent] + [
    0] * (max_seq_length - len(sent))) for sent in tokenized_sentences_val]
padded_sequences = pad_sequence(
    indexed_sequences, batch_first=True, padding_value=0)
pad_sequences_val = pad_sequence(
    indexed_sequences_val, batch_first=True, padding_value=0)

# attention_mask = torch.where(padded_sequences != 0, torch.tensor(1), torch.tensor(0))

X_train = padded_sequences
Xval = pad_sequences_val
y_train_m1 = np.vstack(y_train_m1)
y_train_m3 = np.vstack(y_train_m3)
y_train_m4 = np.vstack(y_train_m4)
y_train_m7 = np.vstack(y_train_m7)

y_val_m1 = np.vstack(y_val_m1)
y_val_m3 = np.vstack(y_val_m3)
y_val_m4 = np.vstack(y_val_m4)
y_val_m7 = np.vstack(y_val_m7)

X_train, y_train_s, y_train_m1, y_train_m3, y_train_m4, y_train_m7 = torch.tensor(X_train).long().to(device), torch.tensor(y_train_s).long().to(device), torch.tensor(
    y_train_m1).long().to(device), torch.tensor(y_train_m3).long().to(device), torch.tensor(y_train_m4).long().to(device), torch.tensor(y_train_m7).long().to(device)

Xval, yval_s, y_val_m1, y_val_m3, y_val_m4, y_val_m7 = torch.tensor(Xval).long().to(device), torch.tensor(yval_s).long().to(device), torch.tensor(
    y_val_m1).long().to(device), torch.tensor(y_val_m3).long().to(device), torch.tensor(y_val_m4).long().to(device), torch.tensor(y_val_m7).long().to(device)

dataset_train=TensorDataset(
    X_train, y_train_s, y_train_m1, y_train_m3, y_train_m4, y_train_m7)
dataloader_train=DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True)

dataset_val=TensorDataset(
    Xval, yval_s, y_val_m1, y_val_m3, y_val_m4, y_val_m7)
dataloader_val=DataLoader(
    dataset_val, batch_size=batch_size, shuffle=True)

task_heads=[SingleLabelTaskHead(input_size=128, output_size=10, device=device).to(device), MultiLabelTaskHead(input_size=128, output_size=5, device=device).to(device), MultiLabelTaskHead(
    input_size=128, output_size=7, device=device).to(device), MultiLabelTaskHead(input_size=128, output_size=5, device=device).to(device), MultiLabelTaskHead(input_size=128, output_size=7, device=device).to(device)]

model=MultiTaskModel(base_network(input_size=7700+1, embedding_size=128,
                       hidden_size=64, num_layers=2, dropout=0.5, bidirectional=True, device=device), task_heads, device=device).to(device)

optimizer=optim.Adam(model.parameters(), lr= 1)
loss_fn=nn.CrossEntropyLoss()
criterion_m=nn.BCEWithLogitsLoss()

def accuracy_multi(prediction, target):
    prediction=torch.round(prediction)
    return torch.mean((prediction == target).float(), dim=0)

def recall_multi(prediction, target):
    prediction = torch.round(prediction)

    tp = torch.sum(torch.logical_and(prediction == 1, target == 1), axis=0)
    fn = torch.sum(torch.logical_and(prediction == 0, target == 1), axis=0)

    recall = tp / (tp + fn)

    return recall

def precision_multi(prediction, target):
    prediction = torch.round(prediction)
    tp = torch.sum(torch.logical_and(prediction == 1, target == 1), axis=0)
    fp = torch.sum(torch.logical_and(prediction == 1, target == 0), axis=0)
    precision = tp / (tp + fp)
    return precision

def train(model, dataloader_train, optimizer, criterion, epoch):
    model.train()
    multi_accuracy=0
    task_outputs_lis = [torch.tensor([]).to(device) for i in range(5)]
    target_lis = [torch.tensor([]).to(device) for i in range(5)]
    loss_lis = []
    for batch_idx, (data, target_s, target_m1, target_m3, target_m4, target_m7) in enumerate(dataloader_train):
        target=[target_s, target_m1, target_m3, target_m4, target_m7]
        optimizer.zero_grad()
        task_outputs=model(data)
        losses=[loss_fn(output, label)
                  for output, label in zip([task_outputs[0]], [target[0]])] + [criterion_m(output, label.float())
                                                                               for output, label in zip(task_outputs[1:], target[1:])]
        loss = sum(losses) 
        loss_lis.append(loss)

        loss.backward()
        optimizer.step()
        task_outputs_lis = [torch.cat((task_outputs_lis[i], task_outputs[i]), dim=0) for i in range(len(task_outputs))]
        target_lis = [torch.cat((target_lis[i], target[i]), dim=0) for i in range(len(target))]
        

    # pdb.set_trace()
    loss = sum(loss_lis)/len(loss_lis)
    task_outputs = task_outputs_lis
    target = target_lis
    multi_accuracy=model.accuracy(task_outputs, target)
    multi_recall=model.recall(task_outputs, target)
    multi_precision=model.precision(task_outputs, target)

    multi_accuracy_label=[accuracy_multi(
        x, y) for x, y in zip(task_outputs[1:], target[1:])]
    
    multi_recall_label=[recall_multi(x, y) for x, y in zip(task_outputs[1:], target[1:])]
    multi_precision_label=[precision_multi(x, y) for x, y in zip(task_outputs[1:], target[1:])]

    # pdb.set_trace()
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(dataloader_train.dataset),
        100. * batch_idx / len(dataloader_train), loss.item()))
    for i in range(len(task_outputs)):
        print(f"Task {i+1} Accuracy: {multi_accuracy[i]}", end="\t")
        print(f"Task {i+1} Recall: {multi_recall[i]}", end="\t")
        print(f"Task {i+1} Precision: {multi_precision[i]}")
        if i > 0:
            print(
                f"Task {i+1} Accuracy Label: {multi_accuracy_label[i-1]}")
            print(
                f"Task {i+1} Recall Label: {multi_recall_label[i-1]}")
            print(
                f"Task {i+1} Precision Label: {multi_precision_label[i-1]}")
            print('----------------------------------------------------------------------')

        else:
            print('----------------------------------------------------------------------')
    print('*'*120)

    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss_fn,
        }, f"saved_model/EXPERIMENT_{experiment_num}/checkpoints/checkpoint_{epoch}_{loss}.pt")

dir_info=natsorted(glob.glob('saved_model/EXPERIMENT_*'))

if len(dir_info) == 0:
    experiment_num=1
else:
    experiment_num=int(dir_info[-1].split('_')[-1]) + 1

if not os.path.isdir('saved_model/EXPERIMENT_{}'.format(experiment_num)):
    os.makedirs('saved_model/EXPERIMENT_{}'.format(experiment_num))
    os.system('cp *.py saved_model/EXPERIMENT_{}'.format(experiment_num))

ckpt_lst=natsorted(
    glob.glob('saved_model/EXPERIMENT_{}/checkpoints/*'.format(experiment_num)))
START_EPOCH=0

if len(ckpt_lst) >= 1:
    ckpt_path=ckpt_lst[-1]
    checkpoint=torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    START_EPOCH=checkpoint['epoch']
    print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
    START_EPOCH += 1
else:
    os.makedirs('saved_model/EXPERIMENT_{}/checkpoints/'.format(experiment_num))


for epoch in range(START_EPOCH, epoch + 1):
    train(model, dataloader_train, optimizer, loss_fn, epoch)
