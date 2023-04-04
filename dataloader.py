# import all standard libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
from sklearn.model_selection import train_test_split
import json

# load the pandas dataset
df = pd.read_csv('data.csv')
l = df.columns

# Remove rows with nan values in particular column
df[l[1]] = df[l[1]].dropna()

# Replace Nan values with a string
df[l[1]] = df[l[1]].fillna('Nan')
df[l[3]] = df[l[3]].fillna('Nan2')
df[l[4]] = df[l[4]].fillna('Nan3')
df[l[6]] = df[l[6]].fillna('Nan4')
df[l[7]] = df[l[7]].fillna('Nan5')

df[l[1]] = df[l[1]].str.split('/')
df[l[3]] = df[l[3]].str.split(',')
df[l[4]] = df[l[4]].str.split('/')
df[l[7]] = df[l[7]].str.split('/')

unique_words_1 = list(set(word for row in df[l[1]] for word in row))
unique_words_3 = list(set(word for row in df[l[3]] for word in row))
unique_words_4 = list(set(word for row in df[l[4]] for word in row))
unique_words_7 = list(set(word for row in df[l[7]] for word in row))

def create_ordered_list(words, unique_words):
    ordered_list = [1 if word in words else 0 for word in unique_words]
    return ordered_list


df['ordered_list_1'] = df[l[1]].apply(lambda x: create_ordered_list(x, unique_words_1))
df['ordered_list_3'] = df[l[3]].apply(lambda x: create_ordered_list(x, unique_words_3))
df['ordered_list_4'] = df[l[4]].apply(lambda x: create_ordered_list(x, unique_words_4))
df['ordered_list_7'] = df[l[7]].apply(lambda x: create_ordered_list(x, unique_words_7))

df.to_csv('new_data.csv', index=False)

l = df.columns

# remove unwanted columns
df = df[[l[0], l[8], l[9], l[10], l[6], l[11]]]

# Split the dataset into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    df[l[0]], df.loc[:, df.columns != l[0]], test_size=0.1, random_state=42)

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

# Save the train and validation dataset
os.makedirs('data', exist_ok=True)
y_train.to_csv('data/data_ytrain.csv', index=False)
y_val.to_csv('data/data_yval.csv', index=False)

with open('data/data_Xtrain.json', 'w') as file:
    print(len(X_train.tolist()))
    json.dump(X_train.tolist(), file)
    
with open('data/data_Xval.json', 'w') as file:
    json.dump(X_val.tolist(), file)



