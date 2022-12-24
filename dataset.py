import os
import numpy as np
import json

class DatasetLoader:
    def __init__(self, path_dataset="../archivos_numpy/generos/"):
        self.path_dataset = path_dataset

    def shuffle(self, data, labels):
        data = np.array(data)
        labels = np.array(labels)
        p = np.random.permutation(len(data))
        return data[p], labels[p]

    def prepare_dataset(self, validation_split=0.2, shuffle=True, sequence_length=768):
        data=[]
        labels=[]
        label_count=0
        for genero in os.listdir(self.path_dataset):
            genero_data=np.load(self.path_dataset+genero, allow_pickle=True)
            # convert to dictionary
            genero_data=genero_data.item()
            for key in genero_data:
                data.append(genero_data[key])
                labels.append(label_count)
            label_count+=1
        # pad data
        for i in range(len(data)):
            if len(data[i])<sequence_length:
                data[i]+=[0]*(sequence_length-len(data[i]))
            else:
                data[i]=data[i][:sequence_length]
        one_hot_labels=np.zeros((len(labels),label_count))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]]=1
        if shuffle:
            data,one_hot_labels=self.shuffle(data,one_hot_labels)
        split_index=int(len(data)*(1-validation_split))
        train_data=data[:split_index]
        train_labels=one_hot_labels[:split_index]
        validation_data=data[split_index:]
        validation_labels=one_hot_labels[split_index:]
        return train_data,train_labels,validation_data,validation_labels