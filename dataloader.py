import pandas as pd
from sklearn.preprocessing import MinMaxScaler , StandardScaler
import torch
from torch.utils.data import Dataset , DataLoader
import matplotlib.pyplot as plt
import numpy as np


class anomalyDataset(Dataset):
    def __init__(self, data , transform , transform_type , feature_num):
        
        self.data = data
        self.transform = transform
        self.normalize = MinMaxScaler(feature_range = (0,1))
        self.standardize = StandardScaler()
        self.transform_type = transform_type
        self.feature_num = feature_num
     
     
    def normalization(self , raw_data):   
        
        if self.transform:
            
            if self.transform_type == 'normalization':
            
                self.features = self.normalize.fit_transform(raw_data.reshape(-1,self.feature_num))
                
            else:
                
                self.features = self.standardize.fit_transform(raw_data.reshape(-1,self.feature_num)) 
                
        else:  
                
            self.features = raw_data.reshape(-1,self.feature_num)
            
        return self.features
        
    
    def __len__(self):
        
        return self.data.__len__() 
        
    
    def __getitem__(self, time_index):
        
        seq_data = self.data[time_index , :]
        transformed_data = self.normalization(seq_data)
        
        return torch.from_numpy(transformed_data).float()
    
