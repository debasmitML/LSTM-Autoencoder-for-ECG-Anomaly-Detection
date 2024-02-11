import os
import numpy as np
import pandas as pd
from model import lstmAutoEncoder
import torch
import torch.nn as nn
from Config import Config as cfg
import argparse
from torch.utils.data import DataLoader
from dataloader import anomalyDataset
from Config import Config as cfg
import matplotlib.pyplot as plt
from utils import reconstruction_plot

def arguments():
    parser = argparse.ArgumentParser(description = "Define dynamic parameters for the model.")
    parser.add_argument('--normalization' , action= 'store_true', help = 'normalization')
    parser.add_argument('--transform_type' , default = 'normalization' , type = str , help = 'define the type of transformation')
    parser.add_argument("--laten_dims", default=32, type = int , help="cuda device, i.e. 0 or cpu")
    parser.add_argument("--feature_num", default=1, type = int , help="no. of features")
    parser.add_argument('--batch_size' , default = 3 , type = int , help = 'define batch size')
    parser.add_argument("--device", default="cuda:0", help="cuda device, i.e. 0 or cpu")
    parser.add_argument("--weight", default="./weight/best_model.pt",type = str , help="weight name")
    parser.add_argument("--threshold", default=0.385,type = float , help="threshold value for anomaly detection") ### use threshold_plot() 
    ##function from utils.py and define threshold for anomaly detection
    
    return parser.parse_args() 

args = arguments()


def infer():
    raw_data = pd.read_csv(cfg.INPUT_PATH)
    anomalous_data = np.array(raw_data.iloc[:,:-1][raw_data.iloc[:,-1]==0.0])


    test_loader = DataLoader(dataset = anomalyDataset(anomalous_data , args.normalization, args.transform_type , args.feature_num), batch_size = args.batch_size , shuffle = False) 

    predict_model = lstmAutoEncoder(args.feature_num,args.laten_dims, anomalous_data.shape[1]).to(args.device)
    predict_model.load_state_dict(torch.load(args.weight))


    predict_model.eval()
    with torch.no_grad():
        batch_test = next(iter(test_loader)).to(args.device)
        predict_test = predict_model(batch_test).to(args.device)
        
    reconstruction_plot(batch_test , predict_test , args.threshold)
        
if __name__ == '__main__':
    infer()
    