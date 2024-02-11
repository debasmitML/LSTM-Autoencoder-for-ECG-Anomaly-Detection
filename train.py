import argparse
import os
import glob
import numpy as np
import pandas as pd
from dataloader import anomalyDataset
from model import lstmAutoEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Config import Config as cfg
from utils import save_json , loss_plot



def arguments():
    parser = argparse.ArgumentParser(description = "Define dynamic parameters for the model.")
    parser.add_argument('--epochs' , default = 10 , type = int , help = 'number of epochs')
    parser.add_argument('--batch_size' , default = 1 , type = int , help = 'define batch size')
    # parser.add_argument('--seq_len' , default = 7 , type = int , help = 'define sequence length of the data')
    parser.add_argument('--normalization' , action= 'store_true', help = 'normalization')
    parser.add_argument('--transform_type' , default = 'normalization' , type = str , help = 'define the type of transformation')
    parser.add_argument("--laten_dims", default=32, type = int , help="cuda device, i.e. 0 or cpu")
    parser.add_argument("--feature_num", default=1, type = int , help="no. of features")
    parser.add_argument("--learning_rate", default=0.001, type = float , help="learning rate")
    parser.add_argument('--test_ratio' , default = 0.2 , type = float , help = "define_test_ratio")
    parser.add_argument("--device", default="cuda:0", help="cuda device, i.e. 0 or cpu")
    return parser.parse_args() 

args = arguments()

def run():
    raw_data = pd.read_csv(cfg.INPUT_PATH)
    normal_data = np.array(raw_data.iloc[:,:-1][raw_data.iloc[:,-1]==1.0])

    train_data , val_data = train_test_split(normal_data , test_size = args.test_ratio)

    train_loader = DataLoader(dataset = anomalyDataset(train_data , args.normalization, args.transform_type , args.feature_num), batch_size = args.batch_size , shuffle = True) 
    val_loader = DataLoader(dataset = anomalyDataset(val_data , args.normalization, args.transform_type , args.feature_num), batch_size = args.batch_size , shuffle = False) 



    device = torch.device(args.device)
    model = lstmAutoEncoder(args.feature_num,args.laten_dims, normal_data.shape[1]).to(args.device)
    criterion = nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(),lr = args.learning_rate)

    ## create folders
    os.makedirs(cfg.MODEL_WEIGHT_PATH , exist_ok=True)
    os.makedirs(cfg.DIRECTORY_SAVE_RESULT , exist_ok=True)

    ## training loop

    best_val_loss = 1000
    train_dict = {}
    val_dict = {}
    for epoch in range(args.epochs):
        model.train(True)
        total_loss_train = 0.0
        last_loss_train = 0.0
        total_loss_val = 0.0 
        
        for idx,batch_train in enumerate(train_loader):
            batch_train = batch_train.to(args.device)
            batch_prediction = model(batch_train)
            opt.zero_grad()
            loss = criterion(batch_prediction, batch_train)
            loss.backward()
            opt.step()
            total_loss_train += loss.item()
            
            if idx % 100 == 99:
                last_loss_train = total_loss_train / 100
                total_loss_train = 0.0
                print('batch {} loss: {}'.format(idx + 1, last_loss_train))
            
            
        model.eval()
        with torch.no_grad():
            for idx_val , batch_val in enumerate(val_loader):
                batch_val = batch_val.to(args.device)
                batch_prediction_val = model(batch_val)
                loss_val = criterion(batch_prediction_val, batch_val)
                total_loss_val += loss_val.item()
                
                
                
        val_loss_per_epoch = total_loss_val / len(val_loader)    
        if val_loss_per_epoch < best_val_loss:
            best_val_loss = val_loss_per_epoch
            torch.save(model.state_dict() , os.path.join(cfg.MODEL_WEIGHT_PATH , 'best_model.pt'))
            
            
                
        train_dict[epoch + 1] = last_loss_train
        val_dict[epoch + 1] = val_loss_per_epoch    
        print(f'Epochs: {epoch + 1} | Train Loss: {last_loss_train: .3f} | Val Loss: {val_loss_per_epoch : .3f}' )
        
    ## save results
    
    save_json('train.json' , train_dict)
    save_json('val.json' , val_dict)
    
    ## plot results
    loss_plot(train_dict, val_dict)
        
        
        
if __name__ == '__main__':
    
    run()
        
    
    

    