import json
import os
from Config import Config as cfg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn as nn

def save_json(file_name , hist):
    
    with open(os.path.join(cfg.DIRECTORY_SAVE_RESULT,file_name),"w") as f:
        json.dump(hist , f)
        
        
def load_json(file_name):
    
    with open(os.path.join(cfg.DIRECTORY_SAVE_RESULT,file_name),"r") as f_read:
        data = json.load(f_read)
    
    return data

def loss_plot(train_loss , val_loss):
    
    epoch = list(train_loss.keys())
    train_loss_vals = list(train_loss.values())
    val_loss_vals = list(val_loss.values())
    plt.plot(epoch, train_loss_vals, 'y', label='Training loss')
    plt.plot(epoch, val_loss_vals, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(cfg.DIRECTORY_SAVE_RESULT,'Training_Validation_loss_plot.png'))
    plt.show()
    
def threshold_plot():
    
    train_hist = load_json('train.json')
    train_loss_values = list(train_hist.values())
    sns.displot(train_loss_values,bins=30,kde=True)
    plt.xlabel('Train Loss')
    plt.ylabel('No. of Examples')
    plt.show()
    threshold = np.mean(np.array(train_loss_values))+np.std(np.array(train_loss_values))
    return threshold

def reconstruction_plot(batch_actual , batch_predict , thresh):
    os.makedirs(cfg.DIRECTORY_SAVE_RESULT , exist_ok=True)
    criterion = nn.L1Loss()
    fig , axes = plt.subplots(1 , batch_actual.shape[0] ,figsize=(20, 5))
    
    for i in range(batch_predict.shape[0]):
        
        test_loss = criterion(batch_predict[i,:],batch_actual[i,:]).item()
        label = f'anomalous ecg signal [loss : {test_loss}]' if test_loss > thresh else f'normal ecg signal [loss : {test_loss}]'
        predict_test_array = batch_predict[i, : , -1].cpu().numpy()
        batch_test_array = batch_actual[i, : , -1].cpu().numpy()
        points_x = np.arange(batch_predict.shape[1])
        axes[i].plot(points_x , predict_test_array , label = "actual")
        axes[i].plot(points_x , batch_test_array , label = "predicted")
        axes[i].set_title(label)
        axes[i].legend(loc="upper left")
    plt.tight_layout() 
    fig.savefig(os.path.join(cfg.DIRECTORY_SAVE_RESULT,'test_result.png'),bbox_inches='tight')  
    plt.show()    
    
    
    
