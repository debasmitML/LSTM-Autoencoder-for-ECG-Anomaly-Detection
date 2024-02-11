import torch
import torch.nn as nn

class timeDistributedDense(nn.Module):
    def __init__(self , input_size , seq_len):
        super(timeDistributedDense,self).__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(64,input_size)
        
    def forward(self,x):
        
        x = torch.cat([ self.fc(x[:,i,:]).unsqueeze(1) for i in range(self.seq_len)],1)
        return x
    
     
class lstmAutoEncoder(nn.Module):
    
    def __init__(self , input_size , latent_size , seq_len):
        super(lstmAutoEncoder,self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, 64 , batch_first = True )
        self.lstm2 = nn.LSTM(64 , latent_size, batch_first = True )
        self.lstm3 = nn.LSTM(latent_size, latent_size , batch_first = True )
        self.lstm4 = nn.LSTM(latent_size , 64, batch_first = True )
        self.time_fc = timeDistributedDense(input_size,seq_len)
        self.dropout = nn.Dropout(p=0.2)
        self.seq_len = seq_len
        self.latent_size = latent_size
        
    def forward(self , x):
        
        x,(_,_) = self.lstm1(x)
        x= self.dropout(x)
        x,(_,_) = self.lstm2(x)
        x= self.dropout(x)
        x = x[:,-1,:]
        x = x.repeat(1,self.seq_len).reshape(x.shape[0], self.seq_len , self.latent_size)
        x,(_,_) = self.lstm3(x)
        x= self.dropout(x)
        x,(_,_) = self.lstm4(x)
        x= self.dropout(x)
        x = self.time_fc(x)
        
        return x 
