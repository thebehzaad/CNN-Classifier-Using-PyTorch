"""*******************************************************************************

                                    CNN Model

*******************************************************************************"""
#%% Importing Libraries

import torch
import torch.nn as nn

#%% CNN Model

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_l1 = nn.Conv2d(3, 32, 3) 
        self.conv_l2 = nn.Conv2d(32, 64, 3)
        self.conv_l3 = nn.Conv2d(64, 128, 3)
        self.conv_l4 = nn.Conv2d(128, 128, 3)
        self.activ = nn.ReLU()
        self.dropout=nn.Dropout(p=0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 23, 64)
        self.fc2 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # -> (bs, depth, height, width)=(bs, 3, 300, 400)
        x = self.pool(self.activ(self.conv_l1(x)))  # conv_l1-> (bs, 32, 298, 398) pooling-> (bs, 32, 149, 199)
        x = self.pool(self.activ(self.conv_l2(x)))  # conv_l2-> (bs, 64, 147, 197) pooling-> (bs, 64, 73, 98)
        x = self.pool(self.activ(self.conv_l3(x)))  # conv_l3-> (bs, 128, 71, 96)  pooling-> (bs, 128, 35, 48)
        x = self.pool(self.activ(self.conv_l4(x)))  # conv_l4-> (bs, 128, 33, 46)  pooling-> (bs, 128, 16, 23)
        x = x.view(-1, 128 * 16 * 23)               # flatten -> (bs, 128*16*32)
        x = self.activ(self.fc1(x))                 # fc1-> (bs, 64)
        x = self.dropout(x)                         # Drop out-> (bs, 64)
        x = self.fc2(x)                             # fc2-> (bs, 2)
        x = self.softmax(x)                         # Softmax-> (bs, 2)
        return x
    
    
if __name__ == '__main__':
    
    x=torch.rand(2,3,300,400, dtype=torch.float32)
    model = ConvNet()
    out = model(x)