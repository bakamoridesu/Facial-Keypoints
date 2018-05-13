## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        c1_size, c2_size, c3_size, c4_size = 6,20,32,64
        fc1_size, fc2_size, fc3_size = 1024,512,256
        output_size = 136
        dropout_conv = 0.2
        dropout_fc = 0.4
        
        self.conv1 = nn.Conv2d(1, c1_size, 2)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(c1_size, c2_size, 3)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(c2_size, c3_size, 5)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.conv4 = nn.Conv2d(c3_size, c4_size, 5)
        nn.init.xavier_uniform_(self.conv4.weight)
        
        self.bn1 = nn.BatchNorm2d(c1_size)
        self.bn2 = nn.BatchNorm2d(c2_size)
        self.bn3 = nn.BatchNorm2d(c3_size)
        self.bn4 = nn.BatchNorm2d(c4_size)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=dropout_conv)
        self.fc1 = nn.Linear(10*10*c4_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.output = nn.Linear(fc3_size, output_size)
        self.fc_drop = nn.Dropout(p=dropout_fc)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(self.pool(F.relu(self.bn3(self.conv3(x)))))
        x = self.dropout(self.pool(F.relu(self.bn4(self.conv4(x)))))
        
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc_drop(x)
        x = self.fc2(x)
        x = self.fc_drop(x)
        x = self.fc3(x)
        x = self.fc_drop(x)
        
        x = self.output(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
