'''

ResNet model was used. 

Paper Link: https://arxiv.org/abs/1512.03385

A modified version of ResNet50 was used.
The four intermediate blocks in the original version consist of 3,4,4,4 layers respectively.
Whereas here intermediate blocks consist of 1,2,2,2 layers.
The dataset used here is not as complex as the images used in ImageNet challenge so I reduced no. of layers.

'''


import torch
import torch.nn as nn




'''
  Intermediate Block has identity connections and forms the core blocks of ResNets
'''

class InterMediateBlock(nn.Module):
    
    def __init__(self,in_channels,inter_channels,identity_connection=None,stride=1):
        
        '''
            in_channels = No.of Channels in Input Images
            inter_channels = No.of Channels that the intermediate layers in each block will have
            identity_connection =  Identity Connection if needed for that block is passed as an argument
        '''
        
        super(InterMediateBlock,self).__init__()
        self.expansion = 4   # Expansion refers to how the channel size should change with respect to the input channel
        
        # 1st layer
        self.conv1 = nn.Conv2d(
            in_channels,
            inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # 2nd layer
        self.conv2 = nn.Conv2d(
            inter_channels,
            inter_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        
        # 3rd layer
        self.conv3 = nn.Conv2d(
            inter_channels,
            inter_channels*self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.bn3 = nn.BatchNorm2d(inter_channels*self.expansion)
        
        self.relu = nn.ReLU()
        
        # Identity Connection
        self.identity_connection = identity_connection
        
        # self.initialize_weights()
        
    
    def forward(self,x):
        
        id_connection = x.clone()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_connection:
            
            id_connection = self.identity_connection(id_connection)
            
        x += id_connection
        
        x = self.relu(x)
        return x      

      

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)



                
                
"""
  The complete ResNet model
"""

class ResNet(nn.Module):
    
    def __init__(self, block, no_layers, channels, num_classes):
        
        '''
            block - A class which is used for each block in the ResNet
            no_layers- A list of values which indicate how many layers each block will have.There are four such blocks
                       with many layers in each 
            channels - No.of Channels in Input Image
            num_classes - No.of Classes present in dataset to train the model for
        '''
        
        super(ResNet,self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(channels,64,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.layer1 = self.layer(InterMediateBlock,no_layers[0],64,1)
        self.layer2 = self.layer(InterMediateBlock,no_layers[1],128,2) 
        self.layer3 = self.layer(InterMediateBlock,no_layers[2],256,2)
        self.layer4 = self.layer(InterMediateBlock,no_layers[3],512,2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc =  nn.Linear(2048,num_classes)
        self.softmax = nn.Softmax(dim=1)
        # self.initialize_weights()
        
    def layer(self,block, no_layer_blocks, out_channels, stride):
        
        identity_connection = None
        layers = []
        
# Whenever there is a change in size between Output Feature Space and Input Feature Space an Identity Connection is needed
# else Input Feature space can be directly added to the Ouput Feature Space

        if stride!=1 or self.in_channels!= out_channels*4: 
            
            identity_connection = nn.Sequential(
                    nn.Conv2d(self.in_channels,out_channels*4,kernel_size=1,stride=stride),
                    nn.BatchNorm2d(out_channels*4),
            )
        layers.append(block(self.in_channels,out_channels,identity_connection,stride=stride))
        self.in_channels = out_channels*4
        
        
        for i in range(no_layer_blocks):
            layers.append(block(self.in_channels,out_channels))
        
        return nn.Sequential(*layers)
        
        
    def forward(self,x):
    
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        # x = self.softmax(x)
        
        return x


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                         