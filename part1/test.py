
# Importing Libraries that are necessary

import cv2 as cv

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim

from matplotlib import pyplot as plt

from test_dataset import TestData

from tqdm import tqdm

from sklearn.metrics import confusion_matrix,classification_report

from model import ResNet,InterMediateBlock
from helper import check_accuracy_probs_preds_loss,save_chkpt,load_chkpoint,ret_preds_labels


# Initialising Torch Device so that if there exists a GPU it can be utilised

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




# Preparing Dataset

'''
    Original Images was first rotated by 30 degrees randomly both clockwise and anti-clockwise to introduce some variation
    And Resize was applied such that aspect ratio is maintained
    Center Crop was done to focus more on the Image
    
'''


transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((300,400)),
        transforms.CenterCrop((210,280)),
        transforms.ToTensor(),
    ]
)

'''
 
 Out of 2480 Images available, 2200 were used for Training and 280 for Validation
 Batch Size of 32 was used
 
'''
test_root = './test_folder'
dataset = TestData(root=test_root ,transform=transform)
test_loader = DataLoader(dataset,batch_size=1,shuffle=True)


# Class labels

num_classes = 62
classes = [str(i) for i in range(num_classes)]




# To create model
def create_model(img_channel=1, layers = [1,1,1,1], num_classes=10):
    return ResNet(InterMediateBlock, layers, img_channel, num_classes)




model = create_model(img_channel=1,layers=[1,2,2,2], num_classes=62).to(device)

load_chkpoint(torch.load('./part1_model_tb/part1_trained_model.pth.tar',map_location=device),model)

criterion = nn.CrossEntropyLoss()

class_preds = []

model.eval()

with torch.no_grad():
    for x in test_loader:
        x = x.to(device=device)

        scores = model(x)

        _, class_preds_batch = torch.max(scores, 1)

        class_preds.append(class_preds_batch)

test_preds = torch.cat(class_preds)

model.train()

for idx,x in enumerate(test_loader):

    plt.imshow(x.squeeze(0).squeeze(0),cmap='gray')
    plt.title(f' Class Label: {int(classes[test_preds[idx]])}')
    plt.show()





