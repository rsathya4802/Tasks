
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

from dataset2 import TrainData

from tqdm import tqdm

from sklearn.metrics import confusion_matrix,classification_report

from model import ResNet,InterMediateBlock
from helper import check_accuracy_probs_preds_loss,save_chkpt,load_chkpoint,ret_preds_labels

print('Import Done')


# Torch SummaryWriter to store generated plots and calculated accuracies

writer = SummaryWriter('runs/part2_custom_mnist')


# Initialising Torch Device so that if there exists a GPU it can be utilised

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


print("Device: ",device)




# Preparing Dataset

'''
    Original Images was first rotated by 30 degrees randomly both clockwise and anti-clockwise to introduce some variation
    And Resize was applied such that aspect ratio is maintained
    Center Crop was done to focus more on the Image
    Then Again Resize was applied to match image dimensions with that of standard MNIST ( 28 X 28 )
    
'''

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomRotation((-30,30)),
        transforms.Resize((300,400)),
        transforms.CenterCrop((280,280)),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
    ]
)

'''
 0-9 Images from the dataset used for Part-1 are taken out and trained
 Out of 400 (40*10) Images available, 342 were used for Training and 58 for Validation
 Batch Size of 32 was used
 
'''
train_root = './trainPart1/train'

dataset = TrainData(root=train_root ,transform=transform)
train_data,val_data = random_split(dataset,[342,58])
train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
val_loader = DataLoader(val_data,batch_size=32,shuffle=True)




print('Dataset Ready !! ')

# Class labels

num_classes = 10
classes = [str(i) for i in range(num_classes)]




# To create model
def create_model(img_channel=1, layers = [1,1,1,1], num_classes=10):
    return ResNet(InterMediateBlock, layers, img_channel, num_classes)

# Configurations

learning_rate = 3e-4
num_epochs = 80
model = create_model(img_channel=1,layers=[1,2,2,2],num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.9)

  
def train():
    tot_train_loss = []
    tot_train_acc = []
    tot_val_loss = []
    tot_val_acc = []
    best_acc = -10

    # Train Network
    for epoch in range(num_epochs):
        losses = []
        loop = tqdm(train_loader)
        num_correct = 0
        num_samples = 0

        for batch_idx, (data, targets) in enumerate(loop):
            # Get data to cuda if possible

            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

            # epoch_loss+= loss.item()

            losses.append(loss.item())
            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
            loop.set_postfix(loss=loss.item(), train_acc=(num_correct * 100 / num_samples).item())

        val_acc, val_probs, val_preds, val_loss = check_accuracy_probs_preds_loss(val_loader, criterion, model, 'val',
                                                                                  epoch)
        train_acc = num_correct / num_samples
        print('Val_acc: {:0.2f} Val_loss: {:0.2f}'.format(val_acc * 100, val_loss))

        writer.add_scalar('Loss/Train', torch.tensor(losses).mean(), epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Acc/Train', train_acc, epoch)
        writer.add_scalar('Acc/Val', val_acc, epoch)

        tot_train_loss.append(torch.tensor(losses).mean())
        tot_train_acc.append(train_acc)
        tot_val_loss.append(val_loss)
        tot_val_acc.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            chkpt = {
                'state_dict': model.state_dict(),
                'val_acc': best_acc,
                'train_acc': train_acc
            }
            save_chkpt(chkpt, 'pretrained_custom_mnist.pth.tar')


if __name__ == '__main__':
    
    train()
    print('Training Done !!')










