'''
    
    Consists of Helper Functions
    
'''


# Returns Accuracy , Predicted Class Probabailties , Predicted Class Predictions , Loss 
# Useful to calculate loss, accuracy , precision and other performance metrics

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def check_accuracy_probs_preds_loss(loader,loss_fn , model,mode='train',global_step=None):

    '''
        
    Inputs:
    
        loader: DataLoader that consists of the data upon which all values have to be calculated
        loss_fn: Loss function used by Model to train
        model: Model used for training
        mode: 'Train' or 'Val' or 'Test' mode based on need
        global_step: For Tensorboard usage
        
    Outputs:
        Accuracy
        Predicted Class Probabailties
        Predicted Class Predictions 
        Loss 
        
    '''
    

    num_correct = 0
    num_samples = 0
    class_probs = []
    class_preds = []
    losses = []

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            loss = loss_fn(scores,y)

            losses.append(loss.item())

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)


            class_probs_batch = [F.softmax(el, dim=0) for el in scores]
            _, class_preds_batch = torch.max(scores, 1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)

    model.train()
    acc = num_correct/num_samples



    return acc, test_probs,test_preds,sum(losses)/len(losses)



# To save checkpoint

def save_chkpt(state,filename='model.pth.tar'):
    
    print('===== Saving Checkpoint =====')
    torch.save(state,filename)

# To load model from checkpoint    
    
def load_chkpoint(chkpt,model):
    
    print('===== Loading Checkpoint =====')
    model.load_state_dict(chkpt['state_dict'])

    
# Return Predictions and Labels for Testing 

def ret_preds_labels(model,loader):

    class_preds = []
    actual_preds = []

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)

            _, class_preds_batch = torch.max(scores, 1)

            class_preds.append(class_preds_batch)
            actual_preds.append(y)

    test_preds = torch.cat(class_preds)
    actual_preds = torch.cat(actual_preds)

    model.train()
    return test_preds,actual_preds


    
    
    