import os
import cv2 as cv
import torch
from torch.utils.data import Dataset,DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import  make_grid


class TrainData(Dataset):

    def __init__(self,root='./trainPart1/train',transform=None):
        self.root = root
        self.transform = transform
        final_files = []
        for idx,directory in enumerate(sorted(os.listdir(root))): # Loop through all directories after sorting them. By this the order would be [0,1,2,....,9]
            dir_path = os.path.join(root,directory)
            for x in sorted(os.listdir(dir_path)):
                if os.path.isfile(os.path.join(dir_path, x)):
                    final_files.append((os.path.join(dir_path, x),idx))
        self.final_files = final_files

    def __len__(self):
        return len(self.final_files)

    def __getitem__(self, index):

        img_path = self.final_files[index][0]
        img = cv.imread(img_path,0)
        y_label = torch.tensor(self.final_files[index][1])

        if self.transform:
            img = self.transform(img)

        return (img,y_label)

# dataset = TrainData('./mnistTask3/mnistTask')
# train_loader = DataLoader(dataset,batch_size=100,shuffle=True)
#
# res = []
# for idx,(x,y) in enumerate(train_loader):
#     # print(len(x))
#     if idx<10:
#         for idx,l in enumerate(y):
#             # print(l)
#             if l==2:
#                 # print('YES')
#                 # print(x[idx].shape,res.shape)
#                 # res = torch.cat((x[idx].unsqueeze(0).unsqueeze(1),res),dim=0)
#                 res.append(x[idx].unsqueeze(0))
#                 # plt.imshow(x[idx],cmap='gray')
#                 # plt.xlabel(str(l))
#                 # # plt.title()
#                 # plt.show()
#     else:
#         break
# # #
# #
#
# plt.imshow(make_grid(torch.stack(res)).permute(1,2,0))
# plt.show()
