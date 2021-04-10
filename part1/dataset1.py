import os
import cv2 as cv
import torch
from torch.utils.data import Dataset,DataLoader
from matplotlib import pyplot as plt


class TrainData(Dataset):

    def __init__(self,root='./trainPart1/train',transform=None):
        self.root = root
        self.transform = transform
        final_files = []
        for idx,directory in enumerate(sorted(os.listdir(root))):
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

# dataset = TrainData()
# train_loader = DataLoader(dataset,batch_size=32,shuffle=True)
# for (x,y) in train_loader:
#     # print(len(x))
#     for idx,l in enumerate(y):
#         # print(l)
#         if l==8:
#             print('YES')
#             plt.imshow(x[idx],cmap='gray')
#             plt.xlabel(str(l))
#             # plt.title()
#             plt.show()
#         # break
#     # break
#
#
