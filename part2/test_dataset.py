import os
import cv2 as cv
import torch
from torch.utils.data import Dataset,DataLoader
from matplotlib import pyplot as plt

# final_files = []
# root = './trainPart1/train'
# # for idx,(root,sub_dirs,files) in enumerate(os.walk('./trainPart1/train',topdown=True)):
# for idx,directory in enumerate(sorted(os.listdir(root))):
#
#     # if idx!=0:
#
#         # print(idx,root)
#         # print(sub_dirs, len(files))
#     dir_path = os.path.join(root,directory)
#     for x in sorted(os.listdir(dir_path)):
#         if os.path.isfile(os.path.join(dir_path, x)):
#             final_files.append((os.path.join(dir_path, x),idx))
#
#         # print(files)
#     print(len(final_files))
#     print('***************')
#
# for x in final_files:
#     if x[1]==0:
#         print(x[0])

class TestData(Dataset):

    def __init__(self,root='./trainPart1/train',transform=None):
        self.root = root
        self.transform = transform
        final_files = []
        for x in sorted(os.listdir(root)):
            if os.path.isfile(os.path.join(root, x)):
                final_files.append(os.path.join(root, x))
        self.final_files = final_files

    def __len__(self):
        return len(self.final_files)

    def __getitem__(self, index):

        img_path = self.final_files[index]
        img = cv.imread(img_path,0)

        if self.transform:
            img = self.transform(img)

        return img

# dataset = TestData('./test_folder')
# test_loader = DataLoader(dataset,batch_size=32,shuffle=True)
# for x in test_loader:
#     # print(len(x))
#     plt.imshow(x[0].squeeze(0),cmap='gray')
#     # plt.title()
#     plt.show()
#     break
#     # break

#
