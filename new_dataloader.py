import torch
from torch.utils.data import SequentialSampler
from torchvision import transforms
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

class ImageFolder(Dataset):
    def __init__(self):
        self.imgpath='E:\PycharmProjects\mynewGAN\data\\nor\Images\\'
        self.gtpath='E:\PycharmProjects\mynewGAN\data\\nor\Masks\\'
        self.transform = transforms.Compose([transforms.Resize([256, 256]),
                          transforms.ToTensor()])
        self.trans = transforms.Compose([transforms.ToTensor()])
        self.images = [self.imgpath + f for f in os.listdir(self.imgpath) if f.endswith('.png')]
        self.gts = [self.gtpath + f for f in os.listdir(self.gtpath) if f.endswith('.png')]

        self.size = len(self.images)

    def __getitem__(self, index):
        fun = self.picloader(self.images[index])
        ang = self.angloader(self.gts[index])

        funda = self.trans(fun)
        angda = self.trans(ang)
        fun_resize = self.transform(fun)
        ang_resize = self.transform(ang)

        return funda,angda,fun_resize,ang_resize

    def __len__(self):
        return self.size

    def picloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    def angloader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
# print(torch.cuda.is_available())
# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dataset=ImageFolder()
# data_loader=DataLoader(dataset,batch_size=2,shuffle=True)
# for i, data in enumerate(data_loader):
#     fun, ang, fun_resize, ang_resize=data
#     fun=fun.cuda()
#     print('fun',fun.shape)
#     print(fun.device)
    # print('ang',ang.shape)
    # print('funre',fun_resize.shape)
    # print('angre',ang_resize.shape)



# fundus_data = FUNDUSDataset(data_dir='E:\PycharmProjects\mynewGAN\data\\nor\Images')
# test_sampler = SequentialSampler(fundus_data)
# fundus_loader = iter(DataLoader(dataset=fundus_data, batch_size=1, shuffle=False,sampler=test_sampler))
#
# fundus_data_resize = FUNDUSDataset(data_dir='E:\PycharmProjects\mynewGAN\data\\nor\Images',transform=trans)
# test_sampler = SequentialSampler(fundus_data)
# fundus_loader_resize = iter(DataLoader(dataset=fundus_data_resize, batch_size=1, shuffle=False,sampler=test_sampler))
#
# angio_data = ANGIODataset(data_dir='E:\PycharmProjects\mynewGAN\data\\nor\Masks')
# test_sampler = SequentialSampler(angio_data)
# angio_loader = iter(DataLoader(dataset=angio_data, batch_size=1, shuffle=False,sampler=test_sampler))
#
# angio_data_resize = ANGIODataset(data_dir='E:\PycharmProjects\mynewGAN\data\\nor\Masks',transform=trans)
# test_sampler = SequentialSampler(angio_data)
# angio_loader_resize = iter(DataLoader(dataset=angio_data_resize, batch_size=1, shuffle=False,sampler=test_sampler))
