from torch.utils.data import SequentialSampler
from torchvision import transforms
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

class FUNDUSDataset(Dataset):
    def __init__(self, data_dir, transform=transforms.ToTensor()):
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform
    def __getitem__(self, index):
        path_img= self.data_info[index]
        img = Image.open(path_img)
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.data_info)
    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            img_names = os.listdir(os.path.join(root))
            img_names = list(filter(lambda x: x.endswith('.png'), img_names))
            for i in range(len(img_names)):
                img_name = img_names[i]
                path_img = os.path.join(root, img_name)
                data_info.append(path_img)
        return data_info

class ANGIODataset(Dataset):
    def __init__(self, data_dir, transform=transforms.ToTensor()):
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform
    def __getitem__(self, index):
        path_img= self.data_info[index]
        img = Image.open(path_img)
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.data_info)
    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            img_names = os.listdir(os.path.join(root))
            img_names = list(filter(lambda x: x.endswith('.png'), img_names))
            for i in range(len(img_names)):
                img_name = img_names[i]
                path_img = os.path.join(root, img_name)
                data_info.append(path_img)
        return data_info

transform_list=[transforms.Resize([256,256]),
                transforms.ToTensor()]
trans = transforms.Compose(transform_list)

fundus_data = FUNDUSDataset(data_dir='E:\PycharmProjects\mynewGAN\data\\nor\Images')
test_sampler = SequentialSampler(fundus_data)
fundus_loader = iter(DataLoader(dataset=fundus_data, batch_size=1, shuffle=False,sampler=test_sampler))

fundus_data_resize = FUNDUSDataset(data_dir='E:\PycharmProjects\mynewGAN\data\\nor\Images',transform=trans)
test_sampler = SequentialSampler(fundus_data)
fundus_loader_resize = iter(DataLoader(dataset=fundus_data_resize, batch_size=1, shuffle=False,sampler=test_sampler))

angio_data = ANGIODataset(data_dir='E:\PycharmProjects\mynewGAN\data\\nor\Masks')
test_sampler = SequentialSampler(angio_data)
angio_loader = iter(DataLoader(dataset=angio_data, batch_size=1, shuffle=False,sampler=test_sampler))

angio_data_resize = ANGIODataset(data_dir='E:\PycharmProjects\mynewGAN\data\\nor\Masks',transform=trans)
test_sampler = SequentialSampler(angio_data)
angio_loader_resize = iter(DataLoader(dataset=angio_data_resize, batch_size=1, shuffle=False,sampler=test_sampler))
#======================================================================================================
ab_fundus_data = FUNDUSDataset(data_dir='E:\PycharmProjects\mynewGAN\data\\ab\Images')
test_sampler = SequentialSampler(ab_fundus_data)
ab_fundus_loader = iter(DataLoader(dataset=ab_fundus_data, batch_size=1, shuffle=False,sampler=test_sampler))

ab_fundus_data_resize = FUNDUSDataset(data_dir='E:\PycharmProjects\mynewGAN\data\\ab\Images',transform=trans)
test_sampler = SequentialSampler(ab_fundus_data)
ab_fundus_loader_resize = iter(DataLoader(dataset=ab_fundus_data_resize, batch_size=1, shuffle=False,sampler=test_sampler))

ab_angio_data = ANGIODataset(data_dir='E:\PycharmProjects\mynewGAN\data\\ab\Masks')
test_sampler = SequentialSampler(ab_angio_data)
ab_angio_loader = iter(DataLoader(dataset=ab_angio_data, batch_size=1, shuffle=False,sampler=test_sampler))

ab_angio_data_resize = ANGIODataset(data_dir='E:\PycharmProjects\mynewGAN\data\\ab\Masks',transform=trans)
test_sampler = SequentialSampler(ab_angio_data)
ab_angio_loader_resize = iter(DataLoader(dataset=ab_angio_data_resize, batch_size=1, shuffle=False,sampler=test_sampler))
