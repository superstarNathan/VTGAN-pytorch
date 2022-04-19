from PIL import Image
import numpy as np
import os
import random
import argparse
from tqdm import tqdm

def random_crop(img, mask, height, width, num_of_crops,name,cla,dir_name='data',stride=1):
    cla_dir = dir_name + '/'+cla
    Image_dir = dir_name+'/'+cla+ '/Images'
    Mask_dir = dir_name +'/'+cla+  '/Masks'
    print(cla_dir)
    directories = [dir_name,Image_dir,Mask_dir,cla_dir]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    max_x = int(((img.shape[0]-height)/stride)+1)
    max_y = int(((img.shape[1]-width)/stride)+1)
    max_crops = (max_x)*(max_y)

    crop_seq = [i for i in range(0,max_crops)]

    for i in range(num_of_crops):
        crop = random.choice(crop_seq)
        #print("crop_value for",i,":",crop)
        if crop ==0:
            x = 0
            y = 0
        else:
            x = int((crop+1)/max_y)
            y = int((crop+1)%max_y)

        crop_img_arr = img[x:x+width,y:y+height]

        crop_mask_arr = mask[x:x+width,y:y+height]
        crop_img = Image.fromarray(crop_img_arr)
        crop_mask = Image.fromarray(crop_mask_arr)
        img_name =  directories[3] + "/Images/" + name + "_" + str(i+1)+".png"
        mask_name = directories[3] + "/Masks/" + name + "_mask_" + str(i+1)+".png"
        crop_img.save(img_name)
        crop_mask.save(mask_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=512)
    parser.add_argument('--n_crops', type=int, default=50)
    parser.add_argument('--datadir', type=str, default='E:\PycharmProjects\mynewGAN\data')
    parser.add_argument('--output_dir', type=str, default='data')
    args = parser.parse_args()

    #final = len(os.listdir('E:\PycharmProjects\mynewGAN\data\ABNORMAL'))
    random_list = []
    data_size = 20
    while len(random_list) < data_size:
        random_number = random.randint(1, 20)
        if (random_number not in random_list):
            random_list.append(random_number)
            print("add success!")
    #Abnormal Fundus/Angio Image pairs

    for item in tqdm(random_list):
        img_name = args.datadir+"/ABNORMAL/"+str(item)+".jpg"
        im = Image.open(img_name)
        img_arr = np.asarray(im)
        mask_name = args.datadir+"/ABNORMAL/"+str(item)+"-"+str(item)+".jpg"
        mask = Image.open(mask_name)
        mask_arr = np.asarray(mask)
        name = str('abnormal')+str(item)
        random_crop(img_arr, mask_arr, args.input_dim, args.input_dim, args.n_crops, name,'ab')
    #Normal Fundus/Angio Image pairs
    for item in tqdm(random_list):
        img_name = args.datadir+"/NORMAL/"+str(item)+"-"+str(item)+".jpg"
        im = Image.open(img_name)
        img_arr = np.asarray(im)
        mask_name = args.datadir+"/NORMAL/"+str(item)+".jpg"
        mask = Image.open(mask_name)
        mask_arr = np.asarray(mask)
        name = str('normal')+str(item)
        random_crop(img_arr, mask_arr, args.input_dim, args.input_dim, args.n_crops,name,'nor')

