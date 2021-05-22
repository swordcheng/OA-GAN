import os
import math
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision

def tensor2im(img, imtype=np.uint8, unnormalize=True, idx=0, nrows=None):

    if img.shape[1] == 1:
        # img = np.repeat(img, 3, axis=1)
        img = torch.cat((img, img, img), dim=1)

    # print(img.shape, type(img))
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows)

    img = img.cpu().float()
    if unnormalize:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        for i, m, s in zip(img, mean, std):
            i.mul_(s).add_(m)

    image_numpy = img.numpy()
    image_numpy_t = image_numpy
    image_numpy_t = image_numpy_t*254.0

    return image_numpy_t.astype(imtype)

def tensor2maskim(mask, imtype=np.uint8, idx=0, nrows=None):
    
    im = tensor2im(mask, imtype=imtype, idx=idx, unnormalize=False, nrows=nrows)

    return im

def mkdirs(paths):

    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):

    if not os.path.exists(path):
        os.makedirs(path)

def save_image(image_numpy, image_path):

    mkdir(os.path.dirname(image_path))
    
    image_numpy = image_numpy.transpose((1,2,0))

    image_pil = Image.fromarray(image_numpy)

    image_pil.save(image_path)

def save_str_data(data, path):

    mkdir(os.path.dirname(path))
    np.savetxt(path, data, delimiter=",", fmt="%s")

def load_pickle(file_path):

    f = open(file_path, 'rb')
    file_pickle = pickle.load(f)
    f.close()
    return file_pickle

def list_reader_2(file_list):

    img_list = []
    label = []
    with open(file_list, 'r') as file:
        for line in file.readlines():
            element = line.split()

            img_path = element[0]
            
            if element[16] == '1':
                for i in range(15):
                    label.append(0)
                    img_list.append(img_path)
            else:
                label.append(1)
                img_list.append(img_path)

    return img_list, label


def list_reader_all22(file_list):

    img_list = []
    with open(file_list, 'r') as file:
        for line in file.readlines():
            img_path = line
            img_list.append(img_path[:-1])
    return img_list

def list_reader_all(file_list):

    img_list = []
    with open(file_list, 'rb') as file:
        for line in file.readlines():
            img_path = line
            img_list.append(img_path.split())
    return img_list

def list_reader(file_list):

    img_list = []
    with open(file_list, 'r') as file:
        for line in file.readlines():
            img_path = line
            img_list.append(img_path.split()[0])
    return img_list



def composite_image(top_x, top_y, occ_img, img):
    
    occ = np.zeros((128, 128, 4))
    img = np.asarray(img)
    
    if occ_img.shape[2] == 3:

        occ_img = np.concatenate((occ_img, 255 * np.ones((occ_img.shape[0], occ_img.shape[1], 1))), axis=2)

    occ[top_y : top_y + occ_img.shape[0], top_x : top_x + occ_img.shape[1], :] += occ_img
    occ = occ.astype('uint8')
    composite_img = np.concatenate((img.astype('uint8'), occ, occ[:, :, 3:4], occ[:, :, 3:4]), axis=2)

    final_composite_img = np.multiply((np.ones((128, 128, 3)) * 255 - \
                                       composite_img[:, :, 6:9]).astype('uint8') / 255, 
                                        composite_img[:, :, 0:3]).astype('uint8') + \
                            np.multiply(composite_img[:, :, 6:9].astype('uint8') / 255, 
                                        composite_img[:, :, 3:6]).astype('uint8')
    final_composite_img = Image.fromarray(final_composite_img)
    

    return final_composite_img



def process_image(img, occ_img, occ_type):


    random_ang = (np.random.rand() - 0.5) * 30
    occ_img = occ_img.rotate(random_ang, expand = 1)


    if occ_type == 'glasses' or occ_type == 'sun_glasses':

        occ_img = occ_img.resize((100, int(occ_img.size[1] / (occ_img.size[0] / 100))))
        top_x = 14
        top_y = int(40 - occ_img.size[1] / 2 + (np.random.rand() - 1) * 10)
        top_y = (top_y if(top_y >= 0) else 0)


    elif occ_type == 'scarf' or occ_type == 'cup':

        if occ_img.size[0] >= occ_img.size[1]:
            occ_img = occ_img.resize((64, int(occ_img.size[1] / (occ_img.size[0] / 64))))
        else:
            occ_img = occ_img.resize(((int(occ_img.size[0] / (occ_img.size[1] / 64))), 64))
        top_x = np.random.randint(0, 128 - occ_img.size[0])
        top_y = np.random.randint(128 - occ_img.size[1] -20 , 128 - occ_img.size[1])

    elif occ_type == 'hand':
        if occ_img.size[0] >= occ_img.size[1]:
            occ_img = occ_img.resize((120, int(occ_img.size[1] / (occ_img.size[0] / 120))))
        else:
            occ_img = occ_img.resize(((int(occ_img.size[0] / (occ_img.size[1] / 120))), 120))
        top_x = np.random.randint(0, 128 - occ_img.size[0])
        top_y = np.random.randint(0, 128 - occ_img.size[1])

    elif occ_type == 'phone':
        if occ_img.size[0] >= occ_img.size[1]:
            occ_img = occ_img.resize((80, int(occ_img.size[1] / (occ_img.size[0] / 80))))
        else:
            occ_img = occ_img.resize(((int(occ_img.size[0] / (occ_img.size[1] / 80))), 80))
        top_x = np.random.randint(0, 128 - occ_img.size[0])
        top_y = np.random.randint(0, 128 - occ_img.size[1])

    elif occ_type == 'mask':
        if occ_img.size[0] >= occ_img.size[1]:
            occ_img = occ_img.resize((100, int(occ_img.size[1] / (occ_img.size[0] / 100))))
        else:
            occ_img = occ_img.resize(((int(occ_img.size[0] / (occ_img.size[1] / 100))), 100))
        top_x = np.random.randint(0, 128 - occ_img.size[0])
        top_y = np.random.randint(128 - occ_img.size[1] -20 , 128 - occ_img.size[1])
    
    else:
        if occ_img.size[0] >= occ_img.size[1]:
            occ_img = occ_img.resize((70, int(occ_img.size[1] / (occ_img.size[0] / 70))))
        else:
            occ_img = occ_img.resize(((int(occ_img.size[0] / (occ_img.size[1] / 70))), 70))
        top_x = np.random.randint(0, 128 - occ_img.size[0])
        top_y = np.random.randint(0, 128 - occ_img.size[1])


    final_composite_img = composite_image(top_x, top_y, np.asarray(occ_img), img)

    return final_composite_img