import torch
import torch.utils.data

import os
import numpy as np
from PIL import Image

from utils.util import *

def loaderAndResize(path):

    return Image.open(path).resize((128, 128))

def loader(path):

    return Image.open(path)


class GlassandAttrFaceImageLoader(torch.utils.data.Dataset):

    def __init__(self, face_img_root, list_wo_g, list_w_g, 
                    transform = None, list_reader = list_reader_all, 
                    loader = loader):

        self.face_img_root = face_img_root
        self.face_list_wo_g = list_reader(list_wo_g)
        self.face_list_w_g = list_reader(list_w_g)

        self.loader = loader
        self.transform = transform


    def __getitem__(self, index):


        none_occ_attr_list = self.face_list_wo_g[index][1 : ]
        face_name = self.face_list_wo_g[index][0]
        none_occ_img = self.loader(os.path.join(self.face_img_root, face_name))

        none_occ_attr = [int(none_occ_attr_list[0]), int(none_occ_attr_list[16]),
                    int(none_occ_attr_list[22]), int(none_occ_attr_list[24]), 
                    int(none_occ_attr_list[30]), int(none_occ_attr_list[20]), 
                    int(none_occ_attr_list[39])]

        for i in range(len(none_occ_attr)):
            if none_occ_attr[i] == -1:
                none_occ_attr[i] = 0

        idx2 = np.random.randint(0, len(self.face_list_w_g))
        occ_attr_list = self.face_list_w_g[idx2][1 : ]
        occ_name = self.face_list_w_g[idx2][0]
        occ_img = self.loader(os.path.join(self.face_img_root, occ_name))

        occ_attr = [int(occ_attr_list[0]), int(occ_attr_list[16]),
                            int(occ_attr_list[22]), int(occ_attr_list[24]), 
                            int(occ_attr_list[30]), int(occ_attr_list[20]), 
                            int(occ_attr_list[39])]



        for i in range(len(occ_attr)):
            if occ_attr[i] == -1:
                occ_attr[i] = 0

        if self.transform is not None:
            occ_img = self.transform(occ_img)
            none_occ_img = self.transform(none_occ_img)

        sample = {'none_occ_img': none_occ_img,
                    'occ_img': occ_img,
                    'occ_attr': torch.from_numpy(np.array(occ_attr)),
                    'none_occ_attr': torch.from_numpy(np.array(none_occ_attr)),
        }

        return sample

    def __len__(self):

        return len(self.face_list_wo_g)

class RandomFaceImageLoader(torch.utils.data.Dataset):

    def __init__(self, face_img_root, img_list, 
                    transform = None, list_reader = list_reader_2, 
                    loader = loader):

        self.face_img_root = face_img_root
        self.face_list, self.label = list_reader(img_list)

        self.loader = loader
        self.transform = transform


    def __getitem__(self, index):


        face_name = self.face_list[index]
        face_label_ori = self.label[index]
        face_label_des = 1
        img = self.loader(os.path.join(self.face_img_root, face_name))

        if self.transform is not None:
            img = self.transform(img)

        sample = {'img': img,
                'label_ori': face_label_ori,
                'label_des': face_label_des,
        }

        return sample


    def __len__(self):

        return len(self.face_list)

class GlassFaceImageLoader(torch.utils.data.Dataset):

    def __init__(self, face_img_root, list_wo_g, list_w_g, 
                    transform = None, list_reader = list_reader, 
                    loader = loader):

        self.face_img_root = face_img_root
        self.face_list_wo_g = list_reader(list_wo_g)
        self.face_list_w_g = list_reader(list_w_g)

        self.loader = loader
        self.transform = transform


    def __getitem__(self, index):


        face_name = self.face_list_wo_g[index]
        none_occ_img = self.loader(os.path.join(self.face_img_root, face_name))

        occ_name = self.face_list_w_g[np.random.randint(0, len(self.face_list_w_g))]
        occ_img = self.loader(os.path.join(self.face_img_root, occ_name))


        if self.transform is not None:
            occ_img = self.transform(occ_img)
            none_occ_img = self.transform(none_occ_img)

        sample = {'none_occ_img': none_occ_img,
                    'occ_img': occ_img
        }

        return sample


    def __len__(self):

        return len(self.face_list_wo_g)

class OccFaceImageLoader(torch.utils.data.Dataset):

    def __init__(self, face_img_root, face_name_list, occ_img_root, 
                    occ_name_list, transform = None, list_reader = list_reader_all, 
                    loader = loader):

        self.face_img_root = face_img_root
        self.face_list = list_reader(face_name_list)
        self.occ_img_root = occ_img_root
        self.occ_list = list_reader(occ_name_list)

        self.loader = loader
        self.transform = transform


    def __getitem__(self, index):

        none_occ_attr_list = self.face_list[index][1 : ]
        face_name = self.face_list[index][0]
        none_occ_img = self.loader(os.path.join(self.face_img_root, face_name))

        none_occ_attr = [int(none_occ_attr_list[0]), int(none_occ_attr_list[16]),
                    int(none_occ_attr_list[22]), int(none_occ_attr_list[24]), 
                    int(none_occ_attr_list[30]), int(none_occ_attr_list[20]), 
                    int(none_occ_attr_list[39])]

        for i in range(len(none_occ_attr)):
            if none_occ_attr[i] == -1:
                none_occ_attr[i] = 0

        occ_name = self.occ_list[np.random.randint(0, len(self.occ_list))][0]
        occ_img = self.loader(os.path.join(self.occ_img_root, occ_name))
        
        if occ_name[0] == 'm':
            occ_type = occ_name.split()[0]
        else:
            occ_type = occ_name.split('_')[0]

        print(occ_type)

        occ_face_img = process_image(none_occ_img, occ_img, occ_type)

        if self.transform is not None:

            occ_face_img = self.transform(occ_face_img)
            none_occ_img = self.transform(none_occ_img)

        sample = {'none_occ_img': none_occ_img,
                    'occ_img': occ_face_img,
                    'occ_attr': torch.from_numpy(np.array(none_occ_attr)),
                    'none_occ_attr': torch.from_numpy(np.array(none_occ_attr)),
        }

        return sample


    def __len__(self):

        return len(self.face_list)

class OccFaceImageMixLoader(torch.utils.data.Dataset):

    def __init__(self, face_wo_occ_root, face_wo_occ_list, 
                        occ_root, occ_list,
                        face_w_occ_root, face_w_occ_list,
                        transform = None, loader = loader):

        self.face_wo_occ_root = face_wo_occ_root
        self.face_wo_occ_list = load_pickle(face_wo_occ_list)
        self.occ_root = occ_root
        self.occ_list = load_pickle(occ_list)
        self.face_w_occ_root = face_w_occ_root
        self.face_w_occ_list = load_pickle(face_w_occ_list)

        self.loader = loader
        self.transform = transform


    def __getitem__(self, index):

        face_wo_occ_attr_list = self.face_wo_occ_list[index][1 : ]
        face_wo_occ_img = self.loader(
            os.path.join(self.face_wo_occ_root, self.face_wo_occ_list[index][0]))

        top_x = np.random.randint(0, 16)
        top_y = np.random.randint(0, 16)
        face_wo_occ_img = face_wo_occ_img.crop((top_x, top_y, top_x + 128, top_y + 128))

        face_wo_occ_attr = [int(face_wo_occ_attr_list[0]), int(face_wo_occ_attr_list[16]),
                    int(face_wo_occ_attr_list[22]), int(face_wo_occ_attr_list[24]), 
                    int(face_wo_occ_attr_list[30]), int(face_wo_occ_attr_list[20]), 
                    int(face_wo_occ_attr_list[39])]

        index1 = np.random.randint(0, len(self.face_wo_occ_list))
        face_w_occ_attr_list = self.face_w_occ_list[index1][1 : ]
        face_w_occ_img = self.loader(
            os.path.join(self.face_w_occ_root, self.face_w_occ_list[index1][0]))

        top_x = np.random.randint(0, 16)
        top_y = np.random.randint(0, 16)
        face_w_occ_img = face_w_occ_img.crop((top_x, top_y, top_x + 128, top_y + 128))

        face_w_occ_attr = [int(face_w_occ_attr_list[0]), int(face_w_occ_attr_list[16]),
                    int(face_w_occ_attr_list[22]), int(face_w_occ_attr_list[24]), 
                    int(face_w_occ_attr_list[30]), int(face_w_occ_attr_list[20]), 
                    int(face_w_occ_attr_list[39])]

        occ_name = self.occ_list[np.random.randint(0, len(self.occ_list))][0]
        occ_img = self.loader(os.path.join(self.occ_root, occ_name))
        
        occ_type = occ_name.split('_')[0]

        occ_img_syn = process_image(face_wo_occ_img, occ_img, occ_type)

        if self.transform is not None:

            occ_img_syn = self.transform(occ_img_syn)
            face_wo_occ_img = self.transform(face_wo_occ_img)
            face_w_occ_img = self.transform(face_w_occ_img)


        sample = {'face_wo_occ_img': face_wo_occ_img,
                    'occ_img_syn': occ_img_syn,
                    'face_wo_occ_attr': torch.from_numpy(np.array(face_wo_occ_attr)),
                    'face_w_occ_img': face_w_occ_img,
                    'face_w_occ_attr': torch.from_numpy(np.array(face_w_occ_attr)),
                    'name': self.face_w_occ_list[index1][0],
        }

        return sample


    def __len__(self):

        return len(self.face_w_occ_list)


class OccFaceImageMixLoader_test(torch.utils.data.Dataset):

    def __init__(self, face_wo_occ_root, face_wo_occ_list, 
                        occ_root, occ_list,
                        face_w_occ_root, face_w_occ_list,
                        transform = None, loader = loader):

        self.face_wo_occ_root = face_wo_occ_root
        self.face_wo_occ_list = load_pickle(face_wo_occ_list)
        self.occ_root = occ_root
        self.occ_list = load_pickle(occ_list)
        self.face_w_occ_root = face_w_occ_root
        self.face_w_occ_list = load_pickle(face_w_occ_list)

        self.loader = loader
        self.transform = transform


    def __getitem__(self, index):

        face_wo_occ_attr_list = self.face_wo_occ_list[index][1 : ]
        face_wo_occ_img = self.loader(
            os.path.join(self.face_wo_occ_root, self.face_wo_occ_list[index][0]))

        top_x = 8
        top_y = 8
        face_wo_occ_img = face_wo_occ_img.crop((top_x, top_y, top_x + 128, top_y + 128))

        face_wo_occ_attr = [int(face_wo_occ_attr_list[0]), int(face_wo_occ_attr_list[16]),
                    int(face_wo_occ_attr_list[22]), int(face_wo_occ_attr_list[24]), 
                    int(face_wo_occ_attr_list[30]), int(face_wo_occ_attr_list[20]), 
                    int(face_wo_occ_attr_list[39])]

        index1 = np.random.randint(0, len(self.face_wo_occ_list))
        face_w_occ_attr_list = self.face_w_occ_list[index1][1 : ]
        face_w_occ_img = self.loader(
            os.path.join(self.face_w_occ_root, self.face_w_occ_list[index1][0]))

        top_x = np.random.randint(0, 16)
        top_y = np.random.randint(0, 16)
        face_w_occ_img = face_w_occ_img.crop((top_x, top_y, top_x + 128, top_y + 128))

        face_w_occ_attr = [int(face_w_occ_attr_list[0]), int(face_w_occ_attr_list[16]),
                    int(face_w_occ_attr_list[22]), int(face_w_occ_attr_list[24]), 
                    int(face_w_occ_attr_list[30]), int(face_w_occ_attr_list[20]), 
                    int(face_w_occ_attr_list[39])]

        occ_name = self.occ_list[np.random.randint(0, len(self.occ_list))][0]
        occ_img = self.loader(os.path.join(self.occ_root, occ_name))
        
        occ_type = occ_name.split('_')[0]

        occ_img_syn = process_image(face_wo_occ_img, occ_img, occ_type)

        if self.transform is not None:

            occ_img_syn = self.transform(occ_img_syn)
            face_wo_occ_img = self.transform(face_wo_occ_img)
            face_w_occ_img = self.transform(face_w_occ_img)


        sample = {'face_wo_occ_img': face_wo_occ_img,
                    'occ_img_syn': occ_img_syn,
                    'face_wo_occ_attr': torch.from_numpy(np.array(face_wo_occ_attr)),
                    'face_w_occ_img': face_w_occ_img,
                    'face_w_occ_attr': torch.from_numpy(np.array(face_w_occ_attr)),
                    'name': self.face_w_occ_list[index1][0],
        }

        return sample


    def __len__(self):

        return len(self.face_w_occ_list)


class OccFaceImageMixLoaderV2(torch.utils.data.Dataset):

    def __init__(self, face_wo_occ_root, face_wo_occ_list, 
                        occ_root, occ_list,
                        face_w_occ_root, face_w_occ_list,
                        transform = None, loader = loader):

        self.face_wo_occ_root = face_wo_occ_root
        self.face_wo_occ_list = load_pickle(face_wo_occ_list)
        self.occ_root = occ_root
        self.occ_list = load_pickle(occ_list)
        self.face_w_occ_root = face_w_occ_root
        self.face_w_occ_list = load_pickle(face_w_occ_list)

        self.loader = loader
        self.transform = transform


    def __getitem__(self, index):

        ####################

        face_wo_occ_attr_list = self.face_wo_occ_list[index][1 : ]
        face_wo_occ_img = self.loader(
            os.path.join(self.face_wo_occ_root, self.face_wo_occ_list[index][0]))

        top_x = np.random.randint(0, 16)
        top_y = np.random.randint(0, 16)
        face_wo_occ_img = face_wo_occ_img.crop((top_x, top_y, top_x + 128, top_y + 128))

        face_wo_occ_attr = [int(face_wo_occ_attr_list[0]), int(face_wo_occ_attr_list[16]),
                    int(face_wo_occ_attr_list[22]), int(face_wo_occ_attr_list[24]), 
                    int(face_wo_occ_attr_list[30]), int(face_wo_occ_attr_list[20]), 
                    int(face_wo_occ_attr_list[39])]

        ####################

        index1 = np.random.randint(0, len(self.face_wo_occ_list))
        face_w_occ_attr_list = self.face_w_occ_list[index1][1 : ]
        face_w_occ_img = self.loader(
            os.path.join(self.face_w_occ_root, self.face_w_occ_list[index1][0]))

        top_x = np.random.randint(0, 16)
        top_y = np.random.randint(0, 16)
        face_w_occ_img = face_w_occ_img.crop((top_x, top_y, top_x + 128, top_y + 128))

        face_w_occ_attr = [int(face_w_occ_attr_list[0]), int(face_w_occ_attr_list[16]),
                    int(face_w_occ_attr_list[22]), int(face_w_occ_attr_list[24]), 
                    int(face_w_occ_attr_list[30]), int(face_w_occ_attr_list[20]), 
                    int(face_w_occ_attr_list[39])]

        ####################

        index2 = np.random.randint(0, len(self.face_wo_occ_list))
        face_w_occ_attr_adv_list = self.face_wo_occ_list[index2][1 : ]
        face_wo_occ_img_adv = self.loader(
            os.path.join(self.face_wo_occ_root, self.face_wo_occ_list[index2][0]))

        top_x = np.random.randint(0, 16)
        top_y = np.random.randint(0, 16)
        face_wo_occ_img_adv = face_wo_occ_img_adv.crop((top_x, top_y, top_x + 128, top_y + 128))

        face_w_occ_attr_adv = [int(face_w_occ_attr_adv_list[0]), int(face_w_occ_attr_adv_list[16]),
                    int(face_w_occ_attr_adv_list[22]), int(face_w_occ_attr_adv_list[24]), 
                    int(face_w_occ_attr_adv_list[30]), int(face_w_occ_attr_adv_list[20]), 
                    int(face_w_occ_attr_adv_list[39])]

        ###################

        occ_name = self.occ_list[np.random.randint(0, len(self.occ_list))][0]
        occ_img = self.loader(os.path.join(self.occ_root, occ_name))
        
        occ_type = occ_name.split('_')[0]

        occ_img_syn = process_image(face_wo_occ_img, occ_img, occ_type)

        if self.transform is not None:

            occ_img_syn = self.transform(occ_img_syn)
            face_wo_occ_img = self.transform(face_wo_occ_img)
            face_w_occ_img = self.transform(face_w_occ_img)
            face_wo_occ_img_adv = self.transform(face_wo_occ_img_adv)


        sample = {
                    'face_w_syn_occ_img': occ_img_syn,
                    'face_w_syn_occ_attr': torch.from_numpy(np.array(face_wo_occ_attr)),
                    'face_w_syn_occ_img_GT': face_wo_occ_img,

                    'face_wo_occ_img_adv': face_wo_occ_img_adv,
                    'face_wo_occ_attr_adv': torch.from_numpy(np.array(face_w_occ_attr_adv)),

                    'face_w_occ_img': face_w_occ_img,
                    'face_w_occ_attr': torch.from_numpy(np.array(face_w_occ_attr)),
        }

        return sample


    def __len__(self):

        return len(self.face_w_occ_list)
