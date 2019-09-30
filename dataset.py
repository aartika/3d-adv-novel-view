from __future__ import print_function
import torch.utils.data as data
import os.path
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
#from utils import *
from scipy.io import loadmat
import random
import json


class ShapeNet(data.Dataset):
    def __init__(self, rootimg="./data/subdatasets/shapenetcore_1/masks", rootvol="./data/modelVoxels64", img_size=64, class_choice="chair", normal=False, idx=0, catfile='', splitfile='', data_split='train'):
        self.train = data_split == 'train'
        self.normal = normal
        self.rootimg = rootimg
        self.rootvol = rootvol
        self.img_size = img_size
        self.datapath = []
        self.catfile = catfile
        self.cat = {}
        self.meta = {}
        self.idx=idx
        self.splitfile = splitfile 

        with open(self.splitfile, 'r') as f:
            splits = json.load(f)
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        print(self.cat)
        empty = []
        for item in self.cat:
            if os.path.exists(os.path.join(self.rootimg, self.cat[item])):
                dir_img  = os.path.join(self.rootimg, self.cat[item])
                dir_vol = os.path.join(self.rootvol, self.cat[item])
                #fns_img = sorted(os.listdir(dir_img))
                fns_img = sorted(splits[self.cat[item]][data_split])
            else:
                fns_img = []

            fns = fns_img

            if len(fns) != 0:
                self.meta[item] = []
                for fn in fns:
                    fn_path = os.path.join(dir_img, fn)
                    imgs = os.listdir(fn_path)
                    imgs = [os.path.join(fn_path, img) for img in imgs]
                    for i, img in enumerate(imgs): 
                        in_img = imgs[i]
                        aux_img = imgs[0:i]+imgs[i+1:] 
                        self.meta[item].append((fn_path, in_img, aux_img, os.path.join(dir_vol, fn + '.mat'), item, fn))
            else:
                empty.append(item)
        for item in empty:
            del self.cat[item]
        self.idx2cat = {}
        self.size = {}
        i = 0
        for item in self.cat:
            self.idx2cat[i] = item
            self.size[i] = len(self.meta[item])
            i = i + 1
            for fn in self.meta[item]:
                self.datapath.append(fn)

        def binarize(img):
            img[img > 0] = 1
            return img

        self.transforms = transforms.Compose([
                             transforms.Grayscale(),
                             transforms.Resize(size = self.img_size, interpolation = 2),
                             transforms.ToTensor(),
                             lambda img: binarize(img)
                        ])

        # RandomResizedCrop or RandomCrop
        self.dataAugmentation = transforms.Compose([
                                         transforms.RandomCrop(127),
                                         transforms.RandomHorizontalFlip(),
                            ])
        self.validating = transforms.Compose([
                        transforms.CenterCrop(127),
                        ])

    def __getitem__(self, index):
        fn = self.datapath[index]
        #print(fn[0])

        # load image
        if self.train:
            #N_tot = len(os.listdir(fn[0])) - 3
            #if N_tot==1:
            #    print("only one view in ", fn)
            #if self.gen_view:
            #    N=0
            #else:
            #    N = np.random.randint(1,N_tot)
            imgs = []
            views = []
            all_imgfiles = [fn[1]] + fn[2]
            for img_file in all_imgfiles:
                im = Image.open(img_file)
                view = os.path.splitext(os.path.basename(img_file))[0][1:]
                views.append(int(view))
                data = self.transforms(im) #scale
                data = data[:3,:,:]
                imgs.append(data)
            return imgs, views 
        else:
            view = self.idx
            im = Image.open(os.path.join(fn[0], "v" + str(view) + ".png"))
            vol = self.load_volume(fn[3])
            data = self.transforms(im) #scale
            data = data[:3,:,:]
            return data, vol, view


    def load_volume(self, f):
        return torch.tensor(loadmat(f)['Volume'].astype('float32'))

    def __len__(self):
        return len(self.datapath)

if __name__  == '__main__':
    class_choice = 'table'
    #print('Testing Shapenet dataset')
    #d  =  ShapeNet(class_choice =  class_choice, data_split='train')
    #print('train size : {}'.format(len(d)))
    #print(d.train)
    #d  =  ShapeNet(class_choice =  class_choice, data_split='val')
    #print('val size : {}'.format(len(d)))
    #print(d.train)
    d  =  ShapeNet(class_choice =  class_choice, data_split='train', idx=0, rootimg="./data/shapenetcore/masks")
    print(d.datapath[0])
    print(d[10][1])
    print(d[22][1])
