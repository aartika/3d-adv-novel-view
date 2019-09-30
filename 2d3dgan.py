import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import random

from keras.utils import np_utils
from math import sqrt

from tqdm import tqdm
import glob
from ops import volume_proj, rotate_volume
from dataset import *
from utils import write_obj_from_array


parser = argparse.ArgumentParser()
parser.add_argument('--gan', type=str, default='2d-3d_gan', help='gan type')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--g_lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--g_lr_2d', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--d_lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--enc_lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--encoding_dim', type=int, default=1024, help='dimensionality of the latent space')
parser.add_argument('--noise_dim', type=int, default=16, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=50, help='interval between image sampling')
parser.add_argument('--eval_interval', type=int, default=5, help='interval in epochs between validating')
parser.add_argument('--n_critic', type=int, default=1, help='number of training steps for discriminator per iter')
parser.add_argument('--g_optim', type=str, default='rmsprop', help='leaky relu in generator')
parser.add_argument('--d_optim', type=str, default='rmsprop', help='leaky relu in generator')
parser.add_argument('--enc_optim', type=str, default='rmsprop', help='leaky relu in generator')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--output_dir', type=str, default='data', help='data directory')
parser.add_argument('--lr_decay', type=float, default=0.5, help='leaky relu in generator')
parser.add_argument('--tau', type=float, default=0.25, help='')
parser.add_argument('--restore', type=bool, default=True, help='')
parser.add_argument('--binarize', type=str, default='', help='')
parser.add_argument('--save_voxels_after', type=int, default=150, help='')
parser.add_argument('--validate_after', type=int, default=35, help='')
parser.add_argument('--lambda_3d', type=float, default=1, help='')
parser.add_argument('--gamma', type=float, default=1, help='')
parser.add_argument('--eta', type=float, default=1, help='')
parser.add_argument('--enc_epochs', type=int, default=10, help='')
parser.add_argument('--base_filters', type=int, default=128, help='')
parser.add_argument('--train_3d', type=bool, default=True, help='')
parser.add_argument('--sup_ratio', type=float, default=1, help='')
parser.add_argument('--n_views', type=int, default=1, help='')
parser.add_argument('--class_choice', type=str, default='plane', help='')
parser.add_argument('--neg_views', default=False, action='store_true', help='')
parser.add_argument('--train_2d', default=False, action='store_true', help='')
parser.add_argument('--eval_only', default=False, action='store_true')
parser.add_argument('--save_voxels', default=False, action='store_true')
parser.add_argument('--loss_avg', default=False, action='store_true')
opt = parser.parse_args()

tqdm.write(str(opt))

opt_dict = vars(opt)
opt_str = '{}/{}'.format(opt.class_choice, '_'.join("{}={}".format(key,opt_dict[key]) for key in ['gan', 'img_size', 'g_lr_2d', 'g_lr', 'd_lr', 'encoding_dim', 'noise_dim', 'g_optim', 'd_optim', 'tau', 'binarize', 'lambda_3d', 'base_filters', 'n_views', 'train_2d', 'neg_views']))

voxels_dir = os.path.join(opt.output_dir, 'voxels/{}'.format(opt_str))
images_dir = os.path.join(opt.output_dir, 'images/{}'.format(opt_str))
models_dir = os.path.join(opt.output_dir, 'models/{}'.format(opt_str))
val_dir = os.path.join(opt.output_dir, 'val/{}/val.txt'.format(opt_str))
val_file=open(os.path.join(val_dir, 'val.txt'.format(opt_str)), "a+")
#os.makedirs('aug19_allviews/{}'.format(opt_str), exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(voxels_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class LayerNorm(nn.Module):
    def __init__(self, num_features):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out = self.layer_norm(x)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out 

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.d_size = 64

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 5, 2, 1),
                        nn.ReLU()]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, self.d_size, bn=False),
            *discriminator_block(self.d_size, self.d_size*2),
            *discriminator_block(self.d_size*2, self.d_size*4),
            *discriminator_block(self.d_size*4, self.d_size*8),
            *discriminator_block(self.d_size*8, self.d_size*16)
        )

        # The height and width of downsampled image
        #ds_size = opt.img_size // 2**4
        ds_size = 1
        #self.adv_layer = nn.Sequential( nn.Linear(self.d_size*8*ds_size**2, 1),
        #                                nn.Sigmoid())
        self.adv_layer = nn.Sequential( nn.Linear(self.d_size*16*ds_size**2, opt.encoding_dim))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        out = self.adv_layer(out)

        return out 

def deconv2d_add3(in_filters, out_filters):
    return torch.nn.ConvTranspose2d(in_filters, out_filters, kernel_size=4, stride=1, padding=0)

def deconv2d_2x(in_filters, out_filters):
    return torch.nn.ConvTranspose2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.d_size = opt.base_filters 
        self.init_size = opt.img_size // 8 
        self.l1 = nn.Sequential(nn.Linear(opt.encoding_dim, self.d_size*self.init_size**2))

        #self.conv_blocks = nn.Sequential(
        #    nn.BatchNorm2d(self.d_size),
        #    nn.Upsample(scale_factor=2),
        #    nn.Conv2d(self.d_size, self.d_size // 2, 3, stride=1, padding=1),
        #    nn.BatchNorm2d(self.d_size // 2, 0.8),
        #    nn.ReLU(),
        #    nn.Upsample(scale_factor=2),
        #    nn.Conv2d(self.d_size // 2, self.d_size // 4, 3, stride=1, padding=1),
        #    nn.BatchNorm2d(self.d_size // 4, 0.8),
        #    nn.ReLU(),
        #    nn.Upsample(scale_factor=2),
        #    nn.Conv2d(self.d_size // 4, self.d_size // 8, 3, stride=1, padding=1),
        #    nn.BatchNorm2d(self.d_size // 8, 0.8),
        #    nn.ReLU(),
        #    nn.Conv2d(self.d_size // 8, opt.channels, 3, stride=1, padding=1),
        #    nn.Sigmoid()
        #)

        if opt.img_size == 32:
            conv_layers = [
                    deconv2d_2x(opt.encoding_dim, self.d_size)
            ]
        else: 
            conv_layers = [
                    deconv2d_add3(opt.encoding_dim, self.d_size)
            ]
        conv_layers += [
            torch.nn.BatchNorm2d(self.d_size),
            nn.LeakyReLU(0.2, inplace=True),

            deconv2d_2x(self.d_size, self.d_size // 2),
            torch.nn.BatchNorm2d(self.d_size // 2),
            nn.LeakyReLU(0.2, inplace=True),

            deconv2d_2x(self.d_size // 2, self.d_size // 4),
            torch.nn.BatchNorm2d(self.d_size // 4),
            nn.LeakyReLU(0.2, inplace=True),

            deconv2d_2x(self.d_size // 4, self.d_size // 8),
            torch.nn.BatchNorm2d(self.d_size // 8),
            nn.LeakyReLU(0.2, inplace=True),
            ]

        if opt.img_size == 128:
            conv_layers += [
                    deconv2d_2x(self.d_size // 8, self.d_size // 8),
                    torch.nn.BatchNorm2d(self.d_size // 8),
                    nn.LeakyReLU(0.2, inplace=True),
            ]

        conv_layers += [
            deconv2d_2x(self.d_size // 8, opt.channels),
            nn.Sigmoid()
        ] 
        self.conv_blocks = torch.nn.Sequential(*conv_layers)
 
    def forward(self, z):
        #out = self.l1(z)
        #out = out.view(out.shape[0], self.d_size, self.init_size, self.init_size)
        z = z.unsqueeze(2).unsqueeze(3)
        img = self.conv_blocks(z)
        return img

class Generator2D(nn.Module):
    def __init__(self):
        super(Generator2D, self).__init__()

        self.d_size = opt.base_filters 
        self.init_size = opt.img_size // 4  
        self.l1 = nn.Sequential(nn.Linear(opt.encoding_dim+opt.noise_dim+9, self.d_size*self.init_size**2))

        if opt.img_size == 32:
            conv_layers = [
                    deconv2d_2x(opt.encoding_dim+opt.noise_dim+9, self.d_size)
            ]
        else: 
            conv_layers = [
                    deconv2d_add3(opt.encoding_dim+opt.noise_dim+9, self.d_size)
            ]
        conv_layers += [
            torch.nn.BatchNorm2d(self.d_size),
            nn.LeakyReLU(0.2, inplace=True),

            deconv2d_2x(self.d_size, self.d_size // 2),
            torch.nn.BatchNorm2d(self.d_size // 2),
            nn.LeakyReLU(0.2, inplace=True),

            deconv2d_2x(self.d_size // 2, self.d_size // 4),
            torch.nn.BatchNorm2d(self.d_size // 4),
            nn.LeakyReLU(0.2, inplace=True),
                
            deconv2d_2x(self.d_size // 4, self.d_size // 8),
            torch.nn.BatchNorm2d(self.d_size // 8),
            nn.LeakyReLU(0.2, inplace=True),
            ]

        if opt.img_size == 128:
            conv_layers += [
                    deconv2d_2x(self.d_size // 8, self.d_size // 8),
                    torch.nn.BatchNorm2d(self.d_size // 8),
                    nn.LeakyReLU(0.2, inplace=True),
            ]

        conv_layers += [
            deconv2d_2x(self.d_size // 8, opt.channels),
            nn.Sigmoid()
        ] 
        self.conv_blocks = torch.nn.Sequential(*conv_layers)
        #self.conv_blocks = nn.Sequential(
        #    nn.BatchNorm2d(128),
        #    nn.Upsample(scale_factor=2),
        #    nn.Conv2d(128, 128, 3, stride=1, padding=1),
        #    nn.BatchNorm2d(128, 0.8),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    nn.Upsample(scale_factor=2),
        #    nn.Conv2d(128, 64, 3, stride=1, padding=1),
        #    nn.BatchNorm2d(64, 0.8),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
        #    nn.Tanh()
        #)

    def forward(self, z):
        #out = self.l1(z)
        #out = out.view(out.shape[0], self.d_size, self.init_size, self.init_size)
        z = z.unsqueeze(2).unsqueeze(3)
        img = self.conv_blocks(z)
        return img


def project_voxels(voxels, views=None, neg_views=None):
    #vp = [0, np.pi/4.0, np.pi/2.0, 3*np.pi/4.0, np.pi, 5*np.pi/4.0, 3*np.pi/2.0, 7*np.pi/4.0, np.pi/2.0]
    vp = [0, np.pi/4.0, np.pi/2.0, 3*np.pi/4.0, np.pi, 5*np.pi/4.0, 3*np.pi/2.0, 7*np.pi/4.0, 0]
    hp = [0, 0, 0, 0, 0, 0, 0, 0, np.pi/2.0]
    proj_imgs = []

    if views is None:
        views = []
        for i in range(voxels.size()[0]):
            population = range(0, 9)
            if opt.neg_views and neg_views is not None:
                population = [j for j in range(0, 9) if j not in [neg_view[i] for neg_view in neg_views]]
            views.append(random.choice(population))

    for i in range(voxels.size()[0]):
        idx = views[i]
        cube_shape = (opt.img_size, opt.img_size, opt.img_size)
        proj_img = volume_proj(voxels[i].view(cube_shape), 
            views=torch.tensor([[vp[idx], 0, hp[idx]]], dtype=torch.float), tau=opt.tau)
        proj_imgs.append(proj_img.unsqueeze(0))

    proj_imgs = torch.cat(proj_imgs, 0)
    proj_imgs = proj_imgs.permute(0, 3, 1, 2)
    return proj_imgs, views 


class Generator3D(nn.Module):
    def __init__(self, cube_len, latent_dim, lrelu=False, tau=0.5, spectralnorm=False):
        super(Generator3D, self).__init__()

        self.cube_len = cube_len
        self.base_filters = 128 
        self.tau = tau
        self.spectralnorm = spectralnorm
        if cube_len == 32:
            self.init_len = 2 
        else:
            self.init_len = 4 

        self.l1 = nn.Sequential(nn.Linear(latent_dim, self.init_len**3*self.base_filters))

        if lrelu:
            self.non_lin = torch.nn.LeakyReLU(0.1)
        else:
            self.non_lin = torch.nn.ReLU()

        conv_layers = [self.conv3d_2x(self.base_filters, self.base_filters//2),
                self.conv3d_2x(self.base_filters//2, self.base_filters//4),
                self.conv3d_2x(self.base_filters//4, self.base_filters//8),
                ]
        
        if cube_len == 128:
            conv_layers.append(self.conv3d_2x(self.base_filters//8, self.base_filters//8))

        padd = (1, 1, 1)
        conv_layers += [
                torch.nn.ConvTranspose3d(self.base_filters//8, 1, kernel_size=4, stride=2, bias=False, padding=padd),
                torch.nn.Sigmoid()
                ]
        self.conv_blocks = torch.nn.Sequential(*conv_layers)

    def forward(self, x):
        #out = x.view(-1, latent_dim, 1, 1, 1)
        out = self.l1(x)
        #out = out.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        out = out.view(-1, self.base_filters, self.init_len, self.init_len, self.init_len)
        out = self.conv_blocks(out)
        return out

    def conv3d_2x(self, in_filters, out_filters):
        padd = (1, 1, 1)
        convlayer = torch.nn.ConvTranspose3d(in_filters, out_filters, kernel_size=4, stride=2, bias=False, padding=padd)
        if self.spectralnorm:
            convlayer = nn.utils.spectral_norm(convlayer)
        return torch.nn.Sequential(
                convlayer,
                torch.nn.BatchNorm3d(out_filters),
                self.non_lin)

#class Discriminator(nn.Module):
#    def __init__(self):
#        super(Discriminator, self).__init__()
#
#        def discriminator_block(in_filters, out_filters, bn=False):
#            block = [   nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1)),
#                        nn.LeakyReLU(0.1, inplace=True)]
#            if bn:
#                block.append(nn.BatchNorm2d(out_filters, 0.8))
#            return block
#
#        self.model = nn.Sequential(
#            *discriminator_block(opt.channels+9, 16),
#            *discriminator_block(16, 32),
#            *discriminator_block(32, 64),
#            *discriminator_block(64, 128),
#        )
#
#        # The height and width of downsampled image
#        ds_size = opt.img_size // 2**4
#        self.adv_layer = nn.Sequential( nn.Linear(128*ds_size**2, 1))
#
#    def forward(self, img):
#        out = self.model(img)
#        out = out.view(out.shape[0], -1)
#        validity = self.adv_layer(out)
#
#        return validity

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.d_size = 16 

        def discriminator_block(in_filters, out_filters, ln=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if ln:
                block.append(LayerNorm(out_filters))
            block.append(nn.LeakyReLU(0.1, inplace=True))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels+9, self.d_size, ln=False),
            *discriminator_block(self.d_size, self.d_size*2),
            *discriminator_block(self.d_size*2, self.d_size*4),
            *discriminator_block(self.d_size*4, self.d_size*8),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2**4
        self.adv_layer = nn.Sequential( nn.Linear(self.d_size*8*ds_size**2, 1))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
encoder = Encoder()
encoder2 = Encoder()
decoder = Decoder()
generator2d = Generator2D()
# Uncomment this to use generator with Upsampling and Conv instead of Deconv
#generator3d = Generator(opt.img_size, opt.encoding_dim+opt.noise_dim, lrelu=True, tau=opt.tau)

generator3d = Generator3D(opt.img_size, opt.encoding_dim+opt.noise_dim, lrelu=True, tau=opt.tau)
discriminator = Discriminator()

if cuda:
    encoder.cuda()
    encoder2.cuda()
    decoder.cuda()
    generator2d.cuda()
    generator3d.cuda()
    discriminator.cuda()

# Initialize weights
encoder.apply(weights_init_normal)
decoder.apply(weights_init_normal)
generator2d.apply(weights_init_normal)
generator3d.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

def binarize(img, binarize):
    if binarize == 'soft':
        img = 2*img
        img[img > 1] = 1
    elif binarize == 'hard':
        img[img > 0] = 1
    return img

# Configure data loader
# os.makedirs('data/airplane64', exist_ok=True)
#dataset = datasets.ImageFolder('data/airplane64_split',
#               transform=transforms.Compose([
#                   transforms.Grayscale(),
#                   transforms.Resize(opt.img_size),
#                   transforms.ToTensor(),
#                   lambda img: binarize(img, opt.binarize),
#               ]))

rootvol = os.path.join(opt.data_dir, 'modelVoxels{}'.format(opt.img_size))
if opt.n_views < 9:
    train_rootimg = os.path.join(opt.data_dir, 'subdatasets/shapenetcore_{}/masks'.format(opt.data_dir, opt.n_views))
else:
    train_rootimg = os.path.join(opt.data_dir, 'shapenetcore/masks')
val_rootimg = os.path.join(opt.data_dir, 'shapenetcore/masks')
test_rootimg = os.path.join(opt.data_dir, 'shapenetcore/masks')
splitfile = os.path.join(opt.data_dir, 'splits.json')
catfile = os.path.join(opt.data_dir, 'synsetoffset2category.txt')
dataset  =  ShapeNet(class_choice =  opt.class_choice, rootimg=train_rootimg, rootvol=rootvol, img_size=opt.img_size, catfile=catfile, splitfile=splitfile, data_split='train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

import ipdb; ipdb.set_trace()

val_dataloaders = []
for idx in range(0,9):
    val_dataset = ShapeNet(class_choice = opt.class_choice, rootimg=val_rootimg, rootvol=rootvol, img_size=opt.img_size, catfile=catfile, splitfile=splitfile, data_split='val', idx=idx)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
    val_dataloaders.append(val_dataloader)

test_dataloaders = []
for idx in range(0,9):
    test_dataset = ShapeNet(class_choice = opt.class_choice, rootimg=test_rootimg, rootvol=rootvol, img_size=opt.img_size, catfile=catfile, splitfile=splitfile, data_split='test', idx=idx)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    test_dataloaders.append(test_dataloader)

# Optimizers
params_enc = [*encoder.parameters(), *decoder.parameters()]
params_3d = [*generator3d.parameters(), *encoder.parameters()]
params_2d = [*generator2d.parameters(), *encoder2.parameters()]
params_G = [*encoder.parameters()]
if opt.g_optim == 'adam':
    optimizer_enc = torch.optim.Adam(params_enc, lr=opt.g_lr, betas=(opt.b1, opt.b2))
    optimizer_2d = torch.optim.Adam(params_2d, lr=opt.g_lr_2d, betas=(opt.b1, opt.b2))
    optimizer_3d = torch.optim.Adam(params_3d, lr=opt.g_lr, betas=(opt.b1, opt.b2))
    optimizer_G = torch.optim.Adam(params_G, lr=opt.g_lr, betas=(opt.b1, opt.b2))
else:
    optimizer_enc = torch.optim.RMSprop(params_enc, lr=opt.g_lr)
    optimizer_2d = torch.optim.RMSprop(params_2d, lr=opt.g_lr_2d)
    optimizer_3d = torch.optim.RMSprop(params_3d, lr=opt.g_lr)
    optimizer_G = torch.optim.RMSprop(params_G, lr=opt.g_lr)

if opt.d_optim == 'adam':
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))
else:
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.d_lr)

# LR schedulers
#scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_2d, 5, opt.lr_decay)
#scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, 5, opt.lr_decay)

# L1 Loss
l1_loss = nn.L1Loss()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    #alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    alpha = torch.rand(real_samples.shape[0], 1)
    alpha = alpha.expand(
            real_samples.shape[0], real_samples.nelement() // real_samples.shape[0]
            ).contiguous().view(*real_samples.shape).cuda()
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def loss_hinge_dis(dis_real, dis_fake):
    loss = torch.nn.functional.relu(1.0 - dis_real).mean() + \
    torch.nn.functional.relu(1.0 + dis_fake).mean()
    return loss

def loss_hinge_gen(dis_fake):
    loss = -dis_fake.mean()
    return loss

# ----------
#  Training
# ----------

enc_loss = torch.tensor(0.0)
g_loss = torch.tensor(0)
loss_3d = torch.tensor(0)
d_loss = torch.tensor(0)

def cosine_similarity(imgs_1, imgs_2):
    imgs_1 = imgs_1.view(imgs_1.size()[0], -1)
    imgs_2 = imgs_2.view(imgs_2.size()[0], -1)
    return torch.nn.functional.cosine_similarity(imgs_1, imgs_2)

epochs_done = 0
best_val_iou = 0
if opt.eval_only:
    saved_models = glob.glob(os.path.join(models_dir, 'best_val*.ckpt'))
else:
    saved_models = glob.glob(os.path.join(models_dir, '*.ckpt'))
if opt.restore and len(saved_models) > 0:
    latest = max(saved_models, key=os.path.getctime)
    print(latest)
    params = torch.load(latest)
    print(params.keys())
    epochs_done = params['epoch']
    best_val_iou = params['best_val_iou']
    encoder.load_state_dict(params['encoder'])
    encoder2.load_state_dict(params['encoder2'])
    decoder.load_state_dict(params['decoder'])
    generator3d.load_state_dict(params['generator3d'])
    optimizer_enc.load_state_dict(params['optimizer_enc'])
    optimizer_3d.load_state_dict(params['optimizer_3d'])
    optimizer_G.load_state_dict(params['optimizer_G'])
    if opt.train_2d:
        generator2d.load_state_dict(params['generator2d'])
        discriminator.load_state_dict(params['discriminator'])
        optimizer_2d.load_state_dict(params['optimizer_2d'])
        optimizer_D.load_state_dict(params['optimizer_D'])

batches_done = len(dataloader)*epochs_done
pbar1 = tqdm(total=opt.n_epochs, initial=epochs_done, mininterval=0, desc='Training')

def save_ckpt(name):
        ckpt = {
                'epoch' : epoch,
                'best_val_iou': best_val_iou,
                'encoder': encoder.state_dict(),
                'encoder2': encoder2.state_dict(),
                'decoder': decoder.state_dict(),
                'generator2d': generator2d.state_dict(),
                'generator3d': generator3d.state_dict(),
                'discriminator' : discriminator.state_dict(),
                'optimizer_enc': optimizer_enc.state_dict(),
                'optimizer_2d': optimizer_2d.state_dict(),
                'optimizer_3d': optimizer_3d.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                }
        torch.save(ckpt, os.path.join(models_dir, name))

def evaluate(epoch, mode='val'):
        if mode == 'test':
            eval_dataloaders = test_dataloaders
        else:
            eval_dataloaders = val_dataloaders

        # VALIDATE
        all_ious = []
        mean_ious = []
        for idx, eval_dataloader in enumerate(eval_dataloaders):
            ious = []
            pbar_val = tqdm(total=len(eval_dataloader), desc='Validating')
            for val_i, (val_imgs, val_vol, val_labels) in enumerate(eval_dataloader):
            
                # Configure input
                val_imgs = Variable(val_imgs.type(Tensor))
            
                # Sample noise as generator input
                enc = encoder(val_imgs[:, 0, :, :].unsqueeze(1))
                if opt.noise_dim > 0:
                    noise = Variable(Tensor(np.random.normal(0, 1, (val_imgs.size()[0], opt.noise_dim))))
                    z = torch.cat([enc, noise], dim=1)
                else:
                    z = enc
            
                # Generate a batch of voxels 
                voxels = generator3d(z)

                voxels = voxels.squeeze(1)
                voxels[voxels > 0.5] = 1
                voxels[voxels <= 0.5] = 0
                val_vol = val_vol.cuda()

                if opt.class_choice in ['car', 'firearm', 'chair', 'table', 'lamp']:
                    for k in range(val_vol.size()[0]):
                        val_vol[k] = rotate_volume(val_vol[k], y=torch.tensor(np.pi), z=torch.tensor(3*np.pi/2))

                intersection = (voxels * val_vol).view(-1, opt.img_size**3).sum(1)
                union = (voxels + val_vol.cuda()).view(-1, opt.img_size**3).sum(1) - intersection
                iou = intersection / union
                #import ipdb; ipdb.set_trace() 
                ious += iou.detach().cpu().numpy().tolist()
            
                pbar_val.set_postfix(intersection=intersection.mean().item(), union=union.mean().item(), idx=idx, iou=np.mean(ious)) 
            
                if opt.eval_only and opt.save_voxels and val_i == 0:
                    nrows = (int)(sqrt(opt.batch_size))
                    save_image(val_imgs.data[:, 0, :, :].unsqueeze(1), os.path.join(images_dir, '{}_val_{}_{}.png'.format(idx, batches_done, val_i)), nrow=nrows, normalize=True)
                    np.savetxt(os.path.join(voxels_dir, '{}_ious_{}_{}.txt'.format(idx, batches_done, val_i)), iou.detach().cpu().numpy())
                    for j in range(voxels.size()[0]):
                        #np.save(os.path.join(voxels_dir, '{}_val_{}_{}_{}.npy'.format(idx, batches_done, val_i, j)), voxels.detach().cpu().numpy()[j, :, :, :])
                        #np.save(os.path.join(voxels_dir, '{}_real_{}_{}_{}.npy'.format(idx, batches_done, val_i, j)), val_vol.cpu().numpy()[j, :, :, :])
                        write_obj_from_array(os.path.join(voxels_dir, '{}_val_{}_{}_{}.obj'.format(idx, batches_done, val_i, j)), voxels.detach().cpu().numpy()[j, :, :, :])
                        if idx == 0:
                            write_obj_from_array(os.path.join(voxels_dir, 'real_{}_{}_{}.obj'.format(batches_done, val_i, j)), val_vol.cpu().numpy()[j, :, :, :])
                pbar_val.update(1)
            #import ipdb; ipdb.set_trace()
            all_ious += ious
            mean_ious.append(np.mean(ious))
        mean_val_iou = np.mean(all_ious)
        val_file.write('mode: {}, epoch: {}, batches_done: {}, mean_val_iou: {}, best_val_iou: {}, view_mean_ious: {}\n'.format(mode, epoch, batches_done, mean_val_iou, best_val_iou, mean_ious))
        val_file.flush()
        return mean_val_iou

if opt.eval_only:
    evaluate(epochs_done, mode='test')
    exit(0)

for epoch in range(epochs_done+1, opt.n_epochs+1):

    # TRAINING
    pbar2 = tqdm(total=len(dataloader), desc='Epoch {}'.format(epoch))
    #scheduler_G.step()
    #scheduler_D.step()
    for i, (imgs, labels) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs[0].type(Tensor))

        # Convert labels to categorical labels and concat with images
        real_views = np_utils.to_categorical(labels[0], num_classes=9)
        real_views =  Variable(torch.tensor(real_views))
        real_views_tiled = real_views.unsqueeze(2).unsqueeze(3).repeat(1, 1, opt.img_size, opt.img_size).cuda()
        real_imgs = torch.cat([real_imgs, real_views_tiled], dim=1)

        # Sample noise as generator input
        enc = encoder(real_imgs[:, 0, :, :].unsqueeze(1))
        enc2 = encoder2(real_imgs[:, 0, :, :].unsqueeze(1))
        if opt.noise_dim > 0:
            noise = Variable(Tensor(np.random.normal(0, 1, (imgs[0].shape[0], opt.noise_dim))))
            z = torch.cat([enc, noise], dim=1)
            z2 = torch.cat([enc2, noise], dim=1)
        else:
            z = enc
            z2 = enc

        if epoch <= opt.enc_epochs:
            optimizer_enc.zero_grad()
            enc = encoder(real_imgs[:, 0, :, :].unsqueeze(1))
            dec = decoder(enc)
            enc_loss = l1_loss(dec, real_imgs[:, 0, :, :].unsqueeze(1))
            enc_loss.backward()
            optimizer_enc.step()

            if batches_done % opt.sample_interval == 0:
                nrows = (int)(sqrt(opt.batch_size))
                save_image(real_imgs.data[:, 0, :, :].unsqueeze(1), os.path.join(images_dir, '{}_enc.png'.format(batches_done)), nrow=nrows, normalize=True)
                save_image(dec.data[:, 0, :, :].unsqueeze(1), os.path.join(images_dir, '{}_dec.png'.format(batches_done)), nrow=nrows, normalize=True)

            encoder2.load_state_dict(encoder.state_dict())
        else:
            # Generate a batch of voxels 
            voxels = generator3d(z)
            proj_imgs, proj_views = project_voxels(voxels, views=None, neg_views=labels)
            proj_views = torch.tensor(np_utils.to_categorical(proj_views, num_classes=9)).cuda()
    
            # Generate a batch of images
            if opt.train_2d:
                fake_imgs = generator2d(torch.cat([z2, proj_views], dim=1))
                proj_views_tiled = proj_views.unsqueeze(2).unsqueeze(3).repeat(1, 1, opt.img_size, opt.img_size)
                fake_imgs = torch.cat([fake_imgs, proj_views_tiled], dim=1)
    
                # real images
                real_validity = discriminator(real_imgs)
                # fake images
                fake_validity = discriminator(fake_imgs)
    
            optimizer_G.zero_grad()
            loss_G = torch.tensor(0.0).cuda()

            if opt.train_3d:
                optimizer_3d.zero_grad()
                loss_3d = torch.tensor(0.0).cuda()
                if opt.train_2d:
                    den = len(imgs) if opt.loss_avg else 1
                    loss_3d -= opt.gamma * cosine_similarity(fake_imgs[:, 0, :, :], proj_imgs).mean() / den

                proj_real_views = []
                for n, _ in enumerate(imgs):
                    in_imgs = Variable(imgs[n].type(Tensor))
                    proj_in_views = project_voxels(voxels, labels[n])[0]
                    proj_real_views.append(proj_in_views)
                    if n == 0:
                        loss_3d += -1 * cosine_similarity(in_imgs[:, 0, :, :], proj_in_views).mean()
                    elif opt.loss_avg:
                        den = len(imgs) if opt.train_2d else len(imgs) - 1
                        loss_3d += -1 * cosine_similarity(in_imgs[:, 0, :, :], proj_in_views).mean() / den
                    else:
                        loss_3d += -1 * cosine_similarity(in_imgs[:, 0, :, :], proj_in_views).mean()

                loss_G += loss_3d

                loss_3d.backward(retain_graph=True)
                optimizer_3d.step()
                if batches_done % opt.sample_interval == 0:
                    nrows = (int)(sqrt(opt.batch_size))
                    if opt.train_2d:
                        save_image(proj_imgs.data[:, 0, :, :].unsqueeze(1), os.path.join(images_dir, '{}_proj.png'.format(batches_done)), nrow=nrows, normalize=True)
                    for n, _ in enumerate(imgs):
                        save_image(proj_real_views[n].data[:, 0, :, :].unsqueeze(1), os.path.join(images_dir, '{}_proj_real_{}.png'.format(batches_done, n)), nrow=nrows, normalize=True)
                        save_image(imgs[n].data[:, 0, :, :].unsqueeze(1), os.path.join(images_dir, '{}_real_{}.png'.format(batches_done, n)), nrow=nrows, normalize=True)
                    if epoch >= opt.save_voxels_after:
                        for i in range(voxels.size()[0]):
                            np.save(os.path.join(voxels_dir, '{}_{}.npy'.format(batches_done, i)), voxels.detach().cpu().numpy()[i, 0, :, :, :])
 
            if opt.train_2d:
                # Train the generator every n_critic steps
                if i % opt.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------
                    optimizer_2d.zero_grad()

                    # Loss measures generator's ability to fool the discriminator
                    g_loss = opt.eta * -torch.mean(fake_validity)
                    if opt.train_3d:
                        g_loss -= opt.eta * opt.lambda_3d * cosine_similarity(fake_imgs[:, 0, :, :], proj_imgs).mean()

                    loss_G += g_loss
                    g_loss.backward(retain_graph=True)
                    optimizer_2d.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

                d_loss.backward(retain_graph=True)
                optimizer_D.step()

                if batches_done % opt.sample_interval == 0:
                    nrows = (int)(sqrt(opt.batch_size))
                    save_image(fake_imgs.data[:, 0, :, :].unsqueeze(1), os.path.join(images_dir, '{}_2d.png'.format(batches_done)), nrow=nrows, normalize=True)

            #import ipdb; ipdb.set_trace()
            #loss_G.backward()
            #optimizer_G.step()

        batches_done += 1

        pbar2.set_postfix(enc_loss=enc_loss.item(), enc_lrate=optimizer_enc.param_groups[0]['lr'], 
                D_loss=d_loss.item(), loss_2d=g_loss.item(), loss_3d=loss_3d.item(), 
                D_lrate=optimizer_D.param_groups[0]['lr'], G_lrate_3d=optimizer_3d.param_groups[0]['lr'], G_lrate_2d=optimizer_2d.param_groups[0]['lr'])

        pbar2.update(1)

    if epoch >= opt.validate_after and epoch % opt.eval_interval == 0:
        mean_val_iou = evaluate(epoch)
        if mean_val_iou > best_val_iou:
            best_val_iou = mean_val_iou
            save_ckpt('best_val.ckpt')


    if epoch % 10 == 0:
        save_ckpt('{}.ckpt'.format(epoch))

    pbar1.update(1)
