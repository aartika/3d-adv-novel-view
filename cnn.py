import torch.nn as nn
import torch.nn.functional as F

from ops import volume_proj, rotate_volume

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

