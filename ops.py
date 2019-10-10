# Volume rotation and projection code borrowed from https://github.com/matheusgadelha/PrGAN.

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import sys

def cov(data):
    dmean = torch.mean(data, dim=0)
    centered_data = data - dmean.expand_as(data)
    return torch.mm(centered_data.transpose(0,1), centered_data)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def rotmat_2d(theta):
    mat = torch.zeros(2,2)
    mat[0,0] = torch.cos(theta)
    mat[0,1] = -torch.sin(theta)
    mat[1,0] = torch.sin(theta)
    mat[1,1] = torch.cos(theta)

    return mat


def rotmat_3d(theta, phi, psi):
    #print(theta, phi, psi)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    cos_psi = torch.cos(psi)
    sin_psi = torch.sin(psi)

    xmat = torch.eye(3)
    xmat[1, 1] = cos_theta
    xmat[1, 2] = sin_theta
    xmat[2, 1] = -sin_theta
    xmat[2, 2] = cos_theta

    ymat = torch.eye(3)
    ymat[0, 0] = cos_phi
    ymat[0, 2] = -sin_phi
    ymat[2, 0] = sin_phi
    ymat[2, 2] = cos_phi

    zmat = torch.eye(3)
    zmat[0, 0] = cos_psi
    zmat[0, 1] = sin_psi
    zmat[1, 0] = -sin_psi
    zmat[1, 1] = cos_psi

    #xmat = torch.tensor(
    #        [[1., 0., 0.],
    #         [0., cos_theta, sin_theta],
    #         [0., -sin_theta, cos_theta]])

    #ymat = torch.tensor(
    #        [[cos_phi, 0., -sin_phi],
    #         [0., 1., 0.],
    #         [sin_phi, 0., cos_phi]])

    #zmat = torch.tensor(
    #        [[cos_psi, sin_psi, 0.],
    #         [-sin_psi, cos_psi, 0.],
    #         [0., 0., 1.]])

    out = zmat.mm(ymat).mm(xmat)
    #out = xmat
    return out


def gridcoord_2d(w, h, start=-1.0, end=1.0):
    max_dim = max(w, h)
    
    start=float(start)
    end=float(end)

    xs = torch.linspace(start, end, steps=w)
    ys = torch.linspace(start, end, steps=h)

    yc = ys.repeat(w)
    xc = xs.repeat(h,1).t().contiguous().view(-1)

    out = torch.cat((xc.unsqueeze(1), yc.unsqueeze(1)), 1)
    return out


def gridcoord_3d(size, start=-1.0, end=1.0):
    start=float(start)
    end=float(end)

    xs = torch.linspace(start, end, steps=size)
    ys = torch.linspace(start, end, steps=size)
    zs = torch.linspace(start, end, steps=size)

    zc = zs.repeat(size*size)
    yc = ys.repeat(size, 1).transpose(0,1).contiguous().view(-1) \
           .repeat(size, 1).contiguous().view(-1) 
    xc = xs.repeat(size, 1).transpose(0,1).contiguous().view(-1) \
           .repeat(size, 1).transpose(0,1).contiguous().view(-1) 

    out = torch.cat((xc.unsqueeze(1), yc.unsqueeze(1), zc.unsqueeze(1)), 1)
    return out


def resample_volume(vol, gridcoords, method='nearest'):
    if method=="nearest":
        round_coords = torch.clamp(torch.floor(gridcoords).long(), 
                min=0, max=vol.size()[0]-1)
        out = vol[round_coords[:, 0], round_coords[:, 1], round_coords[:, 2]]
        out = out.reshape(*(vol.size()))
        return out
    if method=="trilinear":
        coords = torch.clamp(gridcoords, 
                min=0, max=vol.size()[0]-1).cuda()

        xs = coords[:, 0]
        ys = coords[:, 1]
        zs = coords[:, 2]
        
        floor_xs = torch.floor(xs).long()
        floor_ys = torch.floor(ys).long()
        floor_zs = torch.floor(zs).long()

        floor_xs_float = torch.floor(xs)
        floor_ys_float = torch.floor(ys)
        floor_zs_float = torch.floor(zs)

        ceil_xs = torch.ceil(xs).long()
        ceil_ys = torch.ceil(ys).long()
        ceil_zs = torch.ceil(zs).long()

        ceil_xs_float = torch.ceil(xs)
        ceil_ys_float = torch.ceil(ys)
        ceil_zs_float = torch.ceil(zs)

        final_value =( torch.abs((xs-floor_xs_float)*(ys-floor_ys_float)*(zs-floor_zs_float))*vol[ceil_xs, ceil_ys, ceil_zs].cuda() + 
                       torch.abs((xs-floor_xs_float)*(ys-floor_ys_float)*(zs-ceil_zs_float))*vol[ceil_xs, ceil_ys, floor_zs].cuda() +
                       torch.abs((xs-floor_xs_float)*(ys-ceil_ys_float)*(zs-floor_zs_float))*vol[ceil_xs, floor_ys, ceil_zs].cuda() +
                       torch.abs((xs-floor_xs_float)*(ys-ceil_ys_float)*(zs-ceil_zs_float))*vol[ceil_xs, floor_ys, floor_zs].cuda() +
                       torch.abs((xs-ceil_xs_float)*(ys-floor_ys_float)*(zs-floor_zs_float))*vol[floor_xs, ceil_ys, ceil_zs].cuda() +
                       torch.abs((xs-ceil_xs_float)*(ys-floor_ys_float)*(zs-ceil_zs_float))*vol[floor_xs, ceil_ys, floor_zs].cuda() +
                       torch.abs((xs-ceil_xs_float)*(ys-ceil_ys_float)*(zs-floor_zs_float))*vol[floor_xs, floor_ys, ceil_zs].cuda() +
                       torch.abs((xs-ceil_xs_float)*(ys-ceil_ys_float)*(zs-ceil_zs_float))*vol[floor_xs, floor_ys, floor_zs].cuda()
                     )

        out = final_value.reshape(*(vol.size()))
        return out


def resample_image(img, gridcoords, method='nearest'):
    if method=="nearest":
        round_coords = torch.clamp(torch.floor(gridcoords).long(), 
                min=0, max=img.size()[0]-1)
        out = img[round_coords[:, 0], round_coords[:, 1]]
        out = out.reshape(*(img.size()))
        return out


def transform_volume(vol, t, method):
    #Assumes volume is square
    size = vol.size()[0]
    grid3d = gridcoord_3d(size)

    transf_grid = torch.mm(grid3d, t)
    transf_grid = (transf_grid + 1.0) * float(size-0.5)/2
    return resample_volume(vol, transf_grid, method)


def rotate_volume(vol, x=torch.tensor(0.0), 
        y=torch.tensor(0.0), z=torch.tensor(0.0), interp='nearest'):
    r = rotmat_3d(x, y, z)
    out = transform_volume(vol, r, interp)
    return out


def inv_rotate_volume(vol, x=torch.tensor(0.0), 
        y=torch.tensor(0.0), z=torch.tensor(0.0)):
    r = rotmat_3d(x, y, z)
    out = transform_volume(vol, r.transpose(0, 1))
    return out


def transform_image(img, t):
    #Assumes image is square
    size = img.size()[0]
    grid2d = gridcoord_2d(size, size)

    transf_grid = torch.mm(grid2d, t)
    transf_grid = (transf_grid + 1.0) * float(size-0.5)/2
    return resample_image(img, transf_grid)


def binary_proj(vol, tau=0.5):
    if tau == 0:
        return vol.max(1)[0]
    s = vol.sum(1)
    out = 1.0 - torch.exp(-tau * s)
    return out

def depth_proj(vol, tau=1):
    c = vol.cumsum(1)
    accessibility = torch.exp(-tau * c)
    depth = 1.0 - torch.exp(-0.01*(accessibility.sum(1)))
    return depth

def sample_2dpts(img, t=0.0):
    randimg = torch.rand(*(img.size())).cuda()
    ptsimg = torch.zeros(*(img.size())).cuda()
    #ptsimg = (randimg < img - t)
    ptsimg = (img > 0.1)
    
    grid = gridcoord_2d(img.size()[0], img.size()[1], 
            start=0, end=img.size()[0]-1).cuda()
    grid = grid.view(img.size()[0], img.size()[1], -1)
    
    pts = grid[ptsimg]
    out = torch.cat((pts, torch.zeros(pts.size()[0], 1).cuda()), 1)
    return out

def volume_proj(volume, method="binary", half=False, nviews=None, views=None, tau=0.5, interp='nearest', return_vol=False):
    if views is None and nviews is None:
        raise ValueError("nviews or views should be specified.")

    if nviews is None:
        nviews = views.shape[0]

    projs = []
    if half:
        viewspace = torch.linspace(0, np.pi - (np.pi/nviews), nviews)
    else:
        viewspace = torch.linspace(0, 2*np.pi - (2*np.pi/nviews), nviews)

    for i, theta in enumerate(viewspace):
        if views is None:
            v = rotate_volume(volume, x=theta, interp=interp)
        else:
            v = rotate_volume(volume, x=views[i, 0], y=views[i,1], z=views[i,2], interp=interp)
        if method == "binary":
            img = binary_proj(v, tau)
        elif method == "depth":
            img = depth_proj(v)
        projs.append(img.unsqueeze(2))

    out = torch.cat(projs, 2)
    if return_vol:
        return out, v
    else:
        return out

