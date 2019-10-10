# Code to save voxels borrowed from https://github.com/matheusgadelha/PrGAN. 

import torch
from torch.utils.data.dataset import Dataset

import numpy as np
import os
import glob
import matplotlib
import skimage.io as io
from skimage.transform import resize
from torchvision.utils import save_image


def write_obj_from_array(path, volume):
    pts, faces = volume_to_cubes(volume, threshold=0.5)
    write_cubes_obj(path, pts, faces)


def save_voxels(path, voxels, as_mesh=False, init_idx=0):
    os.makedirs(path, exist_ok=True)
    for i in range(voxels.size(0)):
        #check if voxel has multiple channels
        if voxels.size(1) == 1:
            volume = voxels.detach().cpu().numpy()[i, 0, :, :, :]
            if as_mesh:
                write_obj_from_array(os.path.join(path, "v{}.obj".format(str(i+init_idx).zfill(4))), 
                        volume)
            else:
                volume = (volume > 0.5).astype('int8')
                np.save(os.path.join(path, "v{}.npy".format(str(i+init_idx).zfill(4))), volume)
        else:

            #save parts separately
            for j in range(voxels.size(1)):
                volume = voxels.detach().cpu().numpy()[i, j, :, :, :]
                if as_mesh:
                    write_obj_from_array(os.path.join(path, "v{}_p{}.obj".format(str(i+init_idx).zfill(4), j+1)), 
                            volume)
                else:
                    volume = (volume > 0.5).astype('int8')
                    np.save(os.path.join(path, "v{}_p{}.npy".format(str(i+init_idx).zfill(4), j+1)), volume)

            #save complete volume
            volume = voxels.sum(1).detach().cpu().numpy()[i, :, :, :]
            if as_mesh:
                write_obj_from_array(os.path.join(path, "v{}_all.obj".format(str(i+init_idx).zfill(4))), 
                        volume)
            else:
                volume = (volume > 0.5).astype('int8')

def volume_to_cubes(volume, threshold=0, dim=[2., 2., 2.]):
    o = np.array([-dim[0]/2., -dim[1]/2., -dim[2]/2.])
    step = np.array([dim[0]/volume.shape[0], dim[1]/volume.shape[1], dim[2]/volume.shape[2]])
    points = []
    faces = []
    for x in range(1, volume.shape[0]-1):
        for y in range(1, volume.shape[1]-1):
            for z in range(1, volume.shape[2]-1):
                pos = o + np.array([x, y, z]) * step
                if volume[x, y, z] > threshold:
                    vidx = len(points)+1
                    POS = pos + step*0.95
                    xx = pos[0]
                    yy = pos[1]
                    zz = pos[2]
                    XX = POS[0]
                    YY = POS[1]
                    ZZ = POS[2]
                    points.append(np.array([xx, yy, zz]))
                    points.append(np.array([xx, YY, zz]))
                    points.append(np.array([XX, YY, zz]))
                    points.append(np.array([XX, yy, zz]))
                    points.append(np.array([xx, yy, ZZ]))
                    points.append(np.array([xx, YY, ZZ]))
                    points.append(np.array([XX, YY, ZZ]))
                    points.append(np.array([XX, yy, ZZ]))
                    faces.append(np.array([vidx, vidx+1, vidx+2, vidx+3]))
                    faces.append(np.array([vidx, vidx+4, vidx+5, vidx+1]))
                    faces.append(np.array([vidx, vidx+3, vidx+7, vidx+4]))
                    faces.append(np.array([vidx+6, vidx+2, vidx+1, vidx+5]))
                    faces.append(np.array([vidx+6, vidx+5, vidx+4, vidx+7]))
                    faces.append(np.array([vidx+6, vidx+7, vidx+3, vidx+2]))
    return points, faces

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



def write_cubes_obj(path, points, faces):
    f = open(path, 'w')
    for p in points:
      f.write("v {} {} {}\n".format(p[0], p[1], p[2]))
    for q in faces:
      f.write("f {} {} {} {}\n".format(q[0], q[1], q[2], q[3]))



