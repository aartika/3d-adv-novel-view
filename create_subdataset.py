import os
import random
from shutil import copyfile

rootimg = './data/shapenetcore/masks'
rootvol = './data/shapenetcore/masks'

n_views=4
path='data/subdatasets/shapenetcore_{}/masks'.format(n_views)

for cat in ['03001627', '04090263']:
#for cat in os.listdir(rootimg):
    dir_img = os.path.join(rootimg, cat)
    for fn in os.listdir(dir_img):
        dst = os.path.join(path, cat, fn)
        os.makedirs(dst)
        N = random.sample(range(0, 9), n_views)
        for n in N:
            copyfile(os.path.join(dir_img, fn, 'v{}.png'.format(n)), os.path.join(dst, 'v{}.png'.format(n)))
        
    
