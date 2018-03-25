#!/usr/bin/env python
"""
./HornSchunck.py data/box/box
./HornSchunck.py data/office/office
./HornSchunck.py data/rubic/rubic
./HornSchunck.py data/sphere/sphere
"""
from skimage.color import rgb2grey
from scipy.ndimage.filters import gaussian_filter
import imageio
from matplotlib.pyplot import show,close,clf,cla
#
from pyoptflow import HornSchunck, getimgfiles
from pyoptflow.plots import compareGraphs
import numpy as np;
import glob;
FILTER = 7

def demo(stem):
    data_u=list()
    data_v = list()
    flist,ext = getimgfiles(stem)

    for i in range(0,len(flist)-1):
        print(i)
        im1 = imageio.imread(flist[i])
        if im1.ndim>2:
            im1 = rgb2grey(im1)
 #       Iold = gaussian_filter(Iold,FILTER)

        im2 = imageio.imread(flist[i+1])
        if im2.ndim>2:
            im2 = rgb2grey(im2)
#        Inew = gaussian_filter(Inew,FILTER)

        U,V = HornSchunck(im1, im2, 1., 1000)
        #compareGraphs(U,V, im2)

        data_u.append(U)
        data_v.append(V)

        im1 = im2

    return data_u,data_v


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='Pure Python Horn Schunck Optical Flow')
    p.add_argument('stem',help='path/stem of files to analyze')
    p = p.parse_args()

    U,V = demo(p.stem)
    U = np.array(U)
    V = np.array(V)
    np.save("U.npy",U)
    np.save("V.npy",V)
    show()
    close()
