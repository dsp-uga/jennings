#!/usr/bin/env python
"""
./LucasKanade.py data/box/box
./LucasKanade.py data/office/office
./LucasKanade.py data/rubic/rubic
./LucasKanade.py data/sphere/sphere
"""
from skimage.color import rgb2grey
import imageio
from scipy.ndimage.filters import gaussian_filter
#
from pyoptflow import LucasKanade, getPOI, gaussianWeight
from pyoptflow.io import getimgfiles
from pyoptflow.plots import compareGraphsLK
data_u = list();
data_v = list()

u_vector = "u_vector"
v_vector ="v_vector"
def demo(stem, kernel=5,Nfilter=7):
    flist ,ext= getimgfiles(stem)
    ext =ext
    print(flist[0])
    #%% priming read
    im1 = imageio.imread(flist[1])
    if im1.ndim>2:
        im1 = rgb2grey(im1)
    Y,X = im1.shape
#%% evaluate the first frame's POI
    POI = getPOI(X,Y,kernel)
#% get the weights
    W = gaussianWeight(kernel)
#%% loop over all images in directory
    for i in range(1,len(flist)):
        print(flist[i])
        print(flist[i])
        im2 = imageio.imread(flist[i])
        if im2.ndim>2:
            im2 = rgb2grey(im2)

        im2 = gaussian_filter(im2, Nfilter)

        V = LucasKanade(im1, im2, POI, W, kernel)

        #u,v = compareGraphsLK(im1, im2, POI, V)
        data_u.append(POI)
        data_v.append(V)
        im1 = im2
    np.save(u_vector,data_u)
    np.save(v_vector,data_v)

if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='Pure Python Horn Schunck Optical Flow')
    p.add_argument('stem',help='path/stem of files to analyze')
    p = p.parse_args()

    demo(p.stem)
