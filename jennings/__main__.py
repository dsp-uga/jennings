import numpy as np

import jennings
from jennings.loader import train_x, train_y, test_x
from jennings.unet import unet
from jennings.features import first_frame, extract_features
import tarfile
import cv2
import os
from jennings import loader

def pad(a, shape):
    '''Mirror pads an array to the given shape.

    Args:
        a:
            The array to pad, with the same number of dimensions as `shape`.
        shape:
            The shape of the padded array.

    Returns:
        A zero-padded array with the contents of a.
    '''
    pads = []
    for dim, target in zip(a.shape, shape):
        avg = (target - dim) / 2
        before = int(np.floor(avg))
        after = int(np.ceil(avg))
        pads.append((before, after))
    return np.pad(a, pads, 'reflect')


def unpad(a, shape):
    '''Slices a subarray of the given shape from the center of `a`.

    This is the inverse operation of `pad`.

    Args:
        a:
            The array to pad, with the same number of dimensions as `shape`.
        shape:
            The shape of the subarray.

    Returns:
        A view of the given shape taken from the center of `a`.
    '''
    slices = []
    for dim, target in zip(a.shape, shape):
        avg = (dim - target) / 2
        start = int(np.floor(avg))
        stop = dim - int(np.ceil(avg))
        slices.append(slice(start, stop))
    return a[slices]


def train(**kwargs):
    '''Train a U-Net model on the celia segmentation data set.

    Args:
        kwargs:
            Forwarded to Keras's `Model.fit`.

    Returns:
        A fitted Keras model.
    '''

    print('==> Loading train set')
    x = list(train_x())
    y = list(train_y())

    print('==> Extracting features from train set')
    features = [first_frame]
    x = [extract_features(im, features) for im in x]

    print('==> Padding train set to common size')
    max_height = max(im.shape[0] for im in x)
    max_width = max(im.shape[1] for im in x)
    n_features = len(features)
    full_shape = (max_height, max_width, n_features)
    x = np.stack([pad(im, full_shape) for im in x])
    y = np.stack([pad(im, full_shape) for im in y])

    print('==> Fitting the model')
    model = unet(full_shape)
    model.fit(x, y, **kwargs)
    return model


def test(pretrained=False, **kwargs):
    '''Train and test a U-Net model on the celia segmentation data set.

    Args:
        pretrained:
            If true, use pretrained weights rather than training a model.
        kwargs:
            Forwarded to `train` when not using a pretrained model.

    Returns:
        Predictions for the celia segmentation test set.
    '''

    print('==> Loading test set')
    x = list(test_x())

    print('==> Extracting features from test set')
    features = [first_frame]
    x = [extract_features(im, features) for im in x]

    print('==> Padding test set to common size')
    og_shapes = [im.shape for im in x]
    max_height = max(im.shape[0] for im in x)
    max_width = max(im.shape[1] for im in x)
    n_features = len(features)
    full_shape = (max_height, max_width, n_features)
    x = np.stack([pad(im, full_shape) for im in x])

    print('==> Constructing the model')
    if pretrained:
        model = unet(full_shape, pretrained=True)
    else:
        # The input shape for the test set may differ from the train set,
        # so we reconstruct the model after training with the test shape.
        model = train(**kwargs)
        model.save_weights('./unet_weights.h5')
        model = unet(full_shape)
        model.load_weights('./unet_weights.h5')

    print('==> Making predictions')
    y = model.predict(x)

    print('==> Unpadding predictions to original size')
    shapes = [(shape[0], shape[1], 1) for shape in og_shapes]
    y = [unpad(im, shape) for im, shape in zip(y, shapes)]
    return y



def generate_tar(dataset, filename_list, tar_file_name, extension=".png"):
    tar = tarfile.open(tar_file_name + ".tar.gz", "w:gz")
    for entry,fname in zip(dataset, filename_list):
        entry[entry == 1] = 2
        cv2.imwrite(fname + extension, entry)
        tar.addfile(tarfile.TarInfo(fname + extension),open(fname + extension))
        os.remove(fname + extension)
    tar.close();
    
def main():
   data = test(pretrained = True, verbose = 2)
   names = list(loader.test_names())
   generate_tar(data,names,"submission")

if __name__ == '__main__':
    main()
