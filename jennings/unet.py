'''Provides a U-Net based model.

See:
    U-net: Convolutional networks for biomedical image segmentation
    Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas
'''

from pathlib import Path
from urllib import request

from keras import backend as K
from keras import regularizers
from keras.layers import concatenate, Conv2D, Dropout, Conv2DTranspose, Input, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam


def dice_coef(y_true, y_pred, smooth=1):
    '''A Keras function for the dice coefficient.
    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    '''A Keras function for the negative dice coeficient, to use as a loss.
    '''
    return -dice_coef(y_true, y_pred)


def _weights(path='./unet_celia_weights.h5', url=None, force_download=False):
    '''Download pretrained weights for the U-Net model.

    See:
        `jennings.unet.unet`

    Args:
        path: The path to the weights.
        url: Used to override the URL of the weights.
        force_download: Force a download even if the weights exist.

    Returns:
        path: The path to the weights.
    '''
    url = url or 'https://storage.googleapis.com/cbarrick/jennings/unet_celia_weights.h5'
    path = Path(path)

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with request.urlopen(url) as response:
            with path.open('wb') as fd:
                for line in response:
                    fd.write(line)

    return path


def unet(input_shape, lr=1e-5, l2_reg=0.0002, dropout_rate=0.3, kernel_size=3, pretrained=False):
    '''Construct a U-Net model with batch-norm, dropout, and 𝓁2 regularization.

    See:
        U-net: Convolutional networks for biomedical image segmentation.
        Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas

    Args:
        input_shape: The shape of the input images.
        lr: The learning rate for the Adam optimizer.
        l2_reg: The 𝓁2 regularization strength.
        dropout_rate: The regularization strength for dropout layers.
        kernel_size: The size of the convolution kernels.
        pretrained: Use pretrained weights for the celia segmentation dataset.

    Returns:
        A Keras model for U-Net.
    '''

    # Common arguments for all convolution layers.
    conv_args = dict(
        activation = 'relu',
        padding = 'same',
        kernel_regularizer = regularizers.l2(l2_reg),
    )

    inputs = Input(input_shape)

    conv1 = Conv2D(32, (kernel_size, kernel_size), **conv_args)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (kernel_size, kernel_size), **conv_args)(conv1)
    conv1 = BatchNormalization()(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(dropout_rate)(pool1)

    conv2 = Conv2D(64, (kernel_size, kernel_size), **conv_args)(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (kernel_size, kernel_size), **conv_args)(conv2)
    conv2 = BatchNormalization()(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(dropout_rate)(pool2)

    conv3 = Conv2D(128, (kernel_size, kernel_size), **conv_args)(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (kernel_size, kernel_size), **conv_args)(conv3)
    conv3 = BatchNormalization()(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(dropout_rate)(pool3)

    conv4 = Conv2D(256, (kernel_size, kernel_size), **conv_args)(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (kernel_size, kernel_size), **conv_args)(conv4)
    conv4 = BatchNormalization()(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)

    conv5 = Conv2D(512, (kernel_size, kernel_size), **conv_args)(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (kernel_size, kernel_size), **conv_args)(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], name='up6', axis=3)
    up6 = Dropout(dropout_rate)(up6)

    conv6 = Conv2D(256,(kernel_size, kernel_size), **conv_args)(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (kernel_size, kernel_size), **conv_args)(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], name='up7', axis=3)
    up7 = Dropout(dropout_rate)(up7)

    conv7 = Conv2D(128, (kernel_size, kernel_size), **conv_args)(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (kernel_size, kernel_size), **conv_args)(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], name='up8', axis=3)
    up8 = Dropout(dropout_rate)(up8)

    conv8 = Conv2D(64, (kernel_size, kernel_size), **conv_args)(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (kernel_size, kernel_size), **conv_args)(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2DTranspose(32,(2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], name='up9', axis=3)
    up9 = Dropout(dropout_rate)(up9)

    conv9 = Conv2D(32, (kernel_size, kernel_size), **conv_args)(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (kernel_size, kernel_size), **conv_args)(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])

    if pretrained:
        weights = _weights()
        model.load_weights(weights)

    return model
