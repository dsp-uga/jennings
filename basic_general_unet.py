import scipy as sp
import scipy.misc, scipy.ndimage.interpolation
import numpy as np
import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Input,merge, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D,Dropout,Conv2DTranspose
from keras.layers.normalization import BatchNormalization as bn
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras import regularizers 
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import sys
import h5py
import cv2
# creates a basic u-net based on 
#@inproceedings{ronneberger2015u,
#  title={U-net: Convolutional networks for biomedical image segmentation},
#  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
#  booktitle={International Conference on Medical image computing and computer-assisted intervention},
#  pages={234--241},
#  year={2015},
#  organization={Springer}
#}

#batch normalization 
#@inproceedings{ioffe2015batch,
#  title={Batch normalization: Accelerating deep network training by reducing internal covariate shift},
#  author={Ioffe, Sergey and Szegedy, Christian},
#  booktitle={International conference on machine learning},
#  pages={448--456},
#  year={2015}
#}

#Main deep learning API library used Keras - 
#@misc{chollet2015keras,
#  title={Keras},
#  author={Chollet, Fran\c{c}ois and others},
#  year={2015},
#  publisher={GitHub},
#  howpublished={\url{https://github.com/keras-team/keras}},
#}

#default cuda device - 0 
CUDA_VISIBLE_DEVICES = [0]

os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(x) for x in CUDA_VISIBLE_DEVICES])

#Dice coefficient to calculate the intersection over union

#Dice coeff
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# negative dice loss since we want the network to minimize it more
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# define the model
# each u net block is a series of two convolution operations followed by batch normalization
# after each block we use a dropout of 0.3
# In total the downsample and upsample blocks have 6 layers each with an intermediate transition block of  2 convolution layers with 512 feature maps
def UNet(input_shape,learn_rate=1e-3):
    l2_lambda = 0.0002
    DropP = 0.3
    kernel_size=3

    inputs = Input(input_shape)

    conv1 = Conv2D( 32, (kernel_size, kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(inputs)
    
    
    conv1 = bn()(conv1)
    
    conv1 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv1)

    conv1 = bn()(conv1)
    
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    pool1 = Dropout(DropP)(pool1)





    conv2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool1)
    
    conv2 = bn()(conv2)

    conv2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv2)

    conv2 = bn()(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    pool2 = Dropout(DropP)(pool2)



    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool2)

    conv3 = bn()(conv3)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv3)
    
    conv3 = bn()(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    pool3 = Dropout(DropP)(pool3)



    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool3)
    conv4 = bn()(conv4)
    
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv4)
    
    conv4 = bn()(conv4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    pool4 = Dropout(DropP)(pool4)



    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool4)
    
    conv5 = bn()(conv5)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv5)

    conv5 = bn()(conv5)
    
    up6 = concatenate([Conv2DTranspose(256,(2, 2), strides=(2, 2), padding='same')(conv5), conv4],name='up6', axis=3)

    up6 = Dropout(DropP)(up6)


    conv6 = Conv2D(256,(3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up6)
    
    conv6 = bn()(conv6)

    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv6)

    conv6 = bn()(conv6)

    up7 = concatenate([Conv2DTranspose(128,(2, 2), strides=(2, 2), padding='same')(conv6), conv3],name='up7', axis=3)

    up7 = Dropout(DropP)(up7)

    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up7)

    conv7 = bn()(conv7)
    
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv7)

    conv7 = bn()(conv7)

    up8 = concatenate([Conv2DTranspose(64,(2, 2), strides=(2, 2), padding='same')(conv7), conv2],name='up8', axis=3)

    up8 = Dropout(DropP)(up8)

    conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up8)

    conv8 = bn()(conv8)

    
    conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv8)

    conv8 = bn()(conv8)

    up9 = concatenate([Conv2DTranspose(32,(2, 2), strides=(2, 2), padding='same')(conv8), conv1],name='up9',axis=3)

    up9 = Dropout(DropP)(up9)

    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up9)
    
    conv9 = bn()(conv9)

    conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv9)
   
    conv9 = bn()(conv9)
   
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    return model


if __name__ == "__main__":
    if len(sys.argv) !=10:
        print("Usage: u_net_basic.py  <input_array> <ground_truth_array> <test_array> <epochs to train> <batch_size> <enter directory to save output images><enter dimension x> <enter dimension y><enter numchannels 1-gray,3 color>")
        exit(-1)
    dim_x=(int)(sys.argv[7])
    dim_y=(int)(sys.argv[8])
    dim_chan=(int)(sys.argv[9])
    model=UNet(input_shape=(dim_x,dim_y,dim_chan))
    print(model.summary())

    # accepts input array and ground truth and reshapes it to (n,512,512,1) where n is the number of input slices

    X_train=np.load(str(sys.argv[1]))
    X_train=X_train/255.0
    X_train=X_train.reshape(X_train.shape+(1,))
    y_train=np.load(str(sys.argv[2])).reshape(X_train.shape)
    cv2.imwrite('sampley_train_before.png',y_train[0]*255)
    y_train[y_train<2]=0
    y_train[y_train>0]=1
    cv2.imwrite('sampley_train_after.png',y_train[0]*255)
    cv2.imwrite('samplex_train.png',X_train[0]*255)
    epochs_x=(int)(sys.argv[4])
    print('done')

    batchsize=(int)(sys.argv[5])
    path=(str)(sys.argv[6])
    #training network using keras's model.fit api
    model.fit([X_train], [y_train],
                        batch_size=batchsize,
                        nb_epoch=epochs_x,
                        shuffle=True),
                       


    #load the testing array and save the output as numpy array as well as generate a list of .png files
    '''
    X_train=np.load(str(sys.argv[3]))
    X_train=X_train.reshape(X_train.shape+(1,))
    predict=model.predict([X_train],batch_size=4)
    for i in range(0,len(predict)):
        cv2.imwrite(path+"\predicted"+str(i)+".png",predict[i]*255)
    np.save("predicted",predict)
    '''
    # saves model to file
    model.save('basic_unet_dsp_p4.h5')
