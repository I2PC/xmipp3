#!/usr/bin/env python3

import numpy as np
from operator import itemgetter
import os
import sys
from time import time
from scipy.ndimage import shift
from scipy.signal import correlate2d
import xmippLib
from xmipp_base import *


def gaussian_mask(size, sigma):
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    x, y = np.meshgrid(x, y)
    mask = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    mask /= np.sum(mask)
    return mask

def lowpass_filter_mask(shape, cutoff_frequency):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2  # Center of the frequency domain

    # Create a 2D grid of distances from the center of the Fourier domain
    y, x = np.ogrid[:rows, :cols]
    distance_from_center = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    mask = np.fft.ifftshift((distance_from_center <= cutoff_frequency).astype(float))
    return mask

class ScriptDeepCenterTrain(XmippScript):

    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Train a deep center model')
        ## params
        self.addParamsLine(' -i <metadata>                : xmd file with the list of images')
        self.addParamsLine(' --omodel <fnModel>           : Model filename')
        self.addParamsLine('[--sigma <s=5>]               : Shift sigma (px)')
        self.addParamsLine('[--maxEpochs <N=500>]         : Max. number of epochs')
        self.addParamsLine('[--batchSize <N=8>]           : Batch size')
        self.addParamsLine('[--gpu <id=0>]                : GPU Id')
        self.addParamsLine('[--learningRate <lr=0.0001>]  : Learning rate')
        self.addParamsLine('[--precision <s=0.5>]         : Alignment precision measured in pixels')

    def run(self):
        fnXmd = self.getParam("-i")
        fnModel = self.getParam("--omodel")
        sigma = float(self.getParam("--sigma"))
        maxEpochs = int(self.getParam("--maxEpochs"))
        batch_size = int(self.getParam("--batchSize"))
        gpuId = self.getParam("--gpu")
        learning_rate = float(self.getParam("--learningRate"))
        precision = float(self.getParam("--precision"))

        from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed

        if not gpuId.startswith('-1'):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpuId
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
        checkIf_tf_keras_installed()

        import tensorflow as tf
        from keras.models import Model
        from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Activation
        # from keras.applications.efficientnet import EfficientNetB2
           # https://keras.io/api/applications/
        import keras

        def saveImage(Iarray, fnOut):
            I=Image()
            I.setData(Iarray)
            I.write(fnOut)

        class DataGenerator(keras.utils.all_utils.Sequence):
            """Generates data for fnImgs"""

            def __init__(self, mdExp, sigma, batch_size, dim):
                """Initialization"""
                self.fnImgs = fnImgs
                self.sigma = sigma
                self.batch_size = batch_size
                if self.batch_size > len(self.fnImgs):
                    self.batch_size = len(self.fnImgs)
                self.dim = dim
                self.on_epoch_end()

                # Read all data in memory
                self.Xexp = np.zeros((len(self.fnImgs), self.dim, self.dim, 1), dtype=np.float64)
                img=xmippLib.Image()
                id=0
                for i in mdExp:
                    img.readApplyGeo(mdExp, i)
                    Iexp = np.reshape(img.getData(), (self.dim, self.dim, 1))
                    # M = np.max(Iexp)
                    # m = np.min(Iexp)
                    # self.Xexp[i,] = 255*(Iexp-m)/(M-m) # Scale between 0 and 255 for EfficientNet
                    self.Xexp[id,] = (Iexp-np.mean(Iexp))/np.std(Iexp)
                    id+=1

            def __len__(self):
                """Denotes the number of batches per epoch"""
                num_batches = int(np.floor((len(self.fnImgs)) / self.batch_size))
                return num_batches

            def __getitem__(self, index):
                """Generate one batch of data"""
                # Generate indexes of the batch
                indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
                # Find list of IDs
                list_IDs_temp = []
                for i in range(int(self.batch_size)):
                    list_IDs_temp.append(indexes[i])
                # Generate data
                Xexp, y = self.__data_generation(list_IDs_temp)

                return Xexp, y

            def on_epoch_end(self):
                """Updates indexes after each epoch"""
                self.indexes = [i for i in range(len(self.fnImgs))]
                np.random.shuffle(self.indexes)

            def __data_generation(self, list_IDs_temp):
                """Generates data containing batch_size samples"""
                def shift_image(img, shiftx, shifty):
                    """Shifts image in X and Y"""
                    return shift(img, (shifty, shiftx, 0), order=1, mode='wrap')

                Iexp = list(itemgetter(*list_IDs_temp)(self.Xexp))
                # currentX = list(itemgetter(*list_IDs_temp)(self.currentX))
                # currentY = list(itemgetter(*list_IDs_temp)(self.currentY))

                # Data augmentation
                generator = np.random.default_rng()
                rX = self.sigma * generator.normal(0, 1, size=self.batch_size)
                rY = self.sigma * generator.normal(0, 1, size=self.batch_size)
                # Shift image a random amount of px in each direction
                Xexp = np.array(list((map(shift_image, Iexp, rX, rY))))
                #y = np.vstack((currentX-rX, currentY-rY)).T
                y = np.vstack((-rX, -rY)).T
                return Xexp, y

        def constructModel(Xdim):
            """EfficientNet+Dense"""
            inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")

            # Adapt from 1 channel to 3 channels
            # x = Conv2D(3, (1, 1), padding='same')(inputLayer)
            # base_model = EfficientNetB2(weights='imagenet', include_top=False, pooling="avg")
            # base_model.trainable=False
            # L = base_model(x)
            # L = Dense(32, activation="relu")(L)
            # L = Dense(8, activation="relu")(L)

            L = Conv2D(8, (3, 3), padding='same')(inputLayer)
            L = Activation(activation='relu')(L)
            L = MaxPooling2D()(L)

            L = Conv2D(16, (3, 3), padding='same')(L)
            L = Activation(activation='relu')(L)
            L = MaxPooling2D()(L)

            L = Conv2D(32, (3, 3), padding='same')(L)
            L = Activation(activation='relu')(L)
            L = MaxPooling2D()(L)

            # L = Conv2D(64, (3, 3), padding='same')(L)
            # L = Activation(activation='relu')(L)
            # L = MaxPooling2D()(L)

            L = Flatten()(L)

            L = Dense(2, name="output", activation="linear")(L)

            return Model(inputLayer, L)

        def trainModel(model, generator, modeprec, fnModel):
            adam_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            sys.stdout.flush()
            model.compile(loss='mean_absolute_error', optimizer=adam_opt)

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',  # Metric to be monitored
                factor=0.1,  # Factor by which the learning rate will be reduced. new_lr = lr * factor
                patience=10,  # Number of epochs with no improvement after which learning rate will be reduced
                min_lr=1e-8,  # Lower bound on the learning rate
            )

            for epoch in range(maxEpochs):
                start_time = time()
                history = model.fit(generator, epochs=1, verbose=0, callbacks=[reduce_lr])
                end_time = time()
                loss = history.history['loss'][-1]
                print("Epoch %d loss=%f trainingTime=%d" % (epoch, loss, int(end_time-start_time)), flush=True)
                if loss < modeprec:
                    break
            model.save(fnModel)

        Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnXmd)
        mdExp = xmippLib.MetaData(fnXmd)
        fnImgs = mdExp.getColumnValues(xmippLib.MDL_IMAGE)
        # currentX, currentY = computeShifts(fnImgs)
        # currentX = mdExp.getColumnValues(xmippLib.MDL_SHIFT_X)
        # currentY = mdExp.getColumnValues(xmippLib.MDL_SHIFT_Y)
        training_generator = DataGenerator(mdExp, sigma, batch_size, Xdim)

        model = constructModel(Xdim)
        model.summary()
        trainModel(model, training_generator, precision, fnModel)

if __name__ == '__main__':
    exitCode = ScriptDeepCenterTrain().tryRun()
    sys.exit(exitCode)
