#!/usr/bin/env python3

import math
import numpy as np
from operator import itemgetter
import os
import sys
import xmippLib
from time import time
from scipy.ndimage import shift, rotate

if __name__ == "__main__":

    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed

    checkIf_tf_keras_installed()
    fnXmdExp = sys.argv[1]
    fnModel = sys.argv[2]
    sigma = float(sys.argv[3])
    numEpochs = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    gpuId = sys.argv[6]
    numModels = int(sys.argv[7])
    learning_rate = float(sys.argv[8])
    patience = int(sys.argv[9])
    pretrained = sys.argv[10]
    if pretrained == 'yes':
        fnPreModel = sys.argv[11]

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    from keras.callbacks import TensorBoard, ModelCheckpoint
    from keras.models import Model
    from keras.layers import Input, Conv2D, BatchNormalization, Dense, concatenate, \
        Activation, GlobalAveragePooling2D, Add
    import keras
    from keras.models import load_model
    import tensorflow as tf


    class DataGenerator(keras.utils.all_utils.Sequence):
        """Generates data for fnImgs"""

        def __init__(self, fnImgs, labels, sigma, batch_size, dim, shifts, readInMemory):
            """Initialization"""
            self.fnImgs = fnImgs
            self.labels = labels
            self.sigma = sigma
            self.batch_size = batch_size
            if self.batch_size > len(self.fnImgs):
                self.batch_size = len(self.fnImgs)
            self.dim = dim
            self.readInMemory = readInMemory
            self.on_epoch_end()
            self.shifts = shifts


            # Read all data in memory
            if self.readInMemory:
                self.Xexp = np.zeros((len(self.labels), self.dim, self.dim, 1), dtype=np.float64)
                for i in range(len(self.labels)):
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[i]).getData(), (self.dim, self.dim, 1))
                    self.Xexp[i,] = (Iexp - np.mean(Iexp)) / np.std(Iexp)

        def __len__(self):
            """Denotes the number of batches per epoch"""
            num_batches = int(np.floor((len(self.labels)) / self.batch_size))
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
            self.indexes = [i for i in range(len(self.labels))]
            np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):
            """Generates data containing batch_size samples"""
            yvalues = np.array(itemgetter(*list_IDs_temp)(self.labels))
            yshifts = np.array(itemgetter(*list_IDs_temp)(self.shifts))

            # Functions to handle the data
            def get_image(fn_image):
                img = np.reshape(xmippLib.Image(fn_image).getData(), (self.dim, self.dim, 1))
                return (img - np.mean(img)) / np.std(img)

            def shift_image(img, shiftx, shifty, yshift):
                return shift(img, (shiftx-yshift[0], shifty-yshift[1], 0), order=1, mode='wrap')

            def rotate_image(img, angle):
                # angle in degrees
                return rotate(img, angle, order=1, mode='reflect', reshape=False)

            def R_rot(theta):
                return np.array([[1, 0, 0],
                                  [0, math.cos(theta), -math.sin(theta)],
                                  [0, math.sin(theta), math.cos(theta)]])

            def R_tilt(theta):
                return np.array([[math.cos(theta), 0, math.sin(theta)],
                                  [0, 1, 0],
                                  [-math.sin(theta), 0, math.cos(theta)]])

            def R_psi(theta):
                return np.array([[math.cos(theta), -math.sin(theta), 0],
                                  [math.sin(theta), math.cos(theta), 0],
                                  [0, 0, 1]])

            def euler_angles_to_matrix(angles, psi_rotation):
                Rx = R_rot(angles[0])
                Ry = R_tilt(angles[1] - math.pi / 2)
                Rz = R_psi(angles[2] + psi_rotation)
                return np.matmul(np.matmul(Rz, Ry), Rx)

            def matrix_to_rotation6d(mat):
                r6d = np.delete(mat, -1, axis=1)
                return np.array((r6d[0, 0], r6d[0, 1], r6d[1, 0], r6d[1, 1], r6d[2, 0], r6d[2, 1]))

            def euler_to_rotation6d(angles, psi_rotation):
                mat = euler_angles_to_matrix(angles, psi_rotation)
                return matrix_to_rotation6d(mat)

            def make_redundant(rep_6d):
                rep_6d = np.append(rep_6d, 2*rep_6d)
                for i in range(6):
                    j = (i+1) % 6
                    rep_6d = np.append(rep_6d, rep_6d[i]-rep_6d[j])
                for i in range(6):
                    j = (i + 3) % 6
                    rep_6d = np.append(rep_6d, rep_6d[i+6] - rep_6d[j])
                for i in range(6):
                    j = (i + 2) % 6
                    k = (i + 4) % 6
                    rep_6d = np.append(rep_6d, rep_6d[i]+rep_6d[j]-rep_6d[k])
                for i in range(6):
                    j = (i + 5) % 6
                    rep_6d = np.append(rep_6d, rep_6d[i] - rep_6d[j])
                for i in range(6):
                    j = (i + 4) % 6
                    rep_6d = np.append(rep_6d, rep_6d[i] - rep_6d[j])
                return rep_6d

            if self.readInMemory:
                Iexp = list(itemgetter(*list_IDs_temp)(self.Xexp))
            else:
                fnIexp = list(itemgetter(*list_IDs_temp)(self.fnImgs))
                Iexp = list(map(get_image, fnIexp))

            rX = self.sigma * np.random.uniform(-1, 1, size=self.batch_size)
            rY = self.sigma * np.random.uniform(-1, 1, size=self.batch_size)
            # Shift image a random amount of px in each direction
            Xexp = np.array(list((map(shift_image, Iexp, rX, rY, yshifts))))
            # Rotates image a random angle. Psi must be updated
            rAngle = 180 * np.random.uniform(-1, 1, size=self.batch_size)
            Xexp = np.array(list(map(rotate_image, Xexp, rAngle)))
            rAngle = rAngle * math.pi / 180
            yvalues = yvalues * math.pi / 180
            y_6d = np.array(list((map(euler_to_rotation6d, yvalues, rAngle))))
            y = np.array(list((map(make_redundant, y_6d))))

            return Xexp, y

    def conv_block(tensor, filters):
        # Convolutional block of RESNET
        x = Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(tensor)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=3)(x)

        x_res = Conv2D(filters, (1, 1), strides=(2, 2))(tensor)

        x = Add()([x, x_res])
        x = Activation('relu')(x)
        return x

    def constructModel(Xdim):
        """RESNET architecture"""
        inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")

        x = conv_block(inputLayer, filters=64)

        x = conv_block(x, filters=128)

        x = conv_block(x, filters=256)

        x = conv_block(x, filters=512)

        x = conv_block(x, filters=1024)

        x = GlobalAveragePooling2D()(x)

        x = Dense(42, name="output", activation="linear")(x)

        return Model(inputLayer, x)


    def get_labels(fnImages):
        """Returns dimensions, images, angles and shifts values from images files"""
        Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnImages)
        mdExp = xmippLib.MetaData(fnImages)
        fnImg = mdExp.getColumnValues(xmippLib.MDL_IMAGE)
        shiftX = mdExp.getColumnValues(xmippLib.MDL_SHIFT_X)
        shiftY = mdExp.getColumnValues(xmippLib.MDL_SHIFT_Y)
        rots = mdExp.getColumnValues(xmippLib.MDL_ANGLE_ROT)
        tilts = mdExp.getColumnValues(xmippLib.MDL_ANGLE_TILT)
        psis = mdExp.getColumnValues(xmippLib.MDL_ANGLE_PSI)

        label = []
        img_shift = []

        # For better performance, images are selected to be 'homogeneously' distributed in the sphere
        # 50 divisions with equal area
        numTiltDivs = 5
        numRotDivs = 10
        limits_rot = np.linspace(-180.01, 180, num=(numRotDivs+1))
        limits_tilt = np.zeros(numTiltDivs+1)
        limits_tilt[0] = -0.01
        for i in range(1, numTiltDivs+1):
            limits_tilt[i] = math.acos(1-2*(i/numTiltDivs))
        limits_tilt = limits_tilt*180/math.pi

        # Each particle is assigned to a division
        zone = [[] for _ in range((len(limits_tilt)-1)*(len(limits_rot)-1))]
        i = 0
        for r, t, p, sX, sY in zip(rots, tilts, psis, shiftX, shiftY):
            label.append(np.array((r, t, p)))
            img_shift.append(np.array((sX, sY)))
            region_rot = np.digitize(r, limits_rot, right=True) - 1
            region_tilt = np.digitize(t, limits_tilt, right=True) - 1
            # Region index
            region_idx = region_rot * (len(limits_tilt)-1) + region_tilt
            zone[region_idx].append(i)
            i += 1

        return Xdim, fnImg, label, zone, img_shift

    Xdims, fnImgs, labels, zones, shifts = get_labels(fnXmdExp)
    start_time = time()

    # Train-Validation sets
    if numModels == 1:
        lenTrain = int(len(fnImgs)*0.8)
        lenVal = len(fnImgs)-lenTrain
    else:
        lenTrain = int(len(fnImgs) / 3)
        lenVal = int(len(fnImgs) / 12)

    elements_zone = int((lenVal+lenTrain)/len(zones))

    for index in range(numModels):
        # chooses equal number of particles for each division
        random_sample = np.random.choice(range(0, len(fnImgs)), size=lenTrain+lenVal, replace=False)

        training_generator = DataGenerator([fnImgs[i] for i in random_sample[0:lenTrain]],
                                           [labels[i] for i in random_sample[0:lenTrain]],
                                           sigma, batch_size, Xdims, shifts, readInMemory=False)
        validation_generator = DataGenerator([fnImgs[i] for i in random_sample[lenTrain:lenTrain+lenVal]],
                                             [labels[i] for i in random_sample[lenTrain:lenTrain+lenVal]],
                                             sigma, batch_size, Xdims, shifts, readInMemory=False)

        if pretrained == 'yes':
            model = load_model(fnPreModel, compile=False)
        else:
            model = constructModel(Xdims)

        adam_opt = tf.keras.optimizers.Adam(lr=learning_rate)
        model.summary()

        model.compile(loss='mean_squared_error', optimizer=adam_opt)
        save_best_model = ModelCheckpoint(fnModel + str(index) + ".h5", monitor='val_loss',
                                          save_best_only=True)
        patienceCallBack = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)


        history = model.fit_generator(generator=training_generator, epochs=numEpochs,
                                      validation_data=validation_generator, callbacks=[save_best_model, patienceCallBack])

    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)
