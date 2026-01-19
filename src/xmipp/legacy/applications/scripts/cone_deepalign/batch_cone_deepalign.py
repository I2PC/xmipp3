#!/usr/bin/env python3

import numpy as np
import os
import sys
import xmippLib
from time import time


if __name__=="__main__":

    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed
    checkIf_tf_keras_installed()
    fnXmdExp = sys.argv[1]
    fnLabels = sys.argv[2]
    fnODir = sys.argv[3]
    modelFn = sys.argv[4]
    numEpochs = int(sys.argv[5])
    Xdim = int(sys.argv[6])
    numOut = int(sys.argv[7])
    batch_size = int(sys.argv[8])
    gpuId = sys.argv[9]

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId
    

    from keras.callbacks import TensorBoard, ModelCheckpoint
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, concatenate, Subtract, SeparableConv2D, GlobalAveragePooling2D
    from keras.optimizers import *
    import keras
    from keras import callbacks
    from keras.callbacks import Callback
    from keras import regularizers
    from keras.models import load_model
    import tensorflow as tf


    class EarlyStoppingByLossVal(Callback):
        def __init__(self, monitor='val_loss', value=0.01, verbose=0):
            super(Callback, self).__init__()
            self.monitor = monitor
            self.value = value
            self.verbose = verbose

        def on_epoch_end(self, epoch, logs={}):
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

            if current < self.value:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                self.model.stop_training = True


    class DataGenerator(keras.utils.Sequence):
        'Generates data for Keras'
        def __init__(self, list_IDs_zeros, list_IDs_ones, labels, batch_size, dim, shuffle, pathsExp):
            'Initialization'
            self.dim = dim
            self.batch_size = batch_size
            self.labels = labels
            self.list_IDs_zeros = list_IDs_zeros
            self.list_IDs_ones = list_IDs_ones
            self.shuffle = shuffle
            self.pathsExp = pathsExp
            self.on_epoch_end()

        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor((len(self.labels)) / self.batch_size))

        def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch
            indexes_zeros = self.indexes_zeros[index*self.batch_size:(index+1)*self.batch_size]
            indexes_ones = self.indexes_ones[index*self.batch_size:(index+1)*self.batch_size]

            # Find list of IDs
            #list_IDs_temp = [self.list_IDs[k] for k in indexes]
            list_IDs_temp = []
            for i in range(int(self.batch_size//2)):
                list_IDs_temp.append(indexes_zeros[i])
            for i in range(int(self.batch_size//2)):
                list_IDs_temp.append(indexes_ones[i])

            # Generate data
            Xexp, y = self.__data_generation(list_IDs_temp)

            return Xexp, y

        def on_epoch_end(self):
            'Updates indexes after each epoch'
            self.indexes_zeros = self.list_IDs_zeros
            self.indexes_ones = self.list_IDs_ones
            if self.shuffle == True:
                np.random.shuffle(self.indexes_zeros)
                np.random.shuffle(self.indexes_ones)

        def __data_generation(self, list_IDs_temp):
            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
            # Initialization
            Xexp = np.zeros((self.batch_size,self.dim,self.dim,1),dtype=np.float64)
            y = np.empty((self.batch_size), dtype=np.int64)

            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                Iexp = np.reshape(xmippLib.Image(self.pathsExp[ID]).getData(),(self.dim,self.dim,1))
                Iexp = (Iexp-np.mean(Iexp))/np.std(Iexp)
                Xexp[i,] = Iexp

                # Store class
                y[i] = self.labels[ID]

            return Xexp, y


    def constructModel(Xdim, numOut):
        inputLayer = Input(shape=(Xdim,Xdim,1), name="input")

        #Network model
        L = Conv2D(16, (int(Xdim/3), int(Xdim/3)), activation="relu") (inputLayer) #33 filter size before
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Conv2D(32, (int(Xdim/10), int(Xdim/10)), activation="relu") (L) #11 filter size before
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Conv2D(64, (int(Xdim/20), int(Xdim/20)), activation="relu") (L) #5 filter size before
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Dropout(0.2)(L)
        L = Flatten() (L)
        L = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(L)
        L = BatchNormalization()(L)

        if numOut>2:
            L = Dense(numOut, name="output", activation="softmax") (L)
        elif numOut==2:
            L = Dense(1, name="output", activation="sigmoid") (L)
        return Model(inputLayer, L)


    def createValidationData(pathsExp, labels_vector, numOut, percent=0.1):
        sizeValData = int(round(len(pathsExp)*percent))
        val_img_exp = []
        val_labels = []
        if numOut>2:
            for i in range(sizeValData):
                k = np.random.randint(0,len(pathsExp))
                Iexp = xmippLib.Image(pathsExp[k])
                Iexp = np.reshape(Iexp.getData(),(Xdim,Xdim,1))
                Iexp = (Iexp-np.mean(Iexp))/np.std(Iexp)
                val_img_exp.append(Iexp)
                val_labels.append(labels_vector[k])
                del pathsExp[k]
                del labels_vector[k]
        elif numOut==2:
            labels = np.array(labels_vector)
            vectorOnes = np.where(labels==1)
            numberOnes = int(len(vectorOnes[0])*percent)
            vectorZeros = np.where(labels==0)
            numberZeros = int(len(vectorZeros[0])*percent)
            if numberZeros>numberOnes:
                numberZeros= numberOnes
            elif numberOnes>numberZeros:
                numberOnes=numberZeros
            for i in range(numberOnes):
                labels = np.array(labels_vector)
                vectorOnes = np.where(labels==1)
                vectorOnes = vectorOnes[0]
                k = np.random.randint(0,len(vectorOnes))
                k = vectorOnes[k]
                Iexp = xmippLib.Image(pathsExp[k])
                Iexp = np.reshape(Iexp.getData(),(Xdim,Xdim,1))
                Iexp = (Iexp-np.mean(Iexp))/np.std(Iexp)
                val_img_exp.append(Iexp)
                val_labels.append(labels_vector[k])
                del pathsExp[k]
                del labels_vector[k]

            for i in range(numberZeros):
                labels = np.array(labels_vector)
                vectorZeros = np.where(labels==0)
                vectorZeros = vectorZeros[0]
                k = np.random.randint(0,len(vectorZeros))
                k = vectorZeros[k]
                Iexp = xmippLib.Image(pathsExp[k])
                Iexp = np.reshape(Iexp.getData(),(Xdim,Xdim,1))
                Iexp = (Iexp-np.mean(Iexp))/np.std(Iexp)
                val_img_exp.append(Iexp)
                val_labels.append(labels_vector[k])
                del pathsExp[k]
                del labels_vector[k]

        return np.asarray(val_img_exp).astype('float64'), np.asarray(val_labels).astype('int64')

    mdExp = xmippLib.MetaData(fnXmdExp)
    Nexp = mdExp.size()
    labels = np.loadtxt(fnLabels)

    #To generate data for validation set
    pathsExp = []
    labels_vector = []
    cont=0
    allExpFns = mdExp.getColumnValues(xmippLib.MDL_IMAGE)
    for fnExp in allExpFns:
        pathsExp.append(fnExp)
        labels_vector.append(int(labels[cont]))
        cont+=1
    print(cont, len(labels_vector))

    x_val, y_val = createValidationData(pathsExp, labels_vector, numOut, 0.2)

    # Parameters
    params = {'dim': Xdim, 'batch_size': batch_size, 'shuffle': True, 'pathsExp': pathsExp}
    # Datasets
    list_IDs_zeros = np.where(np.array(labels_vector)==0)
    list_IDs_ones = np.where(np.array(labels_vector)==1)
    list_IDs_zeros = list_IDs_zeros[0]
    list_IDs_ones = list_IDs_ones[0]
    list_IDs_zeros_orig = list_IDs_zeros
    list_IDs_ones_orig = list_IDs_ones
    lenTotal = len(list_IDs_zeros)+len(list_IDs_ones)
    if len(list_IDs_zeros)<lenTotal:
        for i in range((lenTotal//len(list_IDs_zeros))-1):
            list_IDs_zeros = np.append(list_IDs_zeros, list_IDs_zeros_orig)
        list_IDs_zeros = np.append(list_IDs_zeros, list_IDs_zeros[0:(lenTotal%len(list_IDs_zeros))])
    if len(list_IDs_ones)<lenTotal:
        for i in range((lenTotal//len(list_IDs_ones))-1):
            list_IDs_ones = np.append(list_IDs_ones, list_IDs_ones_orig)
        list_IDs_ones = np.append(list_IDs_ones, list_IDs_ones[0:(lenTotal%len(list_IDs_ones))])
    print(len(list_IDs_zeros), len(list_IDs_ones))
    print(len(pathsExp), batch_size, round(len(pathsExp)/batch_size))
    labels = labels_vector
    # Generator
    training_generator = DataGenerator(list_IDs_zeros, list_IDs_ones, labels, **params)

    print('Training')
    start_time = time()
    model = constructModel(Xdim, numOut)


    name_model = os.path.join(fnODir, modelFn+'.h5')
    
    callbacks_list = [callbacks.ModelCheckpoint(filepath=name_model, monitor='val_loss', save_best_only=True),
    		      EarlyStoppingByLossVal(monitor='val_loss', value=0.05)]

    model.summary()
    adam_opt = Adam(lr=0.001)
    if numOut>2:
        model.compile(loss='sparse_categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])
    elif numOut==2:
        model.compile(loss='mean_absolute_error', optimizer=adam_opt, metrics=['accuracy'])

    steps = round(len(pathsExp)/batch_size)
    history = model.fit_generator(generator = training_generator, steps_per_epoch = steps, epochs=numEpochs, verbose=1, validation_data = (x_val, y_val), callbacks=callbacks_list, workers=4, use_multiprocessing=True)    #AJ probar estas cosas de multiprocessing
    model.save(name_model)
    elapsed_time = time() - start_time
    print("Time in training model: %0.10f seconds." % elapsed_time)

    model = load_model(name_model)
    Ypred = model.predict(x_val)
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_val, Ypred)
    print("Final model mean absolute error val_loss", mae)
    f = open (os.path.join(fnODir, modelFn+'.txt'),'w')
    f.write(str(mae))
    f.close()



