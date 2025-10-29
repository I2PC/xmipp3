# **************************************************************************
# *
# * Authors:  Ruben Sanchez (rsanchez@cnb.csic.es), April 2017
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

from __future__ import print_function

from six.moves import range
import sys, os, gc

import numpy as np
import scipy
import random

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef
import xmippLib

import keras
import tensorflow as tf
from keras import backend as K
from .deepConsensus_networkDef import main_network, DESIRED_INPUT_SIZE
tf_intarnalError= tf.errors.InternalError

BATCH_SIZE= 64
CHECK_POINT_AT= 50 #In batches

WRITE_TEST_SCORES= True


def loadNetShape(netDataPath):
  '''
      netDataPath= self._getExtraPath("nnetData")
  '''
  netInfoFname = os.path.join(netDataPath, "nnetInfo.txt")
  if not os.path.isfile(netInfoFname):
    return None
  with open(netInfoFname) as f:
    lines = f.readlines()
    dataShape = tuple([int(elem) for elem in lines[0].split()[1:]])
    nTrue = int(lines[1].split()[1])
    nModels = int(lines[2].split()[1])

  return dataShape, nTrue, nModels
    

def writeNetShape(netDataPath, shape, nTrue, nModels):
    '''
        netDataPath= self._getExtraPath("nnetData")
    '''
    netInfoFname = os.path.join(netDataPath, "nnetInfo.txt")
    if not os.path.exists(netDataPath):
      os.makedirs(netDataPath )
    with open(netInfoFname, "w" ) as f:
        f.write("inputShape: %d %d %d\ninputNTrue: %d\nnModels: %d" % (shape+(nTrue, nModels)))

def writeNetAccuracy(netDataPath, val_acc):
  '''
    netDataPath= self._getExtraPath("nnetData")
  '''
  netAccFname = os.path.join(os.path.dirname(netDataPath), "netValAcc.txt")
  os.makedirs(os.path.dirname(netAccFname), exist_ok=True)
  with open(netAccFname, "w") as f:
    f.write("val_acc: %f" %val_acc)

def loadANDwriteNetAccuracy(netDataPath, nModels):
  '''
      netDataPath= self._getExtraPath("nnetData")
  '''
  list_Acc = []
  for n in range(nModels):
    checkPointsName = os.path.join(netDataPath, "tfchkpoints_%d"%n)
    valAccFn = os.path.join(checkPointsName, "netValAcc.txt")
    with open(valAccFn) as f:
      line = f.readline()
      accuracy = float(line.split()[1])
      list_Acc.append(accuracy)

  mean_acc = np.mean(list_Acc)
  print('Mean validation accuracy %f of n %d models' %(mean_acc, nModels))

  netMeanAccFname = os.path.join(netDataPath, "netsMeanValAcc.txt")
  os.makedirs(os.path.dirname(netDataPath), exist_ok=True)
  with open(netMeanAccFname, "w") as f:
    f.write("mean_val_acc: %f" % mean_acc)
        
class DeepTFSupervised(object):
  def __init__(self, numberOfThreads, rootPath, numberOfModels=1, effective_data_size=-1):
    '''
      @param numberOfThreads: int or None if use gpu
      @param rootPath: str. Root directory where neural net data will be saved.
                            Generally "extra/nnetData/"
                                                      tfchkpoints/
                                                      tflogs/
                                                      ...
     @param modelNum: int. The number of models that will be trained on ensemble

    '''
    self.numberOfThreads= numberOfThreads
    self.rootPath= rootPath
    self.numberOfModels= numberOfModels
    self.effective_data_size= effective_data_size
    
    checkPointsName= os.path.join(rootPath,"tfchkpoints_%d")
    for modelNum in range(self.numberOfModels): 
      if not os.path.exists(checkPointsName%(modelNum) ):
        os.makedirs(checkPointsName%(modelNum) )

    self.checkPointsNameTemplate= os.path.join(checkPointsName,"deepModel.hdf5")

    self.nNetModel= None
    self.optimizer= None

  def createNet(self, xdim, ydim, num_chan, nData=2**12, learningRate=1e-4, l2RegStrength=1e-5):
    '''
      @param xdim: int. height of images
      @param ydim: int. width of images
      @param num_chan: int. number of channels of images
      @param nData: number of positive cases expected in data. Not needed
    '''
    print ("Creating net.")
    self.nNetModel, self.optimizerFunLambda = main_network( (xdim, ydim, num_chan),  nData= nData, l2RegStrength= l2RegStrength)
    self.optimizer= self.optimizerFunLambda(learningRate)
    self.nNetModel.compile( self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

  def loadNNet(self, kerasModelFname, keepTraining=True, learningRate=1e-4, l2RegStrength=1e-5):
    self.nNetModel= keras.models.load_model( kerasModelFname , custom_objects={"DESIRED_INPUT_SIZE":DESIRED_INPUT_SIZE})
    self.optimizer= self.nNetModel.optimizer
    if keepTraining:
        K.set_value(self.nNetModel.optimizer.lr, learningRate)    
        for layer in self.nNetModel.layers:
            if hasattr(layer, "kernel_regularizer"):
                if hasattr(layer.kernel_regularizer, "l2"):
                    layer.kernel_regularizer.l2= l2RegStrength
        self.nNetModel.compile( self.nNetModel.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
  def startSessionAndInitialize(self):
    '''
    '''
    if self.numberOfThreads is None:
      physical_devices = tf.config.experimental.list_physical_devices('GPU')
      assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
      config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
      self.session = tf.Session(config=config)
    else:
      self.session= tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=self.numberOfThreads,
                                                     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7),
                                                     allow_soft_placement = True))
    K.set_session(self.session)
    return self.session

  def closeSession(self):
    '''
      Closes a tensorflow connection and related objects.

    '''
    K.clear_session()
    
  def trainNet(self, nEpochs, dataManagerTrain, learningRate, l2RegStrength=1e-5, auto_stop=False):
    '''
      @param nEpochs: int. The number of epochs that will be used for training
      @param dataManagerTrain: DataManager. Object that will provide training batches (Xs and labels)
    '''

    print("Learning rate: %.1e"%(learningRate) )
    print("L2 regularization strength: %.1e"%(l2RegStrength) )
    print("auto_stop:", auto_stop)
    sys.stdout.flush()
    
    n_batches_per_epoch_train, n_batches_per_epoch_val= dataManagerTrain.getNBatchesPerEpoch()
    nEpochs__= nEpochs
    nEpochs= max(1, nEpochs*float(n_batches_per_epoch_train)/CHECK_POINT_AT)
    for modelNum in range(self.numberOfModels):
      self.startSessionAndInitialize()
      print("Training model %d/%d"%((modelNum+1), self.numberOfModels))  
      currentCheckPointName= self.checkPointsNameTemplate%modelNum
      print("current checkpoint name %s"%(currentCheckPointName))
      if os.path.isfile( currentCheckPointName ):
        print("loading previosly saved model %s"%(currentCheckPointName))
        self.loadNNet( currentCheckPointName, keepTraining=True, learningRate= learningRate, l2RegStrength=1e-5)
      else:
        effective_data_size= self.effective_data_size if self.effective_data_size>0 else dataManagerTrain.nTrue
        self.createNet(dataManagerTrain.shape[0], dataManagerTrain.shape[1], dataManagerTrain.shape[2], effective_data_size,
                       learningRate, l2RegStrength)
#      print(self.nNetModel.summary())
      print("nEpochs : %.1f --> Epochs: %d.\nTraining begins: Epoch 0/%d"%(nEpochs__, nEpochs, nEpochs))
      sys.stdout.flush()
      cBacks= [ keras.callbacks.ModelCheckpoint((currentCheckPointName) , monitor='val_acc', verbose=1,
                save_best_only=True, save_weights_only=False, period=1) ]
      if auto_stop:
        cBacks+= [ keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1) ]

      cBacks+= [ keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3, cooldown=1,
                 min_lr= learningRate*1e-3, verbose=1) ]

      history = self.nNetModel.fit_generator(dataManagerTrain.getTrainIterator(),steps_per_epoch= CHECK_POINT_AT,
                                 validation_data=dataManagerTrain.getValidationIterator( batchesPerEpoch= n_batches_per_epoch_val), 
                                 validation_steps=n_batches_per_epoch_val, callbacks=cBacks, epochs=nEpochs,
                                 use_multiprocessing=True, verbose=2)

      last_val_acc = history.history['val_acc'][-1]
      writeNetAccuracy(currentCheckPointName, last_val_acc)
      self.closeSession()

    loadANDwriteNetAccuracy(self.rootPath, self.numberOfModels)
      
      
  def predictNet(self, dataManger):
    n_images, n_batches= dataManger.getIteratorPredictBatchNSteps()
    y_pred_all= np.zeros(n_images)
    for modelNum in range(self.numberOfModels):
      self.startSessionAndInitialize()
      currentCheckPointName= self.checkPointsNameTemplate%modelNum
      if os.path.isfile( currentCheckPointName ):
        print("loading model %s"%(currentCheckPointName)); sys.stdout.flush()
        self.loadNNet( currentCheckPointName, keepTraining=False)
      else:
        raise ValueError("Neural net must be trained before prediction")

      sys.stdout.flush()
      print("predicting with model %d/%d"%((modelNum+1), self.numberOfModels)); sys.stdout.flush()
      y_pred_all+= self.nNetModel.predict_generator( (data for data,label in dataManger.getIteratorPredictBatch() ),
                                                steps= n_batches, use_multiprocessing=True, verbose=0)[:,1]
      print("prediction done"); sys.stdout.flush()
      self.closeSession()
    y_pred_all= y_pred_all/ self.numberOfModels
    return y_pred_all, dataManger.getPredictDataLabel_Id_dataSetNum()

  def getMccPrecRecal(self, labels, scores):

    thr=0.
    bestThr= thr
    bestMcc=-1.0
    for i in range(1000):
      x_bin= [1 if x_i>=thr else 0 for x_i in scores ]
      mcc= matthews_corrcoef(labels, x_bin)
      if mcc> bestMcc:
        bestThr= thr
        bestMcc= mcc
      thr+= 1/1000.
    print("bestThr",bestThr)
    x_bin= [1 if x_i>=bestThr else 0 for x_i in scores ]
    acc= accuracy_score(labels, x_bin)
    precision= precision_score(labels, x_bin)
    recall= recall_score(labels, x_bin)
    return bestMcc, precision, recall, acc

  def evaluateNet(self, dataManger):

    n_images, n_batches= dataManger.getIteratorPredictBatchNSteps()
    y_pred_all= np.zeros( (self.numberOfModels, n_images) )
    y_labels= np.concatenate( [label[:,1] for data,label in dataManger.getIteratorPredictBatch()] )
    stats=[]
    for modelNum in range(self.numberOfModels):
      self.startSessionAndInitialize()
      print("evaluating model %d/%d"%((modelNum+1), self.numberOfModels))
      currentCheckPointName= self.checkPointsNameTemplate%modelNum
      if os.path.isfile( currentCheckPointName ):
        print("loading model %s"%(currentCheckPointName))
        self.nNetModel= keras.models.load_model( currentCheckPointName )
      else:
        raise ValueError("Neural net must be trained before prediction")
      sys.stdout.flush()

      y_pred_all[modelNum,:]= self.nNetModel.predict_generator( ( (data, label) for data,label in 
                                                      dataManger.getIteratorPredictBatch() ),steps= n_batches, 
                                                      use_multiprocessing=True, verbose=0)[:,1]

      curr_auc= roc_auc_score(y_labels, y_pred_all[modelNum,:] )
      curr_acc= accuracy_score(y_labels, [1 if y>=0.5 else 0 for y in  y_pred_all[modelNum,:]])
      print("Model %d test accuracy (thr=0.5): %f  auc: %f"%(modelNum, curr_acc, curr_auc))
      bestMcc, precision, recall, acc= self.getMccPrecRecal(y_labels, y_pred_all[modelNum,:])
      stats.append( (bestMcc, precision, recall, acc, curr_auc) )
      print("Model %d test (thr=bestMcc) mcc: %f  pre: %f  rec: %f  acc: %f "%(modelNum, bestMcc, precision, recall, acc))
      self.closeSession()

    bestMcc, precision, recall, acc, curr_auc= zip(* stats)
    bestMcc, precision, recall, acc, curr_auc= map(np.mean, [bestMcc, precision, recall, acc, curr_auc])
    print(">>>>>>>>>>\nall models mean stats: mcc : %f prec: %f rec: %f  acc: %f roc_auc: %f"%( bestMcc, precision, recall, acc, curr_auc))

    y_pred_all= np.mean(y_pred_all, axis=0)
    global_auc= roc_auc_score(y_labels, y_pred_all )
    global_acc= accuracy_score(y_labels, [1 if y>=0.5 else 0 for y in  y_pred_all])
    print(">>>>>>>>>>>>\nEnsemble test accuracy (thr=0.5)     : %f  auc: %f"%(global_acc , global_auc))
    return global_auc, global_acc, y_labels, y_pred_all

class DataManager(object):

  def __init__(self, posSetDict, negSetDict=None, validationFraction=0.1):
    '''
        posSetDict, negSetDict: { fnameToMetadata:  weight:int ]
    '''
    assert validationFraction <= 0.4, "Error, validationFraction must  <= 0.4"
    if negSetDict is None: validationFraction= -1
    self.mdListFalse=None
    self.nFalse=0 #Number of negative particles in dataManager

    self.mdListTrue, self.fnMergedListTrue, self.weightListTrue, self.nTrue, self.shape= self.colectMetadata(posSetDict)
    self.batchSize= BATCH_SIZE
    self.splitPoint= self.batchSize//2
    self.validationFraction= validationFraction
    
    if validationFraction!=0:
        assert 0 not in self.getNBatchesPerEpoch(), "Error, the number of positive particles for training is to small (%d). Must be >> %d"%(self.nTrue, BATCH_SIZE)
    else:
        assert self.getNBatchesPerEpoch()[0] != 0, "Error, the number of particles for testing is to small (%d). Must be >> %d"%(self.nTrue, BATCH_SIZE)
 
    if validationFraction>0:
      self.trainingIdsPos= np.random.choice( self.nTrue,  int((1-validationFraction)*self.nTrue), False)
      self.validationIdsPos= np.array(list(set(range(self.nTrue)).difference(self.trainingIdsPos)))
    else:
      self.trainingIdsPos= range( self.nTrue )
      self.validationIdsPos= None
        
    if not negSetDict is None:
      self.mdListFalse, self.fnMergedListFalse, self.weightListFalse, self.nFalse, shapeFalse=  self.colectMetadata(negSetDict)
      assert shapeFalse== self.shape, "Negative images and positive images have different shape"
      self.trainingIdsNeg= np.random.choice( self.nFalse,  int((1-validationFraction)*self.nFalse), False)
      self.validationIdsNeg= np.array(list(set(range(self.nFalse)).difference(self.trainingIdsNeg)))
#      if validationFraction>0 and not self.trainingIdsNeg is None:
#        assert len(self.trainingIdsPos)<=  len(self.trainingIdsNeg), "Error, the number of positive particles "+\
#        "must be <= negative particles ( %d / %d)"%(len(self.trainingIdsPos), len(self.trainingIdsNeg))
        
  def colectMetadata(self, dictData):

    mdList=[]
    fnamesList_merged=[]
    weightsList_merged= []
    nParticles=0
    shapeParticles=(None, None, 1)
    for fnameXMDF in sorted(dictData):
      weight= float(dictData[fnameXMDF] )
      mdObject  = xmippLib.MetaData(fnameXMDF)
      I= xmippLib.Image()
      I.read(mdObject.getValue(xmippLib.MDL_IMAGE, mdObject.firstObject()))
      xdim, ydim= I.getData().shape
      imgFnames = mdObject.getColumnValues(xmippLib.MDL_IMAGE)
      mdList+= [mdObject]
      fnamesList_merged+= imgFnames
      tmpShape= (xdim,ydim,1)
      tmpNumParticles= mdObject.size()
      if shapeParticles!= (None, None, 1):
        assert tmpShape== shapeParticles, "Error, particles of different shapes mixed"
      else:
        shapeParticles= tmpShape
      if weight<=0:
          otherParticlesNum=0
          for fnameXMDF_2 in sorted(dictData):
              weight_2= float(dictData[fnameXMDF_2])
              if weight_2>0:
                  otherParticlesNum+= xmippLib.MetaData(fnameXMDF_2).size()
          weight= max(1, otherParticlesNum // tmpNumParticles)
      weightsList_merged+= [ weight  for elem in imgFnames]
      nParticles+= tmpNumParticles
    print(sorted(dictData))
    weightsList_merged= np.array(weightsList_merged, dtype= np.float64)
    weightsList_merged= weightsList_merged/ weightsList_merged.sum()
    return mdList, fnamesList_merged, weightsList_merged, nParticles, shapeParticles

  def getMetadata(self, dataSetNumber=None) :

    if dataSetNumber is None:
      return [mdTrue for mdTrue in self.mdListTrue], [mdFalse for mdFalse in self.mdListFalse] if self.mdListFalse else None
    else:
      mdTrue= self.mdListTrue[dataSetNumber]
      mdFalse= self.mdListFalse[dataSetNumber]
      return  mdTrue, mdFalse

  def getBatchSize(self):
    return self.batchSize

  def _random_flip_leftright(self, batch):
    for i in range(len(batch)):
      if bool(random.getrandbits(1)):
        batch[i] = np.fliplr(batch[i])
    return batch

  def _random_flip_updown(self, batch):
    for i in range(len(batch)):
      if bool(random.getrandbits(1)):
        batch[i] = np.flipud(batch[i])
    return batch

  def _random_90degrees_rotation(self, batch, rotations=[0, 1, 2, 3]):
    for i in range(len(batch)):
      num_rotations = random.choice(rotations)
      batch[i] = np.rot90(batch[i], num_rotations)
    return batch

  def _random_rotation(self, batch, max_angle):
    for i in range(len(batch)):
      if bool(random.getrandbits(1)):
        # Random angle
        angle = random.uniform(-max_angle, max_angle)
        batch[i] = scipy.ndimage.interpolation.rotate(batch[i], angle,reshape=False, mode="reflect")
    return batch

  def _random_blur(self, batch, sigma_max):
    for i in range(len(batch)):
      if bool(random.getrandbits(1)):
        # Random sigma
        sigma = random.uniform(0., sigma_max)
        batch[i] =scipy.ndimage.filters.gaussian_filter(batch[i], sigma)
    return batch

  def augmentBatch(self, batch):
    if bool(random.getrandbits(1)):
      batch= self._random_flip_leftright(batch)
      batch= self._random_flip_updown(batch)
    if bool(random.getrandbits(1)):
      batch= self._random_90degrees_rotation(batch)
    if bool(random.getrandbits(1)):
      batch= self._random_rotation(batch, 10.0)
    return batch

  def getDataAsNp(self):
    allData= self.getIteratorPredictBatch()
    x, labels, __ = zip(* allData)
    x= np.concatenate(x)
    y= np.concatenate(labels)
    return x,y

  def getPredictDataLabel_Id_dataSetNum(self):
    label_Id_dataSetNum=[]
    for dataSetNum in range(len(self.mdListTrue)):
      mdTrue= self.mdListTrue[dataSetNum]
      for objId in mdTrue:
        label_Id_dataSetNum.append((True,objId, dataSetNum))
    if not self.mdListFalse is None:
      for dataSetNum in range(len(self.mdListFalse)):
        mdFalse= self.mdListFalse[dataSetNum]
        for objId in mdFalse:
          label_Id_dataSetNum.append((False,objId, dataSetNum))
    return label_Id_dataSetNum

  def getIteratorPredictBatchNSteps(self):
    '''
    return numberOfItems, numberOfBatches
    '''
    nItems= 0
    for dataSetNum in range(len(self.mdListTrue)):
      nItems+= sum( (1 for elem in self.mdListTrue[dataSetNum]) )
    if not self.mdListFalse is None:
      for dataSetNum in range(len(self.mdListFalse)):
        nItems+= sum( (1 for elem in self.mdListFalse[dataSetNum]) )
    return nItems, int( np.ceil(nItems/float(self.batchSize) ))

  def getIteratorPredictBatch(self):
    batchSize = self.batchSize
    xdim,ydim,nChann= self.shape
    batchStack = np.zeros((self.batchSize, xdim,ydim,nChann))
    batchLabels  = np.zeros((batchSize, 2))
    I = xmippLib.Image()
    n = 0
    for dataSetNum in range(len(self.mdListTrue)):
      mdTrue= self.mdListTrue[dataSetNum]
      for objId in mdTrue:
        fnImage = mdTrue.getValue(xmippLib.MDL_IMAGE, objId)
        I.read(fnImage)
        batchStack[n,...]= np.expand_dims(I.getData(),-1)
        batchLabels[n, 1]= 1
        n+=1
        if n>=batchSize:
#          fig=plt.figure()
#          ax=fig.add_subplot(1,1,1)
#          ax.imshow(np.squeeze(batchStack[np.random.randint(0,n)]), cmap="Greys")
#          fig.suptitle('label==1')
#          plt.show()
          yield batchStack, batchLabels
          n=0
          batchLabels  = np.zeros((batchSize, 2))
    if not self.mdListFalse is None:
      for dataSetNum in range(len(self.mdListFalse)):
        mdFalse= self.mdListFalse[dataSetNum]
        for objId in mdFalse:
          fnImage = mdFalse.getValue(xmippLib.MDL_IMAGE, objId)
          I.read(fnImage)
          batchStack[n,...]= np.expand_dims(I.getData(),-1)
          batchLabels[n, 0]= 1
          n+=1
          if n>=batchSize:
#            fig=plt.figure()
#            ax=fig.add_subplot(1,1,1)
#            ax.imshow(np.squeeze(batchStack[np.random.randint(0,n)]), cmap="Greys")
#            fig.suptitle('label==0')
#            plt.show()
            yield batchStack, batchLabels
            n=0
            batchLabels  = np.zeros((batchSize, 2))
    if n>0:
      yield batchStack[:n,...], batchLabels[:n,...]

  def getNBatchesPerEpoch(self):
    return ( int((1-self.validationFraction)*self.nTrue*2./self.batchSize),
             int(self.validationFraction*self.nTrue*2./self.batchSize ) )

  def getTrainIterator(self, nEpochs=-1):
    if nEpochs<0:
      nEpochs= sys.maxsize
    for i in range(nEpochs):
      for batch in self._getOneEpochTrainOrValidation(isTrain_or_validation="train"):
        yield batch

  def getValidationIterator(self, nEpochs=-1, batchesPerEpoch= None):
    if nEpochs<0:
      nEpochs= sys.maxsize
    for i in range(nEpochs):
      for batch in self._getOneEpochTrainOrValidation(isTrain_or_validation="validation", nBatches= batchesPerEpoch):
        yield batch

  def _getOneEpochTrainOrValidation(self, isTrain_or_validation, nBatches= None):

    batchSize = self.batchSize
    xdim,ydim,nChann= self.shape
    batchStack = np.zeros((self.batchSize, xdim,ydim,nChann))
    batchLabels  = np.zeros((batchSize, 2))
    I = xmippLib.Image()
    n = 0
    currNBatches=0

    if isTrain_or_validation=="train":
      idxListTrue =  np.random.choice(self.trainingIdsPos, len(self.trainingIdsPos), True, 
                                      p= self.weightListTrue[self.trainingIdsPos]/ np.sum(
                                                                                  self.weightListTrue[self.trainingIdsPos]))
      idxListFalse = np.random.choice(self.trainingIdsNeg, len(self.trainingIdsNeg), True,
                                      p= self.weightListFalse[self.trainingIdsNeg]/ np.sum(
                                                                                  self.weightListFalse[self.trainingIdsNeg]))
      augmentBatch= self.augmentBatch
    elif isTrain_or_validation=="validation":
      idxListTrue =  self.validationIdsPos
      idxListFalse = self.validationIdsNeg
      augmentBatch= lambda x: x
    else:
      raise ValueError("isTrain_or_validation must be either train or validation")

    fnMergedListTrue=   ( self.fnMergedListTrue[i] for i in idxListTrue )
    fnMergedListFalse=  ( self.fnMergedListFalse[i] for i in idxListFalse )


    for fnImageTrue, fnImageFalse in zip(fnMergedListTrue, fnMergedListFalse):
      I.read(fnImageTrue)
      batchStack[n,...]= np.expand_dims(I.getData(),-1)
      batchLabels[n, 1]= 1
      n+=1
      if n>=batchSize:
        yield augmentBatch(batchStack), batchLabels
        n=0
        batchLabels  = np.zeros((batchSize, 2))
        currNBatches+=1
        if nBatches and currNBatches>=nBatches:
          break
      I.read(fnImageFalse)
      batchStack[n,...]= np.expand_dims(I.getData(),-1)
      batchLabels[n, 0]= 1
      n+=1
      if n>=batchSize:
        yield augmentBatch(batchStack), batchLabels
        n=0
        batchLabels  = np.zeros((batchSize, 2))
        currNBatches+=1
        if nBatches and currNBatches>=nBatches:
          break
    if n>0:
      yield augmentBatch(batchStack[:n,...]), batchLabels[:n,...]


