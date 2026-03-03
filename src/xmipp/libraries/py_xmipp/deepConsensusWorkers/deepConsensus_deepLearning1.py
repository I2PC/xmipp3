# **************************************************************************
# *
# * Authors:  Ruben Sanchez (rsanchez@cnb.csic.es), April 2017
# * Authors:  Mikel Iceta (miceta@cnb.csic.es), March 2026
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
import subprocess

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
try:
  import tensorflow_addons as tfa
except Exception:
  tfa = None
from .deepConsensus_networkDef import main_network, DESIRED_INPUT_SIZE
tf_intarnalError= tf.errors.InternalError

BATCH_SIZE= 32
CHECK_POINT_AT= 50 #In batches

WRITE_TEST_SCORES= True

def _query_gpu_memory_nvidia_smi(gpu_index=0):
  """Return (total_mb, free_mb) for the selected GPU using nvidia-smi, or None on failure."""
  try:
    out = subprocess.check_output([
      'nvidia-smi',
      '--query-gpu=memory.total,memory.free',
      '--format=csv,nounits,noheader'
    ], encoding='utf-8')
    lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
    if len(lines) == 0:
      return None
    if gpu_index < 0 or gpu_index >= len(lines):
      return None
    parts = [p.strip() for p in lines[gpu_index].split(',')]
    total_mb = int(parts[0])
    free_mb = int(parts[1])
    return total_mb, free_mb
  except Exception:
    return None

def estimate_batch_size_for_image(image_shape, gpu_index=0, bytes_per_element=4,
                                  reserved_memory_mb=512, overhead_ratio=0.9, max_batch=None,
                                  use_fp16=None):
  """Estimate a reasonable batch size for GPU training given an image shape.

  Parameters:
    image_shape: tuple (H, W, C)
    gpu_index: GPU index to query
    bytes_per_element: bytes per image element (float32=4)
    reserved_memory_mb: memory to reserve for model weights/other usage
    overhead_ratio: fraction of available memory to use for activations
    max_batch: optional upper cap for batch size

  Returns: dict with keys: estimated_batch, free_mb, total_mb, image_bytes
  Returns None if GPU memory cannot be queried.
  """
  try:
    H, W, C = map(int, image_shape)
  except Exception:
    raise ValueError('image_shape must be (H,W,C) with integer values')

  # If use_fp16 is not supplied, try to detect current global policy
  if use_fp16 is None:
    try:
      policy = tf.keras.mixed_precision.global_policy()
      use_fp16 = (getattr(policy, 'compute_dtype', '') == 'float16' or getattr(policy, 'name', '').startswith('mixed_float16'))
    except Exception:
      use_fp16 = False

  # adjust bytes_per_element when using FP16 / mixed precision
  bpe = int(bytes_per_element)
  if use_fp16 and bpe == 4:
    bpe = 2

  image_bytes = H * W * C * bpe

  info = _query_gpu_memory_nvidia_smi(gpu_index)
  total_mb = free_mb = None
  if info is None:
    # Try TF query as a fallback
    try:
      mem_info = tf.config.experimental.get_memory_info(f'GPU:{gpu_index}')
      limit = mem_info.get('limit', None)
      current = mem_info.get('current', 0)
      if limit is None:
        return None
      total_mb = int(limit // (1024 * 1024))
      free_mb = int((limit - current) // (1024 * 1024))
    except Exception:
      return None
  else:
    total_mb, free_mb = info

  # If using mixed precision, reserve some extra memory for master weights in fp32 and loss-scaling overhead
  if use_fp16:
    reserved_memory_mb = int(reserved_memory_mb + 256)

  available_mb = max(0, int(free_mb) - int(reserved_memory_mb))
  available_bytes = available_mb * 1024 * 1024
  use_bytes = int(available_bytes * float(overhead_ratio))
  estimated = max(1, int(use_bytes // image_bytes))
  if max_batch is not None:
    estimated = min(estimated, int(max_batch))

  return {
    'estimated_batch': estimated,
    'free_mb': int(free_mb),
    'total_mb': int(total_mb),
    'image_bytes': int(image_bytes)
  }


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
  def __init__(self, numberOfThreads, rootPath, numberOfModels=1, effective_data_size=-1, use_mixed_precision=False, resizeSize=None):
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
    self.use_mixed_precision = use_mixed_precision
    self.resizeSize = resizeSize if resizeSize is not None else DESIRED_INPUT_SIZE
    if self.use_mixed_precision:
      try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
      except Exception:
        pass
    
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
    self.nNetModel, self.optimizerFunLambda = main_network( (xdim, ydim, num_chan),  nData= nData, l2RegStrength= l2RegStrength, resizeSize=self.resizeSize)
    self.optimizer= self.optimizerFunLambda(learningRate)
    # If mixed precision is enabled, wrap optimizer with LossScaleOptimizer
    if self.use_mixed_precision:
      try:
        self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)
      except Exception:
        pass
    self.nNetModel.compile( self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

  def loadNNet(self, kerasModelFname, keepTraining=True, learningRate=1e-4, l2RegStrength=1e-5):
    self.nNetModel= keras.models.load_model( kerasModelFname , custom_objects={"DESIRED_INPUT_SIZE":DESIRED_INPUT_SIZE})
    self.optimizer= self.nNetModel.optimizer
    if keepTraining:
      # TF2: optimizer uses `learning_rate` attribute. If optimizer is a LossScaleOptimizer,
      # access the underlying optimizer via `.optimizer`.
      try:
        opt = getattr(self.nNetModel.optimizer, 'optimizer', self.nNetModel.optimizer)
        try:
          K.set_value(opt.learning_rate, learningRate)
        except Exception:
          opt.learning_rate = learningRate
      except Exception:
        pass
      for layer in self.nNetModel.layers:
        if hasattr(layer, "kernel_regularizer"):
          if hasattr(layer.kernel_regularizer, "l2"):
            layer.kernel_regularizer.l2= l2RegStrength
      self.nNetModel.compile( self.nNetModel.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
  def startSessionAndInitialize(self):
    '''
    '''
    # TF2 runs in eager mode; configure threading and GPU memory growth instead of sessions
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
      except Exception:
        pass
    # Configure threading when numberOfThreads specified
    if self.numberOfThreads is not None:
      try:
        tf.config.threading.set_intra_op_parallelism_threads(self.numberOfThreads)
        tf.config.threading.set_inter_op_parallelism_threads(self.numberOfThreads)
      except Exception:
        pass
    return None

  def closeSession(self):
    '''
      Closes a tensorflow connection and related objects.

    '''
    K.clear_session()
    
  def trainNet(self, nEpochs, dataManagerTrain, learningRate, l2RegStrength=1e-5, auto_stop=False,
               lr_auto_scale=False, lr_base_batch_size=None, batch_log_every_n=10):
    '''
      @param nEpochs: int. The number of epochs that will be used for training
      @param dataManagerTrain: DataManager. Object that will provide training batches (Xs and labels)
    '''

    print("Learning rate: %.1e"%(learningRate) )
    print("L2 regularization strength: %.1e"%(l2RegStrength) )
    print("auto_stop:", auto_stop)
    sys.stdout.flush()
    
    n_batches_per_epoch_train, n_batches_per_epoch_val = dataManagerTrain.getNBatchesPerEpoch()
    # Keep a copy of the user-provided epoch count
    nEpochs__ = nEpochs

    # Determine steps_per_epoch autoscaled inversely with batch size: don't exceed available
    # batches and prefer CHECK_POINT_AT as an upper bound for reporting cadence.
    steps_per_epoch = int(min(CHECK_POINT_AT, max(1, n_batches_per_epoch_train)))

    # Adjust number of epochs to keep the total number of optimizer steps roughly
    # constant across changes in steps_per_epoch: total_steps = nEpochs__ * n_batches_per_epoch_train
    # so new_nEpochs = total_steps / steps_per_epoch
    nEpochs = int(max(1, int(round((nEpochs__ * float(n_batches_per_epoch_train)) / float(steps_per_epoch)))))
    for modelNum in range(self.numberOfModels):
      self.startSessionAndInitialize()
      print("Training model %d/%d"%((modelNum+1), self.numberOfModels))  
      currentCheckPointName= self.checkPointsNameTemplate%modelNum
      print("current checkpoint name %s"%(currentCheckPointName))
      if os.path.isfile( currentCheckPointName ):
        print("loading previosly saved model %s"%(currentCheckPointName))
        # Optionally autoscale learning rate with batch size (linear scaling rule)
        batch_size = dataManagerTrain.getBatchSize()
        if lr_auto_scale:
          base_bs = lr_base_batch_size if lr_base_batch_size is not None else BATCH_SIZE
          scaled_lr = float(learningRate) * (float(batch_size) / float(base_bs))
          print('Auto-scaled learning rate %.3e -> %.3e (batch %d -> base %d)' % (learningRate, scaled_lr, batch_size, base_bs))
        else:
          scaled_lr = learningRate
        self.loadNNet( currentCheckPointName, keepTraining=True, learningRate= scaled_lr, l2RegStrength=1e-5)
      else:
        effective_data_size= self.effective_data_size if self.effective_data_size>0 else dataManagerTrain.nTrue
        batch_size = dataManagerTrain.getBatchSize()
        if lr_auto_scale:
          base_bs = lr_base_batch_size if lr_base_batch_size is not None else BATCH_SIZE
          scaled_lr = float(learningRate) * (float(batch_size) / float(base_bs))
          print('Auto-scaled learning rate %.3e -> %.3e (batch %d -> base %d)' % (learningRate, scaled_lr, batch_size, base_bs))
        else:
          scaled_lr = learningRate
        self.createNet(dataManagerTrain.shape[0], dataManagerTrain.shape[1], dataManagerTrain.shape[2], effective_data_size,
                       scaled_lr, l2RegStrength)
#      print(self.nNetModel.summary())
      print("nEpochs : %.1f --> Epochs: %d.\nTraining begins: Epoch 0/%d"%(nEpochs__, nEpochs, nEpochs))
      sys.stdout.flush()
      cBacks= [ keras.callbacks.ModelCheckpoint((currentCheckPointName) , monitor='val_accuracy', verbose=1,
            save_best_only=True, save_weights_only=False, save_freq='epoch') ]
      if auto_stop:
        cBacks+= [ keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10, verbose=1) ]

      cBacks+= [ keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, cooldown=1,
                 min_lr= learningRate*1e-3, verbose=1) ]

      # Batch-level logging callback to provide more frequent progress when steps_per_epoch is small
      class BatchLoggingCallback(keras.callbacks.Callback):
        def __init__(self, every_n=10):
          super(BatchLoggingCallback, self).__init__()
          self.every_n = max(1, int(every_n))
          self.batch_count = 0

        def on_train_batch_end(self, batch, logs=None):
          self.batch_count += 1
          if (self.batch_count % self.every_n) == 0:
            logs = logs or {}
            loss = logs.get('loss')
            acc = logs.get('accuracy') or logs.get('acc')
            msg = 'batch %d: loss=%.4g' % (self.batch_count, loss if loss is not None else float('nan'))
            if acc is not None:
              msg += ', acc=%.4g' % acc
            print(msg)
            try:
              sys.stdout.flush(); sys.stderr.flush()
              for h in list(logging.root.handlers):
                try:
                  h.flush()
                except Exception:
                  pass
            except Exception:
              pass

      import logging
      cBacks += [ BatchLoggingCallback(every_n=batch_log_every_n) ]

      # Use tf.data.Dataset for training to leverage TF2 performance
      train_ds = dataManagerTrain.getTrainDataset()
      val_ds = dataManagerTrain.getValidationDataset(batchesPerEpoch=n_batches_per_epoch_val)
      history = self.nNetModel.fit(x=train_ds, steps_per_epoch= steps_per_epoch,
             validation_data=val_ds, validation_steps=n_batches_per_epoch_val,
             callbacks=cBacks, epochs=nEpochs, verbose=2)

      # history key in TF2 is 'val_accuracy'
      last_val_acc = history.history.get('val_accuracy', history.history.get('val_acc'))[-1]
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
      preds = self.nNetModel.predict(dataManger.getPredictDataset(), steps=n_batches, verbose=0)
      y_pred_all+= preds[:,1]
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
        # Use the same loader as elsewhere to preserve custom_objects and optimizer wrappers
        self.loadNNet(currentCheckPointName, keepTraining=False)
      else:
        raise ValueError("Neural net must be trained before prediction")
      sys.stdout.flush()

      preds = self.nNetModel.predict(dataManger.getPredictDataset(), steps=n_batches, verbose=0)
      y_pred_all[modelNum,:]= preds[:,1]

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

  def __init__(self, posSetDict, negSetDict=None, validationFraction=0.1, batch_size=None, prefetch_to_device=True, resizeSize=None):
    '''
        posSetDict, negSetDict: { fnameToMetadata:  weight:int ]
    '''
    assert validationFraction <= 0.4, "Error, validationFraction must  <= 0.4"
    if negSetDict is None: validationFraction= -1
    self.mdListFalse = None
    self.nFalse = 0 #Number of negative particles in dataManager
    # defaults for negative dataset attributes so callers don't hit AttributeError
    self.fnMergedListFalse = None
    self.weightListFalse = None
    self.trainingIdsNeg = []
    self.validationIdsNeg = None
    # store user-provided resize size (falls back to DESIRED_INPUT_SIZE)
    self.resizeSize = (resizeSize, resizeSize, 1) if resizeSize is not None else (DESIRED_INPUT_SIZE, DESIRED_INPUT_SIZE, 1)

    self.mdListTrue, self.fnMergedListTrue, self.weightListTrue, self.nTrue, self.shape= self.colectMetadata(posSetDict)

    info = estimate_batch_size_for_image(self.resizeSize, gpu_index=0, reserved_memory_mb=768, overhead_ratio=0.92, max_batch=1024)

    if info and info.get('estimated_batch'):
      est = int(info['estimated_batch'])
      # ensure batch is at least 1 and not absurdly small
      self.batchSize = max(1, est)
      print('Auto-selected batch size %d from GPU free %dMB (image bytes %d) using shape %s'
            % (self.batchSize, info.get('free_mb', 0), info.get('image_bytes', 0), str(self.resizeSize)))
    else:
      self.batchSize = BATCH_SIZE
      
    self.prefetch_to_device = prefetch_to_device
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

    fnMergedListTrue = (self.fnMergedListTrue[i] for i in idxListTrue)
    # fnMergedListFalse may be None or empty when no negative dataset was provided
    if (self.fnMergedListFalse is not None) and (len(ids_false if 'ids_false' in locals() else idxListFalse) > 0):
      fnMergedListFalse = (self.fnMergedListFalse[i] for i in idxListFalse)
    else:
      fnMergedListFalse = None

    if fnMergedListFalse is not None:
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
    else:
      # No negative examples: yield batches composed only of positive examples
      for fnImageTrue in fnMergedListTrue:
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
    if n>0:
      yield augmentBatch(batchStack[:n,...]), batchLabels[:n,...]

  def getTrainDataset(self):
    """Return a tf.data.Dataset that yields (batch_images, batch_labels) for training.

    This dataset yields per-example tensors and performs augmentation with TF ops
    so shuffling, batching and prefetching are effective on the pipeline.
    """
    return self._get_dataset(isTrain_or_validation="train")

  def getValidationDataset(self, batchesPerEpoch=None):
    """Return a tf.data.Dataset that yields (batch_images, batch_labels) for validation."""
    return self._get_dataset(isTrain_or_validation="validation", nBatches=batchesPerEpoch)

  def getPredictDataset(self):
    """Return a tf.data.Dataset that yields (batch_images, batch_labels) for prediction/evaluation."""
    return self._get_dataset(isTrain_or_validation="predict")

  def _load_image_py(self, fname):
    """Load image from disk using xmippLib.Image and return numpy float32 array with channel dim."""
    I = xmippLib.Image()
    I.read(fname)
    arr = np.expand_dims(I.getData().astype(np.float32), -1)
    return arr

  def _py_load_image(self, fname):
    # wrapper for tf.numpy_function: receives a numpy bytes/string and returns a numpy float32 array
    # Handle different possible input types (bytes, numpy.bytes_, or 0-d array)
    if isinstance(fname, (bytes, bytearray)):
      fname_str = fname.decode('utf-8')
    elif hasattr(fname, 'item'):
      v = fname.item()
      if isinstance(v, (bytes, bytearray)):
        fname_str = v.decode('utf-8')
      else:
        fname_str = str(v)
    else:
      fname_str = str(fname)
    arr = self._load_image_py(fname_str)
    return arr

  def _tf_load_image(self, fname):
    img = tf.numpy_function(func=self._py_load_image, inp=[fname], Tout=tf.float32)
    # set static shape if available
    xdim, ydim, nChann = self.shape
    img.set_shape([xdim, ydim, nChann])
    return img

  def _augment_tf(self, image):
    """Apply augmentation with TF ops where possible (flips, 90deg rotations, small gaussian blur)."""
    # image: tf.Tensor [H,W,1], dtype float32
    # random flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # random 90-degree rotation
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    # small random rotation (angle in degrees) using tensorflow_addons when available
    def _apply_small_rotation(img):
      if tfa is not None:
        angle_deg = tf.random.uniform([], -10.0, 10.0)
        angle = angle_deg * (3.14159265 / 180.0)
        # tfa.image.rotate expects shape [H,W, C]
        return tfa.image.rotate(img, angles=angle, interpolation='bilinear', fill_mode='reflect')
      else:
        # fallback to py_function using scipy (CPU)
        def _py_rotate(img_np):
          # img_np is a numpy array when called via tf.numpy_function
          if bool(random.getrandbits(1)):
            angle = random.uniform(-10.0, 10.0)
            img_np = scipy.ndimage.interpolation.rotate(img_np.squeeze(), angle, reshape=False, mode='reflect')
            img_np = np.expand_dims(img_np, -1)
          return img_np.astype(np.float32)
        out = tf.numpy_function(func=_py_rotate, inp=[img], Tout=tf.float32)
        out.set_shape(img.shape)
        return out

    do_rotate = tf.less(tf.random.uniform([], 0.0, 1.0), 0.5)
    image = tf.cond(do_rotate, lambda: _apply_small_rotation(image), lambda: image)

    # Gaussian blur via depthwise conv2d (TF op) for GPU; fallback to py_function if kernel degenerate
    def _apply_gaussian_blur(img):
      # sigma random in [0, 1.0)
      sigma = tf.random.uniform([], 0.0, 1.0)
      # if sigma is very small, skip
      def _no_blur():
        return img

      def _blur():
        # kernel size: odd, proportional to sigma (3*sigma rule)
        radius = tf.cast(tf.math.ceil(3.0 * sigma), tf.int32)
        ksize = tf.maximum(1, radius * 2 + 1)
        # if ksize==1 no blur
        def _no_blur_inner():
          return img

        def _do_blur():
          # create 1D gaussian
          coords = tf.cast(tf.range(-radius, radius + 1), tf.float32)
          sigma_safe = tf.maximum(sigma, 1e-6)
          g = tf.exp(- (coords ** 2) / (2.0 * sigma_safe ** 2))
          g = g / tf.reduce_sum(g)
          kernel2d = tf.tensordot(g, g, axes=0)  # shape [ksize, ksize]
          kernel2d = kernel2d[:, :, tf.newaxis, tf.newaxis]
          # cast to float32
          kernel2d = tf.cast(kernel2d, tf.float32)
          # img shape [H,W,1], add batch dim
          img_batch = tf.expand_dims(img, 0)
          # pad to preserve size
          # perform depthwise conv
          blurred = tf.nn.depthwise_conv2d(img_batch, kernel2d, strides=[1, 1, 1, 1], padding='SAME')
          return tf.squeeze(blurred, 0)

        return tf.cond(tf.equal(ksize, 1), _no_blur_inner, _do_blur)

      return tf.cond(tf.less(sigma, 1e-6), _no_blur, _blur)

    do_blur = tf.less(tf.random.uniform([], 0.0, 1.0), 0.5)
    image = tf.cond(do_blur, lambda: _apply_gaussian_blur(image), lambda: image)
    return image

  def _augment_batch_tf(self, images):
    """Apply augmentation on a batch tensor `images` with shape [B,H,W,C].

    Vectorized where possible; for small-angle rotation and gaussian blur we use
    TF ops per-example via `tf.map_fn` when `tensorflow_addons` is available,
    otherwise fall back to a single `tf.numpy_function` operating on the whole batch.
    """
    # images: [B,H,W,C]
    B = tf.shape(images)[0]
    # Random per-example left-right flip
    mask_lr = tf.random.uniform([B], 0.0, 1.0) < 0.5
    flipped_lr = tf.image.flip_left_right(images)
    images = tf.where(tf.reshape(mask_lr, [B, 1, 1, 1]), flipped_lr, images)

    # Random per-example up-down flip
    mask_ud = tf.random.uniform([B], 0.0, 1.0) < 0.5
    flipped_ud = tf.image.flip_up_down(images)
    images = tf.where(tf.reshape(mask_ud, [B, 1, 1, 1]), flipped_ud, images)

    # Random 90-degree rotations per-example: use tfa if available for batch angles
    k = tf.random.uniform([B], 0, 4, dtype=tf.int32)
    if tfa is not None:
      angles = tf.cast(k, tf.float32) * (3.14159265 / 2.0)
      images = tfa.image.rotate(images, angles=angles, interpolation='nearest')
    else:
      # fallback to map_fn with tf.image.rot90 per example
      def _rot90_one(x_k):
        img, kk = x_k[0], tf.cast(x_k[1], tf.int32)
        return tf.image.rot90(img, kk)
      images = tf.map_fn(_rot90_one, (images, k), fn_output_signature=tf.float32)

    # Small-angle rotation and gaussian blur: prefer TF ops; use map_fn per-example
    def _augment_one(img):
      # small rotation
      if tfa is not None:
        angle_deg = tf.random.uniform([], -10.0, 10.0)
        angle = angle_deg * (3.14159265 / 180.0)
        img = tfa.image.rotate(img, angles=angle, interpolation='bilinear', fill_mode='reflect')
      else:
        # if no tfa, leave rotation to numpy fallback later
        pass

      # gaussian blur per-example using depthwise conv
      sigma = tf.random.uniform([], 0.0, 1.0)
      def _no_blur():
        return img
      def _do_blur():
        radius = tf.cast(tf.math.ceil(3.0 * sigma), tf.int32)
        ksize = tf.maximum(1, radius * 2 + 1)
        coords = tf.cast(tf.range(-radius, radius + 1), tf.float32)
        sigma_safe = tf.maximum(sigma, 1e-6)
        g = tf.exp(- (coords ** 2) / (2.0 * sigma_safe ** 2))
        g = g / tf.reduce_sum(g)
        kernel2d = tf.tensordot(g, g, axes=0)  # shape [ksize, ksize]
        kernel2d = kernel2d[:, :, tf.newaxis, tf.newaxis]
        kernel2d = tf.cast(kernel2d, tf.float32)
        img_batch = tf.expand_dims(img, 0)
        blurred = tf.nn.depthwise_conv2d(img_batch, kernel2d, strides=[1, 1, 1, 1], padding='SAME')
        return tf.squeeze(blurred, 0)
      img = tf.cond(tf.less(sigma, 1e-6), _no_blur, _do_blur)
      return img

    if tfa is not None:
      images = tf.map_fn(lambda x: _augment_one(x), images, fn_output_signature=tf.float32)
    else:
      # Fallback: call a numpy-based batch augmentor once per batch (cheaper than per-example pyfuncs)
      def _py_batch_augment(batch_np):
        out = []
        for i in range(batch_np.shape[0]):
          img = batch_np[i]
          if bool(random.getrandbits(1)):
            angle = random.uniform(-10.0, 10.0)
            img = scipy.ndimage.interpolation.rotate(img.squeeze(), angle, reshape=False, mode='reflect')
            img = np.expand_dims(img, -1)
          if bool(random.getrandbits(1)):
            sigma = random.uniform(0., 1.0)
            img = scipy.ndimage.filters.gaussian_filter(img, sigma)
          out.append(img.astype(np.float32))
        return np.stack(out, axis=0)
      images = tf.numpy_function(func=_py_batch_augment, inp=[images], Tout=tf.float32)
      # restore shape hints
      images.set_shape([None, self.shape[0], self.shape[1], self.shape[2]])

    return images

  def _get_dataset(self, isTrain_or_validation="train", nBatches=None):
    """Construct a per-example tf.data.Dataset for train/validation/predict."""
    # Build lists of filenames and labels depending on mode
    fn_list = []
    label_list = []
    if isTrain_or_validation == "train":
      ids_true = list(self.trainingIdsPos)
      ids_false = list(self.trainingIdsNeg) if self.mdListFalse is not None else []
    elif isTrain_or_validation == "validation":
      ids_true = list(self.validationIdsPos) if self.validationIdsPos is not None else []
      ids_false = list(self.validationIdsNeg) if self.validationIdsNeg is not None else []
    elif isTrain_or_validation == "predict":
      # include all datasets
      for fn in self.fnMergedListTrue:
        fn_list.append(fn)
        label_list.append([0.0, 1.0])
      if self.fnMergedListFalse is not None:
        for fn in self.fnMergedListFalse:
          fn_list.append(fn)
          label_list.append([1.0, 0.0])
      ds = tf.data.Dataset.from_tensor_slices((fn_list, label_list))
      ds = ds.map(lambda f, l: (self._tf_load_image(f), tf.cast(l, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
      ds = ds.batch(self.batchSize)
      ds = ds.prefetch(tf.data.AUTOTUNE)
      return ds
    else:
      raise ValueError("Unknown mode %s" % isTrain_or_validation)

    # build filename and label lists interleaving true/false if both present
    # Use provided id lists to select filenames
    for idx in ids_true:
      fn_list.append(self.fnMergedListTrue[idx])
      label_list.append([0.0, 1.0])
    for idx in ids_false:
      fn_list.append(self.fnMergedListFalse[idx])
      label_list.append([1.0, 0.0])

    ds = tf.data.Dataset.from_tensor_slices((fn_list, label_list))
    if isTrain_or_validation == "train":
      ds = ds.shuffle(buffer_size=len(fn_list))
    ds = ds.map(lambda f, l: (self._tf_load_image(f), tf.cast(l, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
    if isTrain_or_validation == "train":
      ds = ds.map(lambda x, y: (self._augment_tf(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(self.batchSize)
    # For training, repeat indefinitely so `fit(..., steps_per_epoch=...)` drives epochs
    if isTrain_or_validation == "train":
      ds = ds.repeat()

    # Optionally prefetch full batches to the GPU device to hide host->device copies
    if self.prefetch_to_device:
      gpus = tf.config.list_logical_devices('GPU')
      if gpus:
        try:
          device = gpus[0].name
          ds = ds.apply(tf.data.experimental.prefetch_to_device(device))
        except Exception:
          ds = ds.prefetch(tf.data.AUTOTUNE)
      else:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    else:
      ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
