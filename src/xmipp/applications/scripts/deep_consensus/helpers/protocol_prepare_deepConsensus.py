# **************************************************************************
# *
# * Authors:    Ruben Sanchez Garcia (rsanchez@cnb.csic.es)
# *             David Maluenda (dmaluenda@cnb.csic.es)
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
# *  e-mail address 'coss@cnb.csic.es'
# *
# **************************************************************************
"""
Deep Consensus picking protocol
"""
import os
from glob import glob

import json
from pyworkflow.utils.path import makePath, cleanPattern, cleanPath, copyTree, createLink
from pyworkflow.protocol.constants import *
from pwem.constants import RELATION_CTF, ALIGN_NONE
from pwem.convert import ImageHandler
from pwem.objects import SetOfCoordinates, Coordinate, SetOfParticles
from pwem.protocols import ProtParticlePicking

import pyworkflow.utils as pwutils
import pyworkflow.protocol.params as params
import pyworkflow.em.metadata as MD

from xmipp_base import XmippScript
import xmippLib as xmipp

import xmipp3
from xmipp3 import XmippProtocol
from xmipp3.protocols.protocol_pick_noise import pickNoise_prepareInput
from xmipp3.protocols.coordinates_tools.io_coordinates import readSetOfCoordsFromPosFnames
from xmipp3.convert import readSetOfParticles, setXmippAttributes, micrographToCTFParam, \
                           writeSetOfParticles, writeSetOfCoordinates

DEEP_PARTICLE_SIZE = 128
MIN_NUM_CONSENSUS_COORDS = 256

class XmippProtPrepareDeepConsensus(ProtParticlePicking, XmippProtocol):
    """ Protocol to pretrain 
    
    add
        from .protocol_prepare_deepConsensus import XmippProtPrepareDeepConsensus
    to
        scipion-em-xmipp/xmipp3/protocols/__init__.py
        
    so it can be detected
    """
    _label = 'prepare deep consensus picking'
    CONSENSUS_COOR_PATH_TEMPLATE="consensus_coords_%s_%d"
    CONSENSUS_PARTS_PATH_TEMPLATE="consensus_parts_%s_%d"
    PRE_PROC_MICs_PATH="preProcMics_%d"

    ADD_TRAIN_TYPES = ["None", "Precompiled", "Custom"]
    ADD_TRAIN_NONE = 0
    ADD_TRAIN_MODEL = 1
    ADD_TRAIN_CUST = 2

    def __init__(self, **args):
        ProtParticlePicking.__init__(self, **args)
        self.stepsExecutionMode = params.STEPS_SERIAL

    def _defineParams(self, form):
        # GPU settings
        form.addHidden(params.USE_GPU, params.BooleanParam,
                       default=True,
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Use GPU (vs CPU)",
                       help="Set to true if you want to use GPU implementation "
                            "of Optical Flow.")
        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=params.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on.")
                            
        form.addParallelSection(threads=2, mpi=0)
        
        form.addSection(label='Input')
        #CONTINUE FROM PREVIOUS TRAIN

        # CONTINUE PARAMETERS
        form.addParam('inputCoordinatesTrue', params.MultiPointerParam,
                      pointerClass='SetOfCoordinates',
                      label="Input coordinates true",
                      help='Select the set of true particles coordinates')

        form.addParam('inputCoordinatesFalse', params.MultiPointerParam,
                      pointerClass='SetOfCoordinates',
                      label="Input coordinates false",
                      help='Select the set of true particles coordinates')
        form.addSection(label='Preprocess')
        
        form.addParam('notePreprocess', params.LabelParam,
                      label='How to extract particles from micrograph',
                      help='Our method, internally, uses particles that are '
                           'extracted from preprocess micrographs. '
                           'Preprocess steps are:\n'
                           '1) mic donwsampling to the required size such that '
                           'the particle box size become 128 px. \n   E.g. xmipp_transform_downsample -i'
                           ' in/100_movie_aligned.mrc -o out1/100_movie_aligned.mrc --step newSamplingRate --method fourier\n'
                           '2) mic normalization to 0 mean and 1 std and [OPTIONALLY, mic contrast inversion].\n   E.g. '
                           ' xmipp_transform_normalize -i out1/101_movie_aligned.mrc -o out2/101_movie_aligned.mrc --method '
                           'OldXmipp [ --invert ]\n'
                           '3) particles extraction.\n   E.g. xmipp_micrograph_scissor  -i out2/101_movie_aligned.mrc '
                           '--pos particles@Runs/101_movie_aligned.pos -o out3/105_movie_aligned_particles '
                           ' --Xdim 128 --downsampling newSamplingRate --fillBorders  ( Correct your coordinates with '
                           'newSamplingRate if needed)\n'
                           '4) OPTIONAL: phase flipping using CTF.\n xmipp_ctf_phase_flip  -i '
                           'particles/105_movie_aligned_noDust.xmp -o particles/105_movie_aligned_flipped.xmp '
                           '--ctf ctfPath/105_movie_aligned.ctfParam --sampling newSamplingRate')

        form.addParam('doInvert', params.BooleanParam, default=False,
                      label='Invert contrast', 
                      help='If you invert the contrast, your particles will be white over a black background in the micrograph. '
                           'We prefer white particles. Select *Yes* if you have not inverted the constrast in the micrograph'
                           ' so that we can extract white particles')
        
        form.addParam('ignoreCTF', params.BooleanParam, default=True,
                      label='Ignore CTF',
                      help='Deep Consensus extracts particles. Do you want to ignore CTF for '
                            'particle extraction')
        
        form.addParam('ctfRelations', params.MultiPointerParam, allowsNull=True, pointerClass='SetOfCTF',
                      label="Input CTFs",  condition="not ignoreCTF",
                      help='Select the estimated CTF to evaluate')
                      
        form.addSection(label='Training')
        
        form.addParam('nEpochs', params.FloatParam,
                      label="Number of epochs", default=5.0,
                      help='Number of epochs for neural network training.')
        form.addParam('learningRate', params.FloatParam,
                      label="Learning rate", default=1e-4,
                      help='Learning rate for neural network training')
        form.addParam('auto_stopping',params.BooleanParam, default=True,
                      label='Auto stop training when convergency is detected?',
                      help='If you set to *Yes*, the program will automatically '
                           'stop training if there is no improvement for '
                           'consecutive 2 epochs, learning rate will be '
                           'decreased by a factor 10. '
                           'If learningRate_t < 0.01*learningrate_0 training will stop. '
                           'Warning: Sometimes convergency seems to be reached, '
                           'but after time, improvement can still happen. '
                           'Not recommended for very small data sets (<100 true particles)')
        form.addParam('l2RegStrength', params.FloatParam,
                      label="Regularization strength",
                      default=1e-5, expertLevel=params.LEVEL_ADVANCED,
                      help='L2 regularization for neural network weights.'
                           'Make it bigger if suffering overfitting (validation acc decreases but training acc increases)\n'
                           'Typical values range from 1e-1 to 1e-6')
                           
        form.addParam('doTesting', params.BooleanParam, default=False,
                      label='Perform testing after training?', expertLevel=params.LEVEL_ADVANCED,
                      help='If you set to *Yes*, you should select a testing '
                           'positive set and a testing negative set')
        form.addParam('testTrueSetOfParticles', params.PointerParam,
                      label="Set of positive test particles", expertLevel=params.LEVEL_ADVANCED,
                      pointerClass='SetOfParticles',condition='doTesting',
                      help='Select the set of ground true positive particles.')
        form.addParam('testFalseSetOfParticles', params.PointerParam,
                      label="Set of negative test particles", expertLevel=params.LEVEL_ADVANCED,
                      pointerClass='SetOfParticles', condition='doTesting',
                      help='Select the set of ground false positive particles.')

        form.addSection(label='Additional training data')
        form.addParam('addTrainingData', params.EnumParam,
                      choices=self.ADD_TRAIN_TYPES,
                      default=self.ADD_TRAIN_NONE,
                      label='Additional training data',
                      help='If you set to *%s*, you should select positive and/or '
                           'negative sets of particles. Regard that our method, '
                           'internally, uses particles that are extracted from '
                           'preprocess micrographs. Steps are:\n'
                           '1) mic donwsampling to the required size such that '
                           'the particle box size become 128 px. \n   E.g. xmipp_transform_downsample -i'
                           ' in/100_movie_aligned.mrc -o out1/100_movie_aligned.mrc --step newSamplingRate --method fourier\n'
                           '2) mic normalization to 0 mean and 1 std and [OPTIONALLY, mic contrast inversion].\n   E.g. '
                           ' xmipp_transform_normalize -i out1/101_movie_aligned.mrc -o out2/101_movie_aligned.mrc --method '
                           'OldXmipp [ --invert ]\n'
                           '3) particles extraction.\n   E.g. xmipp_micrograph_scissor  -i out2/101_movie_aligned.mrc '
                           '--pos particles@Runs/101_movie_aligned.pos -o out3/105_movie_aligned_particles '
                           ' --Xdim 128 --downsampling newSamplingRate --fillBorders  ( Correct your coordinates with '
                           'newSamplingRate if needed)\n'
                           '4) OPTIONAL: phase flipping using CTF.\n xmipp_ctf_phase_flip  -i '
                           'particles/105_movie_aligned_noDust.xmp -o particles/105_movie_aligned_flipped.xmp '
                           '--ctf ctfPath/105_movie_aligned.ctfParam --sampling newSamplingRate\n'
                           'Then, particles are extracted with no further alteration.\n'
                           'Please ensure that the additional particles have been '
                           'preprocessed as indicated before.\n\n'
                           'Alternatively, you can use a precompiled dataset '
                           'by selecting *%s* or simply, use only the '
                           'input coorditanes (*%s*).'
                           % (self.ADD_TRAIN_CUST, self.ADD_TRAIN_MODEL,
                              self.ADD_TRAIN_NONE))
        form.addParam('trainTrueSetOfParticles', params.PointerParam,
                      label="Positive train particles 128px (optional)",
                      pointerClass='SetOfParticles', allowsNull=True,
                      condition='addTrainingData==%s'%self.ADD_TRAIN_CUST,
                      help='Select a set of true positive particles. '
                           'Take care of the preprocessing')
        form.addParam('trainPosWeight', params.IntParam, default='1',
                      label="Weight of positive additional train particles",
                      condition='addTrainingData==%s' % self.ADD_TRAIN_CUST,
                      allowsNull=True,
                      help='Select the weigth for the additional train set of '
                           'positive particles.The weight value indicates '
                           'internal particles are weighted with 1. '
                           'If weight is -1, weight will be calculated such that '
                           'the contribution of additional data is equal to '
                           'the contribution of internal particles')
        form.addParam('trainFalseSetOfParticles', params.PointerParam,
                      label="Negative train particles 128px (optional)",
                      pointerClass='SetOfParticles',  allowsNull=True,
                      condition='addTrainingData==%s'%self.ADD_TRAIN_CUST,
                      help='Select a set of false positive particles. '
                           'Take care of the preprocessing')
        form.addParam('trainNegWeight', params.IntParam, default='1',
                      label="Weight of negative additional train particles",
                      condition='addTrainingData==%s' % self.ADD_TRAIN_CUST,
                      allowsNull=True,
                      help='Select the weigth for the additional train set of '
                           'negative particles. The weight value indicates '
                           'the number of times each image may be included at '
                           'most per epoch. Deep consensus internal particles '
                           'are weighted with 1. If weight is -1, weight '
                           'will be calculated such that the contribution of '
                           'additional data is equal to the contribution of '
                           'internal particles')

    def _validate(self):
        errorMsg = []
        if sum( [int(self._getBoxSize(i)< DEEP_PARTICLE_SIZE) for i in range(len(self.inputCoordinatesFalse))] ) >0:
          errorMsg.append("Error, too small particles (needed 128 px), "
                          "have you provided already downsampled micrographs? "
                          "If so, use original ones")
        if len(self.ctfRelations)<1:
          errorMsg.append("Error, CTFs must be provided to compute phase flip. "
                          "Please, provide a set of CTFs.")

        if  self.trainTrueSetOfParticles.get() and self.trainTrueSetOfParticles.get().getXDim()!=DEEP_PARTICLE_SIZE:
          errorMsg.append("Error, trainTrueSetOfParticles needed to be 128 px")

        if  self.trainFalseSetOfParticles.get() and self.trainFalseSetOfParticles.get().getXDim()!=DEEP_PARTICLE_SIZE:
          errorMsg.append("Error, trainFalseSetOfParticles needed to be 128 px")

        if  self.testTrueSetOfParticles.get() and self.testTrueSetOfParticles.get().getXDim()!=DEEP_PARTICLE_SIZE:
          errorMsg.append("Error, testTrueSetOfParticles needed to be 128 px")
          
        if  self.testFalseSetOfParticles.get() and self.testFalseSetOfParticles.get().getXDim()!=DEEP_PARTICLE_SIZE:
          errorMsg.append("Error, testFalseSetOfParticles needed to be 128 px")
          
        if (not self.ignoreCTF.get() and ( len(self.ctfRelations)!= len(self.inputCoordinatesTrue) or 
                                    len(self.ctfRelations)!= len(self.inputCoordinatesFalse) ) ) \
           or len(self.inputCoordinatesTrue)!= len(self.inputCoordinatesFalse):
          errorMsg.append("Error, mismatch in number of coordinate sets (or ")
        return errorMsg

#--------------------------- INSERT steps functions ---------------------------
    def _insertAllSteps(self):

        self.inputMicrographs = None
        self.boxSize = None
        self.coordinatesDict = {}

        self._insertFunctionStep("initializeStep")
        self._insertFunctionStep("preprocessMicsStep")
        self.insertExtractPartSteps('TRUE')
        self.insertExtractPartSteps('FALSE')
        self._insertFunctionStep('trainCNN')
        self._insertFunctionStep('predictCNN')


    def initializeStep(self):
        """
            Create paths where data will be saved
        """
        for i in range(len( self.inputCoordinatesTrue)):
          makePath(self._getExtraPath(self.PRE_PROC_MICs_PATH%i))
          
        if self.doTesting.get() and self.testTrueSetOfParticles.get() and self.testFalseSetOfParticles.get():
          writeSetOfParticles(self.testTrueSetOfParticles.get(),
                              self._getExtraPath("testTrueParticlesSet.xmd"))
          writeSetOfParticles(self.testFalseSetOfParticles.get(),
                              self._getExtraPath("testFalseParticlesSet.xmd"))

        if self.addTrainingData.get() == self.ADD_TRAIN_CUST:
          if self.trainTrueSetOfParticles.get():
            writeSetOfParticles(self.trainTrueSetOfParticles.get(),
                                  self._getExtraPath("trainTrueParticlesSet.xmd"))
          if self.trainFalseSetOfParticles.get():
            writeSetOfParticles(self.trainFalseSetOfParticles.get(),
                                  self._getExtraPath("trainFalseParticlesSet.xmd"))
        elif self.addTrainingData.get() == self.ADD_TRAIN_MODEL:
          writeSetOfParticles(self.retrieveTrainSets(),
                              self._getTmpPath("addNegTrainParticles.xmd"))
      
    def retrieveTrainSets(self):
        """ Retrieve, link and return a setOfParticles
            corresponding to the NegativeTrain DeepConsensus trainning set
            with certain extraction conditions (phaseFlip/invContrast)
        """
        prefixYES = ''
        prefixNO = 'no'
        modelType = "negativeTrain_%sPhaseFlip_%sInvert.mrcs" % (
                    prefixYES if self.doInvert.get() else prefixNO,
                    prefixYES if self.ignoreCTF.get() else prefixNO)
        modelPath = xmipp3.Plugin.getModel("deepConsensus", modelType)
        print("Precompiled negative particles found at %s"%( modelPath))
        modelFn = self._getTmpPath(modelType)
        pwutils.createLink(modelPath, modelFn)

        tmpSqliteSuff = "AddTrain"
        partSet = self._createSetOfParticles(tmpSqliteSuff)
        img = SetOfParticles.ITEM_TYPE()

        imgh = ImageHandler()
        _, _, _, n = imgh.getDimensions(modelFn)
        if n > 1:
          for index in range(1, n + 1):
            img.cleanObjId()
            img.setMicId(9999)
            img.setFileName(modelFn)
            img.setIndex(index)
            partSet.append(img)
        partSet.setAlignment(ALIGN_NONE)

        cleanPath(self._getPath("particles%s.sqlite"%tmpSqliteSuff))
        return partSet

    def _getInputMicrographs(self, setNumber=0):

        if not hasattr(self, "inputMicrographs") or not self.inputMicrographs:
          if len(self.inputCoordinatesTrue)==0:
            print("WARNING. PROVIDE MICROGRAPHS FIRST")
          else:
            self.inputMicrographs = self.inputCoordinatesTrue[setNumber].get().getMicrographs()
        return self.inputMicrographs

    def _getBoxSize(self, setNumber=0):
        if not hasattr(self, "boxSize") or not self.boxSize:
          firstCoords = self.inputCoordinatesTrue[setNumber].get()
          self.boxSize = firstCoords.getBoxSize()
          self.downFactor = self.boxSize / float(DEEP_PARTICLE_SIZE)
        return self.boxSize

    def _getDownFactor(self, setNumber=0):

        if not hasattr(self, "downFactor") or not self.downFactor:
          firstCoords = self._getInputMicrographs(setNumber)
          self.boxSize= firstCoords.getBoxSize()
          self.downFactor = self.boxSize /float(DEEP_PARTICLE_SIZE)
          assert self.downFactor >= 1, \
              "Error, the particle box size must be greater or equal than 128."

        return self.downFactor



    def getMicsIds(self, setNumber=-1):
    
        micIds=set([])
        micFnames= set([])
        if setNumber==-1:
          setNumbers= range(len(self.inputCoordinatesTrue))
        else:
          setNumbers= [setNumber]
        for setNumber in setNumbers:
          for coord in self.inputCoordinatesTrue[setNumber].get():
            micIds.add( coord.getMicId())
            micFnames.add( coord.getMicName() )
          for coord in self.inputCoordinatesFalse[setNumber].get():
            micIds.add( coord.getMicId())
            micFnames.add( coord.getMicName() )
        return sorted( micIds )
      
    def preprocessMicsStep(self):

        for i in range(len(self.inputCoordinatesTrue)):
          coordTrue= self.inputCoordinatesTrue[i].get()
          coordFalse= self.inputCoordinatesFalse[i].get()
          mics_ = coordTrue.getMicrographs()
          samplingRate = mics_.getSamplingRate()
          micIds = self.getMicsIds(i )
          micsFnameSet = { mics_[micId].getMicName(): mics_[micId].getFileName() for micId in micIds }
          if self.ignoreCTF.get():
            preproMicsContent="#mics\n"
            for micName in micsFnameSet:
              preproMicsContent+= "%s\n"%(micsFnameSet[micName])
          else:
            preproMicsContent="#mics ctfs\n"
            setOfMicCtf= self.ctfRelations[i].get()
            if setOfMicCtf.getSize() < len(micsFnameSet):
              raise ValueError("Error, there are different number of CTFs compared to "+
                                "the number of micrographs where particles were picked")
            else:
              assert setOfMicCtf is not None, "Error, CTFs must be provided to compute phase flip"
              for ctf in setOfMicCtf:
                ctf_mic = ctf.getMicrograph()
                ctfMicName = ctf_mic.getMicName()
                if ctfMicName in micsFnameSet:
                  ctf_mic.setCTF(ctf)
                  ctfMicName= micsFnameSet[ctfMicName]
                  fnCTF = self._getTmpPath("%s.ctfParam" % os.path.basename(ctfMicName))
                  micrographToCTFParam(ctf_mic, fnCTF) 
                  preproMicsContent+= "%s %s\n"%(ctfMicName, fnCTF)
               
          inputsFname= self._getExtraPath("preprocMic_inputs_%d.txt"%i)
          ouputDir= self._getExtraPath(self.PRE_PROC_MICs_PATH%i)
          nThrs= self.numberOfThreads.get()
          with open(inputsFname, "w") as f:
            f.write(preproMicsContent)
          downFactor = self._getDownFactor()
          args= "-i %s -s %s -d %s -o %s -t %d"%(inputsFname, samplingRate, downFactor, ouputDir, nThrs)
          if self.doInvert.get():
            args+=" --invert_contrast"
              
          if not self.ignoreCTF.get():
            args+=" --phase_flip"
            
          self.runJob('xmipp_preprocess_mics', args, numberOfMpi=1)
#        raise ValueError("peta")
        
    def insertExtractPartSteps(self, mode):
        
        for i in range(len(self.inputCoordinatesTrue)):
          self._insertFunctionStep("extractParticles", i, mode) 
        self._insertFunctionStep("joinSetOfParticlesStep", mode)

    def extractParticles(self, setNumber, mode):

        if mode=="TRUE":
          coords= self.inputCoordinatesTrue[setNumber].get()
        elif mode=="FALSE":
          coords= self.inputCoordinatesFalse[setNumber].get()
        else:
          raise ValueError("Error, mode has to be either TRUE or FALSE")
        mics_ = coords.getMicrographs()
        samplingRate = mics_.getSamplingRate()
        boxSize = self._getBoxSize(setNumber)

        micsFnameSet = {}
        preprocMicsPath= self._getExtraPath(self.PRE_PROC_MICs_PATH%setNumber)
        for micFname in os.listdir(preprocMicsPath):
          micFnameBase= micFname.split(".")[0]
          micFname= os.path.join(preprocMicsPath, micFname)
          micsFnameSet[micFnameBase]= micFname

        posDir= self._getExtraPath( self.CONSENSUS_COOR_PATH_TEMPLATE%(mode, setNumber) )
        makePath(posDir)
        writeSetOfCoordinates(posDir, coords, scale=1.0 )
        extractCoordsContent="#mics coords\n"
        for posFname in os.listdir(posDir):
          posNameBase=  posFname.split(".")[0]
          posFname= os.path.join(posDir, posFname)
          if posNameBase in micsFnameSet:
            extractCoordsContent+= "%s particles@%s\n"%(micsFnameSet[posNameBase], posFname)
          else:
            print("WARNING, no micFn for coords %s"%(posFname))
          
        inputsFname= self._getExtraPath("extractParticles_inputs_%s_%d.txt"%(mode, setNumber) )
        ouputDir= self._getExtraPath(self.CONSENSUS_PARTS_PATH_TEMPLATE%(mode, setNumber) )
        makePath(ouputDir)
        nThrs= self.numberOfThreads.get()
        with open(inputsFname, "w") as f:
          f.write(extractCoordsContent)
        downFactor= self._getDownFactor()
        args= "-i %s -s %s -d %s -o %s -t %d"%(inputsFname, DEEP_PARTICLE_SIZE, downFactor, ouputDir, nThrs)

        self.runJob('xmipp_extract_particles', args, numberOfMpi=1)   

    def joinSetOfParticlesStep( self, mode):
        #Create images.xmd metadata joining from different .stk
        fnImages = self._getExtraPath("particles_%s.xmd" % mode)
        imgsXmd = MD.MetaData()
        for i in range(len(self.inputCoordinatesTrue)):
          posFiles = glob(self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE%(mode, i), '*.pos'))
          for posFn in posFiles:
            xmdFn = self._getExtraPath(self.CONSENSUS_PARTS_PATH_TEMPLATE%(mode, i),
                                       pwutils.replaceBaseExt(posFn, "xmd"))
            if os.path.exists(xmdFn):
              mdFn = MD.MetaData(xmdFn)
              mdPos = MD.MetaData('particles@%s' % posFn)
              mdPos.merge(mdFn) 
              imgsXmd.unionAll(mdPos)
            else:
              self.warning("The coord file %s wasn't used for extraction! "
                           % os.path.basename(posFn))
              self.warning("Maybe you are extracting over a subset of micrographs")
        imgsXmd.write(fnImages)
        
    def __dataDict_toStrs(self, dataDict):
        fnamesStr=[]
        weightsStr=[]
        for fname in dataDict:
          fnamesStr.append(fname)
          weightsStr.append(str(dataDict[fname]) )
        return ":".join(fnamesStr), ":".join(weightsStr)
        
    def trainCNN(self):

        if self.usesGpu():
          numberOfThreads = None
          gpuToUse = self.getGpuList()[0]
        else:
          numberOfThreads = self.numberOfThreads.get()
          gpuToUse = None
            
        for effectiveSize in [1000, 5000, 50000]:
          netDataPath = self._getExtraPath("nnetData_%d"%effectiveSize)
          makePath(netDataPath)
          nEpochs = self.nEpochs.get()
              
          posTrainDict = {self._getExtraPath("particles_TRUE.xmd"):  1}
          negTrainDict = {self._getExtraPath("particles_FALSE.xmd"):  1}

          if self.addTrainingData.get() == self.ADD_TRAIN_MODEL:
            negTrainDict[self._getTmpPath("addNegTrainParticles.xmd")]= 1
          
          if self.trainTrueSetOfParticles.get():
            posTrainFn = self._getExtraPath("trainTrueParticlesSet.xmd")
            posTrainDict[posTrainFn] = self.trainPosWeight.get()
          if self.trainFalseSetOfParticles.get():
            negTrainFn = self._getExtraPath("trainFalseParticlesSet.xmd")
            negTrainDict[negTrainFn] = self.trainNegWeight.get()

          fnamesPos, weightsPos= self.__dataDict_toStrs(posTrainDict)
          fnamesNeg, weightsNeg= self.__dataDict_toStrs(negTrainDict)
          args= " -n %s --mode train -p %s -f %s --trueW %s --falseW %s"%(netDataPath, 
                        fnamesPos, fnamesNeg, weightsPos, weightsNeg)
          args+= " -e %s -l %s -r %s -m %s --effective_data_size %d"%(nEpochs, self.learningRate.get(), 
                                                                  self.l2RegStrength.get(), 1, effectiveSize)
          if not self.auto_stopping.get():
            args+=" -s"
            
          if not gpuToUse is None:
            args+= " -g %s"%(gpuToUse)
          if not numberOfThreads is None:
            args+= " -t %s"%(numberOfThreads)
          self.runJob('xmipp_deep_consensus', args, numberOfMpi=1)
        
    def predictCNN(self):

        netDataPath = self._getExtraPath('nnetData_1000')

        if self.usesGpu():
          numberOfThreads = None
          gpuToUse = self.getGpuList()[0]
        else:
          numberOfThreads = self.numberOfThreads.get()
          gpuToUse = None

        predictDict = {self._getExtraPath("particles_TRUE.xmd"): 1}
        if self.doTesting.get() and self.testTrueSetOfParticles.get() and self.testFalseSetOfParticles.get():
          posTestDict = {self._getExtraPath("testTrueParticlesSet.xmd"): 1}
          negTestDict = {self._getExtraPath("testFalseParticlesSet.xmd"): 1}
        else:            
          posTestDict = None
          negTestDict = None
        outParticlesPath = self._getPath("particles.xmd")

        fnamesPred, weightsPred= self.__dataDict_toStrs(predictDict)
        
        args= " -n %s --mode score -i %s -o %s "%(netDataPath, fnamesPred, outParticlesPath)
                
        if posTestDict and negTestDict:
          fnamesPosTest, weightsPosTest= self.__dataDict_toStrs(posTestDict)
          fnamesNegTest, weightsNegTest= self.__dataDict_toStrs(negTestDict)
          args+= " --testingTrue %s --testingFalse %s "%(fnamesPosTest, fnamesNegTest)
          
        if not gpuToUse is None:
          args+= " -g %s"%(gpuToUse)
        if not numberOfThreads is None:
          args+= " -t %s"%(numberOfThreads)
        self.runJob('xmipp_deep_consensus', args, numberOfMpi=1)
        
    def createOutputStep(self):
        # PARTICLES
        cleanPattern(self._getPath("*.sqlite"))
        partSet = self._createSetOfParticles()
        readSetOfParticles(self._getPath("particles.xmd"), partSet)
        inputSampling = self.inputCoordinates[0].get().getMicrographs().getSamplingRate()
        partSet.setSamplingRate(self._getDownFactor() * inputSampling)
        boxSize = self._getBoxSize()

        # COORDINATES
        writeSet=False
        if self.checkIfPrevRunIsCompatible("coords_"):
          writeSet=True
        if not "OR" in self.coordinatesDict:
            self.loadCoords(self._getExtraPath(self.CONSENSUS_COOR_PATH_TEMPLATE % 'TRUE'),
                                     'OR', writeSet=False)

        coordSet = SetOfCoordinates(filename=self._getPath("coordinates.sqlite"))
        coordSet.copyInfo(self.coordinatesDict['OR'])
        coordSet.setBoxSize(boxSize)
        coordSet.setMicrographs(self.coordinatesDict['OR'].getMicrographs())

        downFactor = self._getDownFactor()
        for part in partSet:
            coord = part.getCoordinate().clone()
            coord.scale(downFactor)

            deepZscoreLabel = '_xmipp_%s' % xmipp.label2Str(MD.MDL_ZSCORE_DEEPLEARNING1)
            setattr(coord, deepZscoreLabel, getattr(part, deepZscoreLabel))
            coordSet.append(coord)
        
        coordSet.write()
        partSet.write()
        
        self._defineOutputs(outputCoordinates=coordSet)        
        self._defineOutputs(outputParticles=partSet)

        for inSetOfCoords in self.inputCoordinates:
            self._defineSourceRelation(inSetOfCoords.get(), coordSet)
            self._defineSourceRelation(inSetOfCoords.get(), partSet)

#        print("OR", self.coordinatesDict['OR'].getBoxSize())
#        raise ValueError("peta")

    def _summary(self):
        message = []
        for i, coordinates in enumerate(self.inputCoordinatesTrue):
            protocol = self.getMapper().getParent(coordinates.get())
            message.append("Data source  %d %s" % (i + 1, protocol.getClassLabel()))
        message.append("Relative Radius = %f" % self.consensusRadius)
        return message

    def _methods(self):
        return []


    #--------------------------- UTILS functions --------------------------------------------
    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, MD.MDL_ZSCORE_DEEPLEARNING1)
        if row.getValue(MD.MDL_ENABLED) <= 0:
            item._appendItem = False
        else:
            item._appendItem = True
