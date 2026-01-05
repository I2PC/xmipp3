#!/usr/bin/env python3
""""
**************************************************************************
*
* Authors:  Mikel Iceta Tena (miceta@cnb.csic.es)
* 
*
* Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
* 02111-1307  USA
*
*  All comments concerning this program package may be sent to the
*  e-mail address 'scipion@cnb.csic.es'
*
* Initial version: sept 2023
**************************************************************************
"""
import sys
import os
import numpy as np
from keras.optimizers import adam_v2
from xmipp_base import XmippScript
import xmippLib

from xmippPyModules.deepPickingConsensusTomo.deepPickingConsensusTomo_networks_sx import NetMan

MODEL_TRAIN_NEW         = 0
MODEL_TRAIN_PRETRAIN    = 1
MODEL_TRAIN_PREVRUN     = 2
MODEL_TRAIN_TYPELIST    = ["From scratch", "Existing model", "Previous run"]

NN_TRAINWORDS           = ["train", "training", "t"]
NN_SCOREWORDS           = ["score", "scoring", "s", "predict"]
NN_NAME                 = "dpc_nn.h5"

DEFAULT_MP              = 8

class ScriptDeepConsensus3D(XmippScript):
    
    _conda_env = "xmipp_DLTK_v1.0"

    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('DeepConsensus3D. Launches a CNN to process cryoET (tomo) subvolumes.\n'
                          'It can be used in these cases:\n'
                          '1) Train network from scratch\n'
                          '2) Load a network and keep training\n'
                          '3) Load a network and score\n'
                          'Keep in mind that training and scoring are separated, so two separate\n'
                          'calls to this program are needed for a train+score scenario')

        # Application parameters
        self.addParamsLine('==== Application ====')
        self.addParamsLine('[ -t <numThreads=4> ] : Number of threads')
        self.addParamsLine(' -g <gpuId> : comma separated GPU Ids. Set to -1 to use all CUDA_VISIBLE_DEVICES') 
        self.addParamsLine(' --mode <execMode> : training or scoring')
        self.addParamsLine(' --netpath <netpath> : path for network models read/write (needed in any case)')
        self.addParamsLine(' --batchsize <size=16> : amount of images that will be fed each time to the network.')
        self.addParamsLine('[ --netname <filename> ] : filename of the network to load, only for train-pretrain or score-pretrain')

        # Tomo
        self.addParamsLine('==== Tomo ====')
        self.addParamsLine(' --consboxsize <boxsize> : desired box size (int)')
        self.addParamsLine(' --conssamprate <samplingrate> : desired sampling rate (float)')

        # Score parameters
        self.addParamsLine('==== Scoring mode ====')
        self.addParamsLine('[ --inputvolpath <path> ] : path to the metadata files of the input doubtful subtomos (mrc)')
        self.addParamsLine('[ --outputpath <path> ] : path for the program to write the scored coordinates (xmd)')

        # Train parameters
        self.addParamsLine('==== Training mode ====')
        self.addParamsLine('[ --ttype <traintype=0> ] : train mode')
        self.addParamsLine('[ --valfrac <fraction=0.15> ] : fraction of the labeled dataset to use in validation.')
        self.addParamsLine('[ --truevolpath <truevolpath> ] : path to the positive subtomos (mrc)')
        self.addParamsLine('[ --falsevolpath <falsevolpath> ] : path to the negative subtomos (mrc)')
        self.addParamsLine('[ -e <numberOfEpochs=5> ]  : Number of training epochs (int).')
        self.addParamsLine('[ -l <learningRate=0.0001> ] : Learning rate (float).')
        self.addParamsLine('[ -r <regStrength=0.00001> ] : L2 regularization level (float).')
        self.addParamsLine('[ -s ] : Autostop on convergency detection.')
        self.addParamsLine('[ --ensemble <numberOfModels=1> ] : If set, an ensemble of models will be used in a voting instead one.')

    def parseParams(self):
        """
        This function does the reading of input flags and parameters. It sanity-checks all
        inputs to make sure the program does not unnecesarily crash later.
        """

        # Default for CPU threads
        if self.checkParam('-t'):
            self.numThreads = self.getIntParam('-t')
        else:
            self.numThreads = DEFAULT_MP
        # GPU acceleration - assuming GPU is always used
        gpustr : str = self.getParam('-g')
        self.gpus : list = [ int(item) for item in gpustr.split(",")]
        # -1 means all CUDA_VISIBLE_DEVICES must be used
        if -1 in self.gpus:
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                gpustr = os.environ.get('CUDA_VISIBLE_DEVICES')
                self.gpus : list = [int(item) for item in gpustr.split(",")]
        # Execution mode
        self.execMode = str(self.getParam('--mode'))
        # Netpath
        self.netPath = self.getParam('--netpath')
        if not os.path.isdir(self.netPath):
            print("Network path is not a valid path")
        # Netname
        if self.checkParam('--netname'):
            self.netName = self.getParam('--netname')
            self.netPointer = os.path.join(self.netPath, self.netName)
            if not os.path.isfile(self.netPath+self.netName):
                print("NN file does not exist inside path")
        else:
            self.netName = NN_NAME
        # Consensuated boxsize and sampling ratesize
        self.consBoxSize : int = self.getIntParam('--consboxsize')
        self.consSampRate : float = self.getDoubleParam('--conssamprate')
        # Desired batch size
        self.batchSize : int = self.getIntParam('--batchsize')
       
        # The desired running mode is training
        if self.execMode.strip() in NN_TRAINWORDS:
            self.execMode = "train"
            print("Execution mode is: TRAINING", flush=True)

            # Training type
            self.trainType = int(self.getParam('--ttype'))
            # Read paths
            self.posPath : str = self.getParam('--truevolpath')
            self.negPath : str = self.getParam('--falsevolpath')
            self.doubtPath = None
            # Learning rate
            self.learningRate = float(self.getDoubleParam('-l'))
            
            # Epochs for training
            if self.checkParam('-e'):
                self.nEpochs = int(self.getIntParam('-e'))
            else:
                self.nEpochs = 5
            # Regularization strenght
            if self.checkParam('-r'):
                self.regStrength= float(self.getDoubleParam('-r'))
            else:
                self.regStrength = 1.0e-5
            # Auto stop feature
            self.autoStop = self.checkParam('-s')
            # Ensemble amount
            if self.checkParam('--ensemble'):
                self.ensemble = self.getIntParam('--ensemble')
            else:
                self.ensemble = 1
            # Check if it's in the correct range and set to default if needed
            if self.trainType not in [0, 1, 2]:
                print("Training mode %d not recognized. Running a new model instead." % self.trainType)
                self.trainType = MODEL_TRAIN_NEW
            else:
                print("Training in mode: " +  MODEL_TRAIN_TYPELIST[self.trainType])
            # Validation fraction
            self.valFrac = self.getDoubleParam('--valfrac')

        # The desired mode is scoring
        elif self.execMode.strip() in NN_SCOREWORDS:
            print("Execution mode is: SCORING", flush=True)
            self.execMode = "score"

            if self.checkParam('-l'):
                self.learningRate = float(self.getDoubleParam('-l'))
            else:
                self.learningRate = 1.0e-5

            # Input/Output
            self.doubtPath = str(self.getParam('--inputvolpath'))
            self.posPath = None
            self.negPath = None
            if not os.path.exists(self.doubtPath):
                print("Path to input subtomograms does not exist. Exiting.")
                sys.exit(-1)
            self.outputFile = str(self.getParam('--outputpath'))
            
    def run(self):
        '''
        Instantiates the data managing class object (DataMan) and then launches the appropriate
        program to train or score with the neural network.
        '''
        print("deep_picking_consensus_tomo.py is launched\nParsing input...", flush=True)
        self.parseParams()
        print("Execution will be done using %d threads." % self.numThreads, flush = True)
        gpustring = str(self.gpus)
        print("Execution will use GPUS with ID: " + gpustring, flush=True)

        if self.execMode == "train":
            self.doTrain()
        elif self.execMode == "score":
            self.doScore()
        sys.exit(0)

    def doTrain(self):
        
        netMan = NetMan(nThreads = self.numThreads, gpuIDs = self.gpus, rootPath = self.netPath,
                        batchSize = self.batchSize, boxSize = self.consBoxSize, posPath = self.posPath, negPath = self.negPath,
                        doubtPath = self.doubtPath, netName = self.netName)

        # Generate or load a model, depending on what is wanted
        if self.trainType == MODEL_TRAIN_NEW:
            amount_of_data = 1
            netMan.createNetwork(self.consBoxSize, amount_of_data)
            # TODO: use amount_of_data correctly here and in netman
        elif self.trainType == MODEL_TRAIN_PRETRAIN:
            netMan.loadNetwork(modelFile = self.netPointer)
        else:
            print("Specified NN training mode yet implemented, use new or pretrained")
            exit(-1)
        
        # netMan.net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        netMan.compileNetwork(pLoss='binary_crossentropy', pOptimizer=adam_v2.Adam(learning_rate=self.learningRate), pMetrics=['accuracy'])
        netMan.trainNetwork(nEpochs = self.nEpochs, learningRate = self.learningRate, autoStop=True)

    def doScore(self):
        netMan = NetMan(nThreads = self.numThreads, gpuIDs = self.gpus, rootPath = self.netPath,
                        batchSize = self.batchSize, boxSize = self.consBoxSize, posPath = self.posPath, negPath = None,
                        doubtPath = self.doubtPath, netName = self.netName)
        netMan.loadNetwork(modelFile = self.netPointer)
        netMan.compileNetwork(pLoss='binary_crossentropy', pOptimizer=adam_v2.Adam(learning_rate=self.learningRate), pMetrics=['accuracy'])
        netMan.predictNetwork()
        netMan.writeScoresXMD(self.outputFile)
    
    def writeResults(self, results, file):
        # Write results vector in numpy format TXT
        txtname = file + ".txt"
        np.savetxt(txtname, results)

        md = xmippLib.MetaData()
        for item in results:
            row_id = md.addObject()
            # Save the coordinates of the representative
            md.setValue(xmippLib.MDL_XCOOR, int(), row_id)
            md.setValue(xmippLib.MDL_YCOOR, int(), row_id)
            md.setValue(xmippLib.MDL_ZCOOR, int(), row_id)
            # Save the probability
            # TODO: QUE XXXXXX ES ESE XXXX? MIRAR PROBABILIDADES
            # md.setValue(xmippLib.XXXXXXXXXX, float(), row_id)
            # A futuro: interesante guardar cuantas representa
        
        md.write(file)
        print("Written predictions to " + file)

if __name__ == '__main__':
    ScriptDeepConsensus3D().tryRun()