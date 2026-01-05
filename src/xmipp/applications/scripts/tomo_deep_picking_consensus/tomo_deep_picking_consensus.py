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

from xmippPyModules.deepPickingConsensusTomo.dpc3d_data_manager import DataManDPC3D
from xmippPyModules.deepPickingConsensusTomo.dpc3d_parameter_manager import ParaManDPC3D
from xmippPyModules.deepPickingConsensusTomo.dpc3d_model_manager import ModelManDPC3D


MODEL_TRAIN_NEW         = 0
MODEL_TRAIN_PRETRAIN    = 1
MODEL_TRAIN_PREVRUN     = 2
MODEL_TRAIN_TYPELIST    = ["From scratch", "Existing model", "Previous run"]

NN_TRAINWORDS           = ["train", "training", "t"]
NN_SCOREWORDS           = ["score", "scoring", "s", "predict"]
NN_NAME                 = "dpc_nn.h5"

DEFAULT_MP              = 16

class ScriptDeepConsensus3D(XmippScript):
    
    _conda_env = "xmipp_DLTK_v1.0"
    n_cpus : int
    s_gpus : str
    l_gpus : list[int]
    exec_mode : str
    net_path : str
    net_name : str
    net_pointer : str
    cons_box_size : tuple[int, int, int]
    cons_samp_rate : float 
    cons_batch_size: int
    tight_box_size: bool
    
    nn_train_type : int
    nn_epochs : int
    nn_reg : float
    nn_ensemble : bool
    nn_learning_rate : float
    nn_val_frac : float

    path_pos : str
    path_neg : str
    path_doubt : str

    paraman : ParaManDPC3D
    dataman : ParaManDPC3D
    modelman : ModelManDPC3D

    def __init__(self) -> None:
        XmippScript.__init__(self)

    def defineParams(self) -> None:
        self.addUsageLine('DeepConsensus3D. Launches a CNN to process cryoET (tomo) subvolumes.\n'
                          'It can be used in these cases:\n'
                          '1) Train network from scratch\n'
                          '2) Load a network and keep training\n'
                          '3) Load a network and score\n'
                          'Keep in mind that training and scoring are separated, so two separate\n'
                          'calls to this program are needed for a train+score scenario')

        # Application parameters
        self.addParamsLine('==== Application ====')
        self.addParamsLine('[ -t <numThreads=16> ] : Number of threads')
        self.addParamsLine(' -g <gpuId> : comma separated GPU Ids. Set to -1 to use all CUDA_VISIBLE_DEVICES') 
        self.addParamsLine(' --mode <execMode> : training or scoring')
        self.addParamsLine(' --netpath <netpath> : path for network models read/write (needed in any case)')
        self.addParamsLine('[ --netname <filename> ] : filename of the network to load, only for train-pretrain or score-pretrain')

        # Tomo
        self.addParamsLine('==== Tomo ====')
        self.addParamsLine(' --consboxsize <boxsize> : box size (int)')
        self.addParamsLine(' --conssamprate <samplingrate> : sampling rate (float)')
        self.addParamsLine('[ --tightboxsize ] : to be added if the box is tightly bound to the ROI.')

        # Train parameters
        self.addParamsLine('==== Training mode ====')
        self.addParamsLine('[ --ttype <traintype=0> ] : train mode')
        self.addParamsLine('[ --valfrac <fraction=0.15> ] : fraction of the labeled dataset to use in validation.')
        self.addParamsLine('[ --truevolpath <truevolpath> ] : path to the positive subtomos (mrc)')
        self.addParamsLine('[ --falsevolpath <falsevolpath> ] : path to the negative subtomos (mrc)')
        self.addParamsLine('[ -e <numberOfEpochs=5> ]  : Number of training epochs (int).')
        self.addParamsLine('[ -l <learningRate=0.0001> ] : Learning rate (float).')
        self.addParamsLine('[ -r <regStrength=0.00001> ] : L2 regularization level (float).')
        self.addParamsLine('[ --ensemble <numberOfModels=1> ] : If set, an ensemble of models will be used in a voting instead one.')

        # Score parameters
        self.addParamsLine('==== Scoring mode ====')
        self.addParamsLine('[ --inputvolpath <path> ] : path to the metadata files of the subtomos (mrc) to be scored')
        self.addParamsLine('[ --outputpath <path> ] : path for the program to write the scored coordinates (xmd)')

    def parseParams(self) -> None:
        """
        This function does the reading of input flags and parameters. It sanity-checks all
        inputs to make sure the program does not unnecesarily crash later.
        """

        # CMD or default for CPU threads
        if self.checkParam('-t'):
            self.n_cpus = self.getIntParam('-t')
        else:
            self.n_cpus = DEFAULT_MP

        # GPU Management
        if not self.checkparam('-g'):
             print("No GPUs were specified, but this program requires GPU. Exiting...")
             sys.exit(-1)
        # What does CMD say?
        self.s_gpus : str = self.getParam('-g')
        self.gpus = [ int(item) for item in self.s_gpus.split(",")]

        # Use all GPU option
        if -1 in self.gpus:
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                envvar_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
                self.s_gpus =  envvar_cuda if envvar_cuda is not None else ""
                self.l_gpus = [int(item) for item in self.s_gpus.split(",")]
            else:
                print("CUDA_VISIBLE_DEVICES is not present, but program was asked to infer all GPUs."
                       "Please set the GPU to other than -1 and run again. Exiting...")
                sys.exit(-1)

        # Network files
        self.net_path = self.getParam('--netpath')
        if not os.path.isdir(self.net_path):
            print("Network path is not a valid path")
            sys.exit(-1)
        if self.checkParam('--netname'):
            self.net_name = self.getParam('--netname') # Get from CLI
        else:
            self.net_name = NN_NAME # Set default
        self.net_pointer = os.path.join(self.net_path, self.net_name)
        if not os.path.isfile(self.net_pointer):
                print(f"NN model file {self.net_pointer} was not found")
                print("Loading will fail and training will create/override the file")

        # Data information
        self.cons_box_size = self.getIntParam('--consboxsize')
        self.cons_samp_rate = self.getDoubleParam('--conssamprate')
        if self.checkParam('--tightboxsize'): # Will affect bounding boxes and patching algorithm
            self.tight_box_size = True
        else:
            self.tight_box_size = False

        self.nn_learning_rate = float(self.getDoubleParam('-l'))

        # Training or scoring?
        mode : str = self.getParam('--mode')
        if mode.strip() in NN_TRAINWORDS: # TRAIN NEW OR EXISTING
            self.exec_mode = "train"
            print("Execution mode is TRAINING")
            self.train_type = int(self.getParam('--ttype'))
            self.path_pos = self.getParam('--truevolpath')
            self.path_neg = self.getParam('--falsevolpath')
            self.nn_val_frac = self.getDoubleParam('--valfrac')

            # Epochs for training
            if self.checkParam('-e'):
                self.nEpochs = int(self.getIntParam('-e'))
            else:
                self.nEpochs = 20
            # Regularization strenght
            if self.checkParam('-r'):
                self.regStrength= float(self.getDoubleParam('-r'))
            else:
                self.regStrength = 1.0e-5

            # Ensemble amount
            if self.checkParam('--ensemble'):
                self.ensemble = self.getIntParam('--ensemble')
            else:
                self.ensemble = 1
            
            # Guard - Check if it's in the correct range and set to default if needed
            if self.trainType not in [0, 1, 2]:
                print("Training mode %d not recognized. Running a new model instead lol." % self.trainType)
                self.trainType = MODEL_TRAIN_NEW
            else:
                print("Training in mode: " +  MODEL_TRAIN_TYPELIST[self.trainType])
            

        elif mode.strip() in NN_SCOREWORDS: # SCORE MODE
            self.exec_mode = "score"
            print("Execution mode is INFERENCE/SCORING")
            # In scoring, we want all input data in one set (doubt)
            self.path_doubt = str(self.getParam('--inputvolpath'))
            if not os.path.exists(self.doubtPath):
                print("Path to input subtomograms does not exist. Exiting.")
                sys.exit(-1)
            self.outputFile = str(self.getParam('--outputpath'))  
        else:
            print("Unrecognized --mode parameter. Exiting...")
            sys.exit(-1)
        
    def run(self) -> None:
        """
        Logic for the main loop of the application. Selects running types, finds configurations using libraries
        and dispatches the actual work.
        """

        print("XMIPP3 DPC3D is running")
        print("Parsing inputs...")
        self.parseParams()
        print(f"Threds: {self.n_cpus} and GPUs:{self.s_gpus}")

        if self.exec_mode == "train":
            self.do_train()
        if self.exec_mode == "score":
            self.do_score()
        sys.exit(0)

    def do_train(self) -> None:
        # TODO Steps:
        # Instantiate a NN
        # Generate the settings (patch, batch, etc)
        # Generate a dataloader with specified settings
        # Activate the data augmentation
        # Train on the specified data
        # Save the checkpoints, final result.
        pass

    def do_score(self) -> None:
        # TODO Steps:
        # Find the NN weights file, load them
        # Ensure loaded model is compatible with data shape
        # Score the data
        pass

    def do_write(self) -> None:
        # TODO Steps:
        # np.savetxt the information
        # xmippLib per-row add object to MD and write predictions
        pass

if __name__ == '__main__':
    ScriptDeepConsensus3D().tryRun()