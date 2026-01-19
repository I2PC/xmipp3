#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Ruben Sanchez Garcia
 *
 * CSIC
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
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/
"""
from xmipp_base import *
import xmippLib
from xmippPyModules.deepLearningToolkitUtils.utils import (checkIf_tf_keras_installed,
                                                                         updateEnviron)

checkIf_tf_keras_installed()

from xmippPyModules.deepConsensusWorkers.deepConsensus_deepLearning1 import (loadNetShape, writeNetShape,
                                                                                           DeepTFSupervised, DataManager,
                                                                                           tf_intarnalError)
WRITE_TEST_SCORES= True

class ScriptDeepScreeningTrain(XmippScript):
    _conda_env= CondaEnvManager.CONDA_DEFAULT_ENVIRON
    def __init__(self):
        XmippScript.__init__(self)
        
    def defineParams(self):
        self.addUsageLine('Cleans sets of particles using Deep Learning. It works in two modes:\n'
                          '* mode 1: Train a neural network using a training dataset of particles. The training set '
                          'consists of one or several sets of true particles and one or several sets of false particles\n'
                          '* mode 2: Score a putative set of particles as good particles or bad particles. An already '
                          'trained network and a set of particles must be provided')
        ## params
        self.addParamsLine(' -n <netDataPath>               : A path where the networks will be saved or loaded. '
                           'If there is an already network in the path, it will be loaded and 1) the training continues '
                           '(2) the putative set of particles can be scored')
                           
        self.addParamsLine(' --mode <mode> : "training"|"scoring". Select training or scoring mode')

        self.addParamsLine('[ -g <gpuId>  ]               : GPU Id. By default no gpu will be used')   
        self.addParamsLine('[ -t <numThreads>  <N=2>  ]   : Number of threads')
        
        
        self.addParamsLine("== Scoring mode ==")
        self.addParamsLine('[ -i <trueTrainData>   ]      : A path to metada particles (xmd) to be scored.'
                           'already trained' )
        
        self.addParamsLine('[ -o <trueTrainData>   ]      : A text file where metada particles (xmd) predictions will be saved. '
                           'Requires -n already trained' )

        self.addParamsLine('[ --testingTrue <trueTestData>   ]  : A path to metada true particles (curated) (xmd) to be used for '
                           'evaluation purposes' )
        self.addParamsLine('[ --testingFalse <falseTestData>   ]  : A path to metada false particles (curated) (xmd) to be used for '
                           'evaluation purposes' )

        self.addParamsLine("== Training mode ==")
        self.addParamsLine('[ -p <trueTrainData>  ]       : Path to training positive metada particles (xmd). '
                           'if many paths, they must be separated by ":" e.g /path1/parts1.xmd:/path2/parts2.xmd')
        
        self.addParamsLine('[ -f <falseTrainData>  ]      : Path to training negative metada particles (xmd). ' 
                           'if many paths, they must be separated by ":" e.g /path1/parts1.xmd:/path2/parts2.xmd')

        self.addParamsLine('[ --trueW <trueTrainWeights> ]     : Weights for positive particles datasets (int). ' 
                           '-1 for autoweight. if many paths, the must be separated by ":" e.g 2:3. Default -1:-1:...')
                           
        self.addParamsLine('[ --falseW <falseTrainWeights> ]   : Weights for negative particles datasets (int). ' 
                           '-1 for autoweight. if many paths, the must be separated by ":" e.g 2:3. Default -1:-1:...')
                           
        self.addParamsLine('[ -e <numberOfEpochs> <N=5> ]   : Number of epochs to train the network')
        
        self.addParamsLine('[ -l <learningRate>  <F=1e-4> ] : Learning rate to train the network')
        
        self.addParamsLine('[ -r <regularizationStrengh> <F=1e-5> ] : l2 regularization to train the network')
        
        self.addParamsLine('[ -s <autoStop>  ]                      : Do not stop training when convengercy is '
                           'detected')          
        self.addParamsLine('[ -m <numberOfModels> <N=2> ]           : Number of models used as ensemble')
        self.addParamsLine('[ --effective_data_size <numberOfModels> <N=-1> ] : Number of effective data points for '
                           ' training. If -1 use the number of true particles')        
        
        ## examples
        self.addExampleLine('trainNet net:  xmipp_deep_screen -n ./netData --train_mode -p trueParticles.xmd -f '
                            'falseParticles1.xmd:falseParticles2.xmd -g 0')

        self.addExampleLine('predict particles:  xmipp_deep_screen -n ./netData --score_mode -i unknownParticles.xmd -o '
                            'unknownPredictions.txt -g 0')
    def run(self):

        numberOfThreads=self.getIntParam('-t')
        gpuToUse=None
        if self.checkParam('-g'):
          gpuToUse= self.getIntParam('-g')
          numberOfThreads= None
        netDataPath= self.getParam('-n')
        
        mode= self.getParam('--mode')
        trainKeyWords=["train", "training"]
        predictKeyWords=["predict", "score", "scoring"]

        if mode in trainKeyWords:
          print("mode 1: training")
          trueDataPaths= self.getParam('-p').split(":")
          falseDataPaths= self.getParam('-f').split(":")
          
          trueWeights=  [-1 for elem in falseDataPaths]
          falseWeights= [-1 for elem in falseDataPaths]
          if self.checkParam('--trueW'):
            trueWeights= self.getParam('--trueW').split(":")
            assert len(trueWeights)==len(trueDataPaths), "Error, the number of weights provided --falseW does not match the "+\
                                                            "number of negative data paths provided"
          if self.checkParam('--falseW'):
            falseWeights= self.getParam('--falseW').split(":")
            assert len(falseWeights)==len(falseDataPaths), "Error, the number of weights provided --falseW does not match the "+\
                                                            "number of negative data paths provided"
                                                            
          nEpochs= self.getDoubleParam('-e')
          learningRate= self.getDoubleParam('-l')
          l2RegStrength= self.getDoubleParam('-r')
          auto_stop=True
          if self.checkParam('-s'):
            auto_stop= False
          
          numModels=self.getIntParam('-m')
          effective_data_size= self.getIntParam("--effective_data_size")
          posTrainDict= {path: weight for path, weight in zip(trueDataPaths, trueWeights) }
          negTrainDict= { path: weight for path, weight in zip(falseDataPaths, falseWeights) }
          trainArgs={"netDataPath":netDataPath, "posTrainDict":posTrainDict, "negTrainDict":negTrainDict, 
                     "nEpochs":nEpochs, "learningRate":learningRate, "l2RegStrength":l2RegStrength,
                     "auto_stop":auto_stop, "numModels":numModels , "effective_data_size":effective_data_size,
                     "gpuToUse": gpuToUse, "numberOfThreads":numberOfThreads}
          print("TRAIN ARGS: %s"%(trainArgs) )
          ScriptDeepScreeningTrain.trainWorker(**trainArgs)
        elif mode in predictKeyWords:
          print("mode 2: scoring")
          predictDict= { self.getParam('-i'):1 }
          outParticlesPath= self.getParam('-o')
          posTestDict, negTestDict= (None, None)
          if self.checkParam('--testingTrue') and self.checkParam('--testingFalse'):
            posTestDict= { self.getParam('--testingTrue'):1 }
            negTestDict= { self.getParam('--testingFalse'):1 }
                  
          scoringArgs={"netDataPath":netDataPath, "predictDict":predictDict, "outParticlesPath":outParticlesPath, 
                     "posTestDict":posTestDict, "negTestDict":negTestDict,
                     "gpuToUse": gpuToUse, "numberOfThreads":numberOfThreads}
          print("SCORING ARGS: %s"%(scoringArgs) )
          ScriptDeepScreeningTrain.predictWorker(**scoringArgs)
        else:
          raise Exception("Error, --mode must be set to 'training' or 'scoring'")

    @staticmethod
    def trainWorker(netDataPath, posTrainDict, negTrainDict, nEpochs, learningRate,
                    l2RegStrength, auto_stop, numModels, effective_data_size, gpuToUse,
                    numberOfThreads):
        '''
            netDataPath=self._getExtraPath("nnetData")
            posTrainDict, negTrainDict: { fnameToMetadata:  weight:int }
                      e.g. : {'Runs/006003_XmippProtScreenDeepLearning1/extra/negativeSet_1.xmd': 1}
            learningRate: float
        '''
 
        if gpuToUse >= 0:
          numberOfThreads = None

        else:
          gpuToUse = None
          if numberOfThreads < 0:
            import multiprocessing          
            numberOfThreads = multiprocessing.cpu_count()

        updateEnviron(gpuToUse)

        trainDataManager = DataManager(posSetDict=posTrainDict,
                                       negSetDict=negTrainDict)
                                       
        dataShape_nTrue_numModels= loadNetShape(netDataPath)
        if dataShape_nTrue_numModels:
          dataShape, nTrue, numModels= dataShape_nTrue_numModels
          assert dataShape == trainDataManager.shape, \
                  "Error, data shape mismatch in input data compared to previous model"
        else:
          nTrue= effective_data_size if effective_data_size>0 else trainDataManager.nTrue

        writeNetShape(netDataPath, trainDataManager.shape, nTrue, numModels)
        if nEpochs == 0:
          print("training is not required. If more training desired select more epochs '-e'")
          return
          
        assert numModels >=1, "Error, nModels<1"
        try:
            nnet = DeepTFSupervised(numberOfThreads= numberOfThreads, rootPath= netDataPath,
                                    numberOfModels=numModels, effective_data_size=effective_data_size)
            nnet.trainNet(nEpochs, trainDataManager, learningRate,
                          l2RegStrength, auto_stop)
        except tf_intarnalError as e:
            if e._error_code == 13:
                raise Exception("Out of gpu Memory. gpu # %d"%(gpuToUse))
            else:
                raise e
        del nnet

    @staticmethod
    def predictWorker(netDataPath, posTestDict, negTestDict, predictDict,
                      outParticlesPath, gpuToUse, numberOfThreads):
        '''
            outParticlesPath= self._getPath("particles.xmd")
            posTestDict, negTestDict, predictDict: { fnameToMetadata:  weight:int }
        ''' 
        if gpuToUse >= 0:
            numberOfThreads = None
        else:
            gpuToUse = None
            if numberOfThreads < 0:
                import multiprocessing          
                numberOfThreads = multiprocessing.cpu_count()

        updateEnviron(gpuToUse)

        predictDataManager = DataManager(posSetDict=predictDict, negSetDict=None)
        dataShape_nTrue_numModels = loadNetShape(netDataPath)
        if dataShape_nTrue_numModels:
          numModels= dataShape_nTrue_numModels[-1]
        else:
          numModels=1
        try:
            nnet = DeepTFSupervised(numberOfThreads=numberOfThreads,
                                    rootPath=netDataPath,
                                    numberOfModels=numModels)
        except tf_intarnalError as e:
            if e._error_code == 13:
                raise Exception("Out of GPU Memory. GPU # %d" % (gpuToUse))
            else:
                raise e

        y_pred, label_Id_dataSetNumIterator = nnet.predictNet(predictDataManager)

        metadataPosList, metadataNegList = predictDataManager.getMetadata(None)
        for score, (isPositive, mdId, dataSetNumber) in zip(y_pred, label_Id_dataSetNumIterator):
            if isPositive==True:
                metadataPosList[dataSetNumber].setValue( xmippLib.MDL_ZSCORE_DEEPLEARNING1, score, mdId)
            else:
                metadataNegList[dataSetNumber].setValue( xmippLib.MDL_ZSCORE_DEEPLEARNING1, score, mdId)

        assert len(metadataPosList) == 1, \
                "Error, predict setOfParticles must contain one single object"

        metadataPosList[0].write(outParticlesPath)
        if posTestDict and negTestDict:
            testDataManager = DataManager(posSetDict=posTestDict,
                                          negSetDict=negTestDict,
                                          validationFraction=0)
            print("Evaluating test set")
            global_auc, global_acc, y_labels, y_pred_all = nnet.evaluateNet(testDataManager)
            if WRITE_TEST_SCORES:
              with open(os.path.join(netDataPath, "testPredictions.txt"), "w") as f:
                  f.write("label score\n")
                  for l, s in zip(y_labels, y_pred_all):
                      f.write("%d %f\n" % (l, s))
    
if __name__ == '__main__':
    '''
cd ~/ScipionUserData/projects/10028_mini
scipion python `which xmipp_deep_consensus` -g 0 -n /home/rsanchez/tmp/xmipp/xmippNNet --mode train -p Runs/000660_XmippProtScreenDeepLearning1/extra/inputTrueParticlesSet.xmd -f Runs/000660_XmippProtScreenDeepLearning1/extra/negativeSet_1.xmd 

scipion python `which xmipp_deep_consensus` -g 0 -n /home/rsanchez/tmp/xmippNNet --mode predict -i Runs/000660_XmippProtScreenDeepLearning1/extra/inputTrueParticlesSet.xmd -o ~/tmp/xmipp/out.xmd
    '''
    exitCode=ScriptDeepScreeningTrain().tryRun()
    sys.exit(exitCode)
    
