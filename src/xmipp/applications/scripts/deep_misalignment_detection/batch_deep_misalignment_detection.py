#!/usr/bin/env python3
import sys
import os

import numpy as np

import xmippLib
from xmipp_base import XmippScript
from xmippLib import Image

from tensorflow.keras.models import load_model

class ScriptDeepMisalignmentDetection(XmippScript):
    _conda_env="xmipp_DLTK_v1.0" 

    def __init__(self):

        XmippScript.__init__(self)
        
        self.inputModel1 = None
        self.inputModel2 = None
        self.subtomoFilePath = None
        self.misaliThrBool = None
        self.misaliThr = None
        self.outputXmdFilePath = None
        self.misalignmentCriteria = 1  # Use mean mode

    #  --------------------- DEFINE PARAMS -----------------------------
    def defineParams(self):
        # Description
        self.addUsageLine('Detect artifacted tomographic reconstruction from extracted fiducial markers')
        
        # Params
        self.addParamsLine(' --subtomoFilePath <subtomoFilePath>: file path of the xmd file containg the coordiantes of the extracted subtomos. ' +
                           'The extracted subtomo should be in the same folder with the same basename + "-[numberOfSubtomo]." ' +
                           'This is the output got when extracting with xmipp_tomo_extract_subtomograms.')
        self.addParamsLine(' --misaliThr <misaliThr>: Threshold to settle if a tomogram presents weak or strong misalignment. If this value is '
                           'not provided two output set of tomograms are generated, those discarded which present '
                           'strong misalignment and those which do not. If this value is provided the second group of '
                           'tomograms is splitted into two, using this threshold to settle if the tomograms present'
                           'or not a weak misalignment.')
        self.addParamsLine(' -g <gpuId> : comma separated GPU Ids. Set to -1 to use all CUDA_VISIBLE_DEVICES') 
        self.addParamsLine(' [--misalignmentCriteriaVotes]: Define criteria used for making a decision on the presence '
                           'of misalignment on the tomogram based on the individual scores of each subtomogram. If '
                           'this option is not provided (default) the mean of this scores is calculated. If '
                           'provided a votting system based on if each subtomo score is closer to 0 o 1 is implented. ')

        
        # Examples       
        self.addExampleLine('xmipp_deep_misalingment_detection --subtomoFilePath path/to/coords.xmd --misaliThr 0.45 -g 0')

    
    #  --------------------- I/O FUNCTIONS -----------------------------
    def readInputParams(self):
        
        self.inputModel1 = self.getModel("deepTomoMisalignment", "xmipp_FS_phc_model.10-0.01.h5")
        self.inputModel2 = self.getModel("deepTomoMisalignment", "xmipp_SS_phc_model.17-0.25.h5")

        self.subtomoFilePath = self.getParam('--subtomoFilePath')

        self.misaliThrBool = self.checkParam('--misaliThr')
        if self.misaliThrBool:
            self.misaliThr = self.getDoubleParam('--misaliThr')

        self.misalignmentCriteriaVotesBool = self.checkParam('--misalignmentCriteriaVotes')
        if self.misalignmentCriteriaVotesBool:
            self.misalignmentCriteria = 0


        self.outputSubtomoXmdFilePath = os.path.join(os.path.dirname(self.subtomoFilePath), "misalignmentSubtomoStatistics.xmd")
        self.outputTomoXmdFilePath = os.path.join(os.path.dirname(self.subtomoFilePath), "misalignmentTomoStatistics.xmd")

        # Set CUDA devices
        gpustr = self.getParam('-g')
        gpus = [int(item) for item in gpustr.split(",")]

        if -1 not in gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpustr


    def generateOutputXmd(self, overallPrediction, predictionAverage, firstPredictionArray, secondPredictionArray):
        
        print("Writting output subtomo stats at " + self.outputSubtomoXmdFilePath)

        # Write subtomo statistics
        mData1 = xmippLib.MetaData()

        for i in range(len(firstPredictionArray)):
            md_id = mData1.addObject()
            mData1.setValue(xmippLib.MDL_MAX, float(firstPredictionArray[i]),  md_id)
            mData1.setValue(xmippLib.MDL_MIN, float(secondPredictionArray[i]), md_id)
        
        mData1.write(self.outputSubtomoXmdFilePath)

        # Write tomo statistics
        print("Writting output tomo stats at " + self.outputTomoXmdFilePath)

        mData2 = xmippLib.MetaData()

        md_id = mData2.addObject()
        mData2.setValue(xmippLib.MDL_MAX, float(overallPrediction),  md_id)
        mData2.setValue(xmippLib.MDL_MIN, float(predictionAverage), md_id)

        mData2.write(self.outputTomoXmdFilePath)


    #  --------------------- MAIN FUNCTIONS -----------------------------
    def getSubtomoPathList(self, coordFilePath):
        coordFilePath_noExt = os.path.splitext(coordFilePath)[0]
        counter = 1

        subtomoPathList = []

        while True:
            subtomoPath = coordFilePath_noExt + '-' + str(counter) + '.mrc'

            if not os.path.exists(subtomoPath):
                break

            subtomoPathList.append(subtomoPath)
            counter += 1

        return subtomoPathList
    
    def makePrediction(self, subtomoPathList):
        """
        :param subtomoPathList: list to every subtomo extracted to be analyzed
        :return: overallPrediction: alignment statement for the whole tomograms obtained from the estimations of each
        subtomo:
            1: strong misalignment (first split negative)
            2: weak misalignment (second split negative). Implies the existence of an input alignment threshold
            3: alignment (second split positive)
        """

        numberOfSubtomos = len(subtomoPathList)

        subtomoArray = np.zeros((numberOfSubtomos, 32, 32, 32), dtype=np.float64)

        for index, subtomo in enumerate(subtomoPathList):
            subtomoDataTmp = Image(subtomo).getData()

            subtomoArray[index, :, :, :] = subtomoDataTmp[:, :, :]

        std = subtomoArray.std()
        mean = subtomoArray.mean()

        subtomoArray = (subtomoArray - mean) / std

        firstPredictionArray = self.firstModel.predict(subtomoArray)

        overallPrediction, predictionAverage = self.determineOverallPrediction(firstPredictionArray, self.misalignmentCriteria)

        if not overallPrediction:
            overallPrediction = 1  # Strong misalignment

            # Set misalignment score to -1 if subtomos removed by the first network
            secondPredictionArray = np.full(firstPredictionArray.shape, -1)

        else:
            secondPredictionArray = self.secondModel.predict(subtomoArray)

            overallPrediction, predictionAverage = self.determineOverallPrediction(secondPredictionArray, self.misalignmentCriteria)

            if self.misaliThrBool:  # Using threshold

                if predictionAverage > self.misaliThr:
                    overallPrediction = 3  # Alignment
                else:
                    overallPrediction = 2  # Weak misalignment

        return overallPrediction, predictionAverage, firstPredictionArray, secondPredictionArray


    #  --------------------- UTILS FUNCTIONS -----------------------------
    def loadModels(self):
        self.firstModel = load_model(self.inputModel1)
        # print(self.firstModel.summary())

        self.secondModel = load_model(self.inputModel2)
        # print(self.secondModel.summary())

    def determineOverallPrediction(self, predictionList, overallCriteria):
        """
        This method return an overall prediction based on the different singular predictions for each gold bead. This
        can be estimated with a voting system (no considering the actual score value) or as the average of the obtained
        scores for each gold beads.
        :param predictionList: vector with the score values predicted for each gold bead
        :param overallCriteria: criteria to be used to calculate the overall prediction as the most voted option (0) or
        the average of all the scores (1)
        :return: bool indicating if the tomogram present misalignment or not
        :return average of the predicted scores
        """

        predictionAvg = np.average(predictionList)

        if overallCriteria == 0:
            predictionClasses = np.round(predictionList)

            overallPrediction = 0

            for prediction in predictionClasses:
                overallPrediction += prediction

            print("Subtomo analysis: " + str(overallPrediction) + " aligned vs " +
                  str(predictionList.size - overallPrediction) + "misaligned")

            overallPrediction = overallPrediction / predictionList.size

            # aligned (1) or misaligned (0)
            return (True if overallPrediction > 0.5 else False), predictionAvg

        elif overallCriteria == 1:
            print("prediction list:")
            print(predictionList)

            print("Subtomo analysis preditcion: " + str(predictionAvg))

            # aligned (1) or misaligned (0)
            return (True if predictionAvg > 0.5 else False), predictionAvg


    #  --------------------- RUN -----------------------------
    def run(self):
        # Read input params
        self.readInputParams()

        # Get subtomo path list from directory
        subtomoPathList = self.getSubtomoPathList(self.subtomoFilePath)

        # Make prediction from subtomos in list
        overallPrediction = None
        predictionAverage = None
        firstPredictionArray = None
        secondPredictionArray = None

        if len(subtomoPathList) != 0:
            totalNumberOfSubtomos = len(subtomoPathList)
            print("Total number of subtomos: " + str(totalNumberOfSubtomos))

            self.loadModels()

            overallPrediction, predictionAverage, firstPredictionArray, secondPredictionArray = \
                self.makePrediction(subtomoPathList)

            print("For coordinates in " + str(os.path.basename(self.subtomoFilePath)) + " obtained prediction from " +
                    str(len(subtomoPathList)) + " subtomos is " + str(overallPrediction))
            
        # Generate output
        self.generateOutputXmd(overallPrediction, predictionAverage, firstPredictionArray, secondPredictionArray)
            
    
if __name__ == "__main__":
    exitCode = ScriptDeepMisalignmentDetection().tryRun()
    sys.exit(exitCode)
