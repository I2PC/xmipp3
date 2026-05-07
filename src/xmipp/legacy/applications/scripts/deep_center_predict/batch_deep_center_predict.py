#!/usr/bin/env python3

import glob
import math
import numpy as np
from numpy.linalg import norm
import os
import sys
import xmippLib
from xmipp_base import *

class ScriptDeepCenterPredict(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Predict center using a deep center model')

        ## params
        self.addParamsLine(' -i <metadata>                : xmd file with the list of experimental images')
        self.addParamsLine(' --model <model>              : .tf file with the centering model')
        self.addParamsLine(' -o <metadata>                : Output filename')
        self.addParamsLine('[--scale <K=1>]               : Multiply the shifts by this factor')
        self.addParamsLine('[--gpu <id=0>]                : GPU Id')

    def run(self):
        fnExp = self.getParam("-i")
        fnModel = self.getParam("--model")
        fnOut = self.getParam("-o")
        gpuId = self.getParam("--gpu")
        K = float(self.getParam("--scale"))
        maxSize = 32

        from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed
        checkIf_tf_keras_installed()

        if not gpuId.startswith('-1'):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

        import keras

        def produce_output(fnExp, shifts, itemIds, fnOut):
            Ydict = {itemId: index for index, itemId in enumerate(itemIds)}
            mdExp = xmippLib.MetaData(fnExp)
            for objId in mdExp:
                itemId = mdExp.getValue(xmippLib.MDL_ITEM_ID, objId)
                x, y = shifts[Ydict[itemId]]
                mdExp.setValue(xmippLib.MDL_SHIFT_X, x, objId)
                mdExp.setValue(xmippLib.MDL_SHIFT_Y, y, objId)
                mdExp.setValue(xmippLib.MDL_ANGLE_ROT, 0.0, objId)
                mdExp.setValue(xmippLib.MDL_ANGLE_TILT, 0.0, objId)
                mdExp.setValue(xmippLib.MDL_ANGLE_PSI, 0.0, objId)
            mdExp.write(fnOut)

        md = xmippLib.MetaData(fnExp)
        Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnExp)
        fnImgs = md.getColumnValues(xmippLib.MDL_IMAGE)
        itemIds = md.getColumnValues(xmippLib.MDL_ITEM_ID)

        numImgs = len(fnImgs)
        numBatches = numImgs // maxSize
        if numImgs % maxSize > 0:
            numBatches = numBatches + 1

        shiftModel = keras.models.load_model(fnModel)

        k = 0
        shifts = np.zeros((numImgs, 2))
        for i in range(numBatches):
            numPredictions = min(maxSize, numImgs - i * maxSize)
            Xexp = np.zeros((numPredictions, Xdim, Xdim, 1), dtype=np.float64)
            for j in range(numPredictions):
                Iexp = np.reshape(xmippLib.Image(fnImgs[k]).getData(), (Xdim, Xdim, 1))
                Xexp[j,] = (Iexp - np.mean(Iexp)) / np.std(Iexp)
                k += 1
            shifts[i * maxSize:(i * maxSize + numPredictions), :] = K*shiftModel.predict(Xexp)

        produce_output(fnExp, shifts, itemIds, fnOut)

if __name__ == '__main__':
    exitCode = ScriptDeepCenterPredict().tryRun()
    sys.exit(exitCode)
