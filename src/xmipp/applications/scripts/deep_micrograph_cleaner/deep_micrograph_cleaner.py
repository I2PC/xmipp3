#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Ruben Sanchez Garcia rsanchez@cnb.csic.es
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

import os
import sys

from xmipp_base import XmippScript
import xmippLib


class ScriptMicrographCleanerEm(XmippScript):
    _conda_env="xmipp_MicCleaner"
    def __init__(self):

        XmippScript.__init__(self)

    def getDoubleParamWithDefault(self, paramName, conditionFun= lambda x: False, defaultVal=None):
      if self.checkParam(paramName):
        x = self.getDoubleParam(paramName)
        if conditionFun(x):
          return defaultVal
        return x
      else:
        return defaultVal

    def defineParams(self):
        self.addUsageLine('Compute goodness score for picked coordinates. Rule out bad coordinates')
        ## params
        self.addParamsLine(' -i <inputMicsMetada>               : metadata containing the micrograph(s) '+
                           'filenames where coordinates were picked (.mrc or .tif).\n')
                           
        self.addParamsLine(' [ -c <inputCoordsDir>  ] : input coordinates directory (.pos or tab separated x y). Filenames '+
                           'must agree with input micrographs except for file extension.')

        self.addParamsLine(' [ -o <outputCoordsDir> ] : output coordinates directory.')
        self.addParamsLine(' [ -d <deepLearningModel> ]  : (optional) deep learning model filename. If not provided, default model will be used')
        self.addParamsLine('-b <boxSize>     : particles box size in pixels')
        self.addParamsLine(' [ -s <downFactor>   <F=1.0> ]   : (optional) micrograph downsampling factor to scale coordinates, Default no scaling')
        self.addParamsLine(' [ --deepThr <deepThr> ]: (optional) deep learning threshold to rule out a coordinate. The smaller the treshold '+
                           'the more coordiantes will be rule out. Ranges 0..1. Recommended 0.8')
        self.addParamsLine(' [--sizeThr <sizeThr> <F=0.8> ]: Failure threshold. Fraction of the micrograph predicted as contamination to ignore predictions. '+
                           '. Ranges 0..1. Default 0.8')
        self.addParamsLine('[ --predictedMaskDir <predictedMaskDir> ] : directory to store the predicted masks. If a given mask already existed, it will be used instead'+
                           ' of a new prediction')
        self.addParamsLine('[ -g <gpuIds>   <N=0> ] : GPU ids to employ. Comma separated list. E.g. "0,1". Default 0. '
                            'use -1 for CPU-only computation or "all" to use all devices found in CUDA_VISIBLE_DEVICES (option for slurm)')
        
        ## examples
        self.addExampleLine('xmipp_deep_micrograph_cleaner -c path/to/inputCoords/ -o path/to/outputCoords -b $BOX_SIXE  -i  /path/to/micrographs/')
        
    def run(self):
        # checkIf_tf_keras_installed()
        args={}
        gpusToUse="0"
        if self.checkParam('-g'):
          gpusToUse= self.getParam('-g')
          if "None" in gpusToUse or "-1" in gpusToUse:
            gpusToUse=-1
        args["gpus"]=gpusToUse

        if self.checkParam('-i'):
          mdObj= xmippLib.MetaData(os.path.expanduser( self.getParam('-i')))

          args["inputMicsPath"]= []
          for objId in mdObj:
            args["inputMicsPath"]+= [mdObj.getValue(xmippLib.MDL_IMAGE, objId)]
          args["inputMicsPath"]= " ".join(args["inputMicsPath"])
        else:
          raise Exception("Error, input micrographs fnames are requried as argument")

        if self.checkParam('-b'):
          args["boxSize"]= self.getIntParam('-b')
        else:
          raise Exception("Error, box size in pixels is required as argument")

        args["inputCoordsDir"], args["outputCoordsDir"], args["predictedMaskDir"]= None, None, None
        if self.checkParam('-c'):
          args["inputCoordsDir"]= os.path.expanduser( self.getParam('-c'))
        if self.checkParam('-o'):
          args["outputCoordsDir"]= os.path.expanduser( self.getParam('-o'))
        if self.checkParam('--predictedMaskDir'):
          args["predictedMaskDir"]= os.path.expanduser( self.getParam('--predictedMaskDir'))


        args["downFactorCoords"]= self.getDoubleParamWithDefault('-s', defaultVal=1.0)
        args["deepThr"]= self.getDoubleParamWithDefault('--deepThr', conditionFun= lambda x: x <= 0 or x >= 1,
                                                        defaultVal=None)

        args["sizeThr"]= self.getDoubleParamWithDefault('--sizeThr', conditionFun= lambda x: x <= 0 or x >= 1,
                                                        defaultVal=0.8)
          
        if args["inputCoordsDir"] is None and args["predictedMaskDir"] is None:
          raise Exception("Either inputCoordsDir or predictedMaskDir (or both) must be provided")

        if args["inputCoordsDir"] is not None and args["outputCoordsDir"] is None:
          raise Exception("Error, if inputCoordsDir provided, then outputCoordsDir must also be provided")

        if args["outputCoordsDir"] is not None and args["inputCoordsDir"] is None:
          raise Exception("Error, if outputCoordsDir provided, then inputCoordsDir must also be provided")
    

        if self.checkParam('-d'):
          args["deepLearningModel"]= self.getParam('-d')
        else:
          args["deepLearningModel"]= self.getModel('deepMicrographCleaner', 'defaultModel.keras')

        cmdArgs= " ".join(["--"+str(key)+" "+str(args[key]) for key in args if args[key] is not None ])
        self.runCondaCmd("cleanMics", cmdArgs)

if __name__ == '__main__':
    '''
scipion python `which xmipp_deep_micrograph_cleaner` -g 0 
    '''
    exitCode=ScriptMicrographCleanerEm().tryRun()
    sys.exit(exitCode)
    
