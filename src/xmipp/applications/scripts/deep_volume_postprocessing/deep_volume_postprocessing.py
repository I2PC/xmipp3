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
    _conda_env = 'xmipp_deepEMhancer'

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
        self.addUsageLine('DeepEMhancer. Apply a CCN to post-process an EM volume to obtain a masked and sharpened-like volume in an automatic fashion\n.'
                          'Normalization of the input volume is key, so unmasked volumes should be provided as input. There are 3 normalization options: \n'
                          '1) Automatic (default)\n'
                          '2) Providing the statistics of the noise'
                          '3) Using a binary mask')
        ## params
        self.addParamsLine(' -i <inputVol>        : input volume to postprocess (or half map 1). Only mrc format allowed ')
        self.addParamsLine(' [-i2 <inputVol>]     : input half map 2. Only mrc format allowed ')

        self.addParamsLine(' -o <outputVol>       : output fname to save postprocessed volume. Only mrc format allowed ')

        self.addParamsLine(' [ --sampling_rate <AperVoxel>  ] :  (optional) The sampling rate of the volume. If not provided, it will be read from  -i header')

        self.addParamsLine(' [ --checkpoint <deepLearningModel> ]  : (optional) deep learning model filename. If not provided, default model will be used')

        self.addParamsLine(' [ --cleaningStrengh <F=0.1> ]  : (optional)  Max size of connected componemts to remove 0<s<1 or -1 to deactivate. Default: 0.1"')


        self.addParamsLine(' [--sizeThr <sizeThr> <F=0.8> ]: Failure threshold. Fraction of the micrograph predicted as contamination to ignore predictions. '+
                             '. Ranges 0..1. Default 0.8')

        self.addParamsLine('[ -g <gpuId>   <N=0> ] : GPU id to employ. Default 0. use -1 for CPU-only computation or "all" to use all devices found in '
                           'CUDA_VISIBLE_DEVICES (option for slurm)')
        self.addParamsLine('[ -b <batchSize>   <N=6> ] : Number of cubes to process simultaneously. Lower it if CUDA Out Of Memory error happens and increase it if low GPU performance observed')

        self.addParamsLine(' [--binaryMask <binMask>]        : Normalization-> Binary mask volume to compute stats for normalization. Only mrc format allowed ')
        self.addParamsLine(' [--noise_stats_mean <mean> ]   : Normalization-> Noise stats mean  for normalization ')
        self.addParamsLine(' [--noise_stats_std <std> ]   : Normalization-> Noise stats standard deviation for normalization ')


        ## examples
        self.addExampleLine('xmipp_deep_volume_postprocessing -i path/to/inputVol.mrc -o path/to/outputVol.mrc ')
        
    def run(self):


        params= " -i %s " % self.getParam('-i')
        if self.checkParam('-i2'):
          params += " -i2 %s " % self.getParam('-i2')

        params += " -o %s " % self.getParam('-o')

        if self.checkParam('--checkpoint'):
          params += " --deepLearningModelPath %s "%os.path.expanduser(self.getParam("--checkpoint"))
        else:
          params += " --deepLearningModelPath  %s "%XmippScript.getModel("deepEMhancer_v016", "production_checkpoints/deepEMhancer_tightTarget.hd5")

        if self.checkParam('--sampling_rate'):
          params += " --samplingRate %f " %  self.getDoubleParam('--sampling_rate')

        if self.checkParam('--binaryMask'):
          params += " --binaryMask %s " % (os.path.abspath(self.getParam('--binaryMask')))

        elif self.checkParam('--noise_stats_mean'):
          params += " --noiseStats %f %f " % (self.getDoubleParam('--noise_stats_mean'), self.getDoubleParam('--noise_stats_std'))

        if self.checkParam('--cleaningStrengh'):
          params += " --cleaningStrengh %f " % self.getDoubleParamWithDefault('--cleaningStrengh', defaultVal=-1)

        params+= "-g %s "%self.getParam("-g")
        params+= "-b %s "%self.getParam("-b")

        cmd= "deepemhancer"
        print( cmd+" "+params)
        self.runCondaCmd(cmd, params)


if __name__ == '__main__':
    '''
scipion xmipp_deep_volume_postprocessing -g 0
    '''
    exitCode=ScriptMicrographCleanerEm().tryRun()
    sys.exit(exitCode)
    
