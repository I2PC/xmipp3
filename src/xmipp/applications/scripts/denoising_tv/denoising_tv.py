#!/usr/bin/env python3
# *****************************************************************************
# *
# * Authors:     J.L. Vilas (jlvilas@cnb.csic.es) [1]
# *
# * [1] Centro Nacional de Biotecnologia, CSIC, Spain
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 3 of the License, or
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
# *****************************************************************************

import sys
import os
import mrcfile

import numpy as np

import scipy as sp
import tigre

import xmippLib
from xmipp_base import XmippScript


class DenoisingTV(XmippScript):
    _conda_env="xtomo_tigre" 

    def __init__(self):

        XmippScript.__init__(self)
                
        # Global parameters
        self.xdim = None
        self.ydim = None
        

    #  --------------------- DEFINE PARAMS -----------------------------
    def defineParams(self):
        # Description
        self.addUsageLine('The program will make denoising using total variation\n')
        
        # Params
        self.addParamsLine(' -i <fnTomo>                          : Volume or tomogram to be denoised')
        self.addParamsLine(' -o <fnOut>                           : Output filename of the volume/tomogram.')
        self.addParamsLine(' [--iters <iterations>]               : Number of iterations for the reconstructoin algorithm. ')
        self.addParamsLine(' [--lambda <lmbda>]                   : Hyperparameter. The update will be multiplied by this number every iteration, to make the steps bigger or smaller. Default: lmbda=1.0 for all algorithms except for irn-tv-cgls for which lmbda=5.0')

        self.addParamsLine(' [--gpu <gpuId>]                      : GPU Ids to be use in the image processing. (by default gpu 0) If this parameter is not set, the gpu 0 will be used') 
        
        # Examples       
        self.addExampleLine('xmipp_tomogram_reconstruction --tiltseries ts.mrc --angles angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method wbp --filter ram_lak')
        self.addExampleLine('xmipp_tomogram_reconstruction --tiltseries ts.mrc --angles angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method fdk --filter hamming')


    def readInputParams(self):
        '''
        In this function the parameters are read. For for information about their use see the help
        '''

        self.fnTomo        = self.getParam('-i')
        self.fnOut         = self.getParam('-o')
        self.gpuId         = self.getIntParam('--gpu') if self.checkParam('--gpu') else 0
        
        self.iterations    = self.getIntParam('--iters') if self.checkParam('--iters') else 50
        self.lmbda         = self.getDoubleParam('--lambda') if self.checkParam('--lambda') else 15.0

    def run(self):
        print('Starting ...')
        
        self.readInputParams()
        self.tigreDenoisingTV()

    def getGPUs(self):
        return str(self.gpuId)

    
    def tigreDenoisingTV(self):
        from tigre.utilities import gpu, im_3d_denoise

        gpuids = gpu.GpuIds()
        gpuids.devices = [int(self.getGPUs())]

        tomo = mrcfile.read(self.fnTomo)

        denoisedTomo = im_3d_denoise.im3ddenoise(np.array(tomo, dtype=np.float32), iter=self.iterations, lmbda=self.lmbda, gpuids=None)

        with mrcfile.new(self.fnOut, overwrite=True) as mrc:
            mrc.set_data(denoisedTomo)

    '''
    def readVolTomo(self):
        
        #Get tilt series
        tomo = mrcfile.read(self.fnTomo)
        
        #Define volume
        dims = np.shape(ts_aux)
        self.xdim = dims[2]
        self.ydim = dims[1]
        self.nimages = dims[0]
        
        stdTi = []
        meanTi = []
        if self.normalizeTi == 'standard':
            for i in range(0, self.nimages):
                ti = ts_aux[i,:,:]
                stdTi = np.std(ti)
                meanTi = np.mean(ti)
                ti = (ti-meanTi)/stdTi
                ts_aux[i,:,:] = ti

        return ts_aux.astype(np.float32)
    '''

if __name__ == '__main__':
    '''
    scipion xmipp_denoising_tv -g 0
    '''
    exitCode=DenoisingTV().tryRun()
    sys.exit(exitCode)
