#!/usr/bin/env python3
""""
**************************************************************************
*
* Authors:  Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
*           Ruben Sanchez (rsanchez@cnb.csic.es)
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
**************************************************************************
"""
import math
import os
import itertools
import numpy as np
import pywt
import pywt.data

from scipy.ndimage import zoom
from xmipp_base import XmippScript
import xmippLib


class ScriptVolumeConsensus(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Volume consensus')
        ## params
        self.addParamsLine('-i <inputFile>   : A .txt file that contains the path of input volumes')
        self.addParamsLine('-o  <outputFile>   : Consensus volume filename')
        ## examples
        self.addExampleLine('xmipp_volume_consensus -i1 path/to/inputs/file.txt -o path/to/outFile.mrc')

    def run(self):
        inputFile = self.getParam('-i')
        outVolFn = self.getParam('-o')
        self.computeVolumeConsensus(inputFile, outVolFn)

    def resize(self, image, dim):
        imageFt = np.fft.rfftn(image)
        resultFt = np.zeros(dim[:-1] + (dim[-1]//2+1,), dtype=imageFt.dtype)

        copyExtent = np.minimum(image.shape, dim) // 2
        srcCornerStart = image.shape-copyExtent
        dstCornerStart = dim-copyExtent
        for corners in itertools.product(range(2), repeat=len(dim)-1):
            corners = np.array(corners + (0, ))
            srcStart = np.where(corners, srcCornerStart, 0)
            srcEnd = srcStart + copyExtent
            dstStart = np.where(corners, dstCornerStart, 0)
            dstEnd = dstStart + copyExtent
            srcSlices = [slice(s, e) for s, e in zip(srcStart, srcEnd)]
            dstSlices = [slice(s, e) for s, e in zip(dstStart, dstEnd)]
            resultFt[tuple(dstSlices)] = imageFt[tuple(srcSlices)]
            
        return np.fft.irfftn(resultFt)
        
    def computeVolumeConsensus(self, inputFile, outVolFn, wavelet='sym11'):
        outputWt = None
        outputMin = None
        xdim2 = None
        xdimOrig = None
        image = xmippLib.Image()
        with open(inputFile) as f:
            for line in f:
                fileName = line.split()[0]
                if fileName.endswith('.mrc'):
                    fileName += ':mrc'
                    
                image.read(fileName)
                volume = image.getData()
                
                if xdimOrig is None:
                    xdimOrig = volume.shape[0]
                    xdim2 = 2**(math.ceil(math.log2(xdimOrig))) # Next power of 2
                    ydimOrig = volume.shape[1]
                    ydim2 = 2**(math.ceil(math.log2(ydimOrig)))  # Next power of 2
                    zdimOrig = volume.shape[2]
                    zdim2 = 2**(math.ceil(math.log2(zdimOrig)))  # Next power of 2
                    
                if xdimOrig!=xdim2 or ydimOrig!=ydim2 or zdimOrig!=zdim2:
                    #volume = zoom(volume, (xdim2/xdimOrig,ydim2/ydimOrig,zdim2/zdimOrig))
                    volume = self.resize(volume, (zdim2, ydim2, xdim2))
                
                nlevel = pywt.dwtn_max_level(volume.shape, wavelet=wavelet)
                wt = pywt.wavedecn(
                    data=volume, 
                    wavelet=wavelet, 
                    level=nlevel
                )
                
                if outputWt == None:
                    outputWt = wt
                    outputMin = np.zeros_like(volume)
                else:
                    diff = np.abs(np.abs(wt[0]) - np.abs(outputWt[0]))
                    diff = self.resize(diff, outputMin.shape)
                    np.maximum(
                        diff, outputMin,
                        out=outputMin
                    )
                    outputWt[0] = np.where(
                        np.abs(wt[0]) > np.abs(outputWt[0]),
                        wt[0], outputWt[0]
                    )
                            
                    for level in range(1, nlevel+1):
                        wtLevel = wt[level]
                        outputWtLevel = outputWt[level]
                        for detail in wtLevel:
                            wtLevelDetail = wtLevel[detail]
                            outputWtLevelDetail = outputWtLevel[detail]
                            
                            diff = np.abs(np.abs(wtLevelDetail) - np.abs(outputWtLevelDetail))
                            diff = self.resize(diff, outputMin.shape)
                            np.maximum(
                                diff, outputMin,
                                out=outputMin
                            )
     
                            outputWtLevelDetail[...] = np.where(
                                np.abs(wtLevelDetail) > np.abs(outputWtLevelDetail),
                                wtLevelDetail, outputWtLevelDetail
                            )
                            

            f.close()
        consensus = pywt.waverecn(outputWt, wavelet)
        if xdimOrig!=xdim2 or ydimOrig!=ydim2 or zdimOrig!=zdim2:
            consensus = self.resize(consensus, (zdimOrig, ydimOrig, xdimOrig))
        image.setData(consensus)
        image.write(outVolFn)
        if xdimOrig!=xdim2 or ydimOrig!=ydim2 or zdimOrig!=zdim2:
            outputMin = self.resize(outputMin, (zdimOrig, ydimOrig, xdimOrig))
        image.setData(outputMin)
        outVolFn2 = os.path.splitext(outVolFn)[0] + '_diff.mrc'
        image.write(outVolFn2)


if __name__=="__main__":
    ScriptVolumeConsensus().tryRun()
