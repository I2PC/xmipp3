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

import sys, os
from joblib import delayed, Parallel
from scipy.spatial.distance import cdist

import numpy as np
from xmipp_base import *
import xmippLib
from xmipp3.convert import readPosCoordinates
from pwem.objects import Coordinate, Micrograph
from xmippPyModules.coordinatesTools.coordinatesTools import (readPosCoordsFromFName,
                                                                            writeCoordsListToPosFname)

class ScriptPickNoise(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)
        
    def defineParams(self):
        self.addUsageLine('Picks random coordinates from one micrograph')
        ## params
        
        self.addParamsLine('-i <pathToMics>         : A path to the directory of micrographs'
                           ' path/\n  mic1.mrc\n  mic2.mrc')
                           
        self.addParamsLine('-c <pathToAlreadyPicked>  : A path to the directory of coordinates, in pos format. e.g. ' 
                           ' path/\n  coor1.pos\n  coor2.pos\nThis coordinates are used to prevent random picking '
                           'from selecting coordinates close to true particles')

        self.addParamsLine('-o <pathToNewCoords>  : A path to the directory where random coordinates will be saved')
                                                  
        self.addParamsLine('-s <boxSize>              : Particle box size, in pixels')
        self.addParamsLine('[-n <numberPicksPerMic> <F=-1> ]    : Number of coordiantes to pick. If -1, pick as many '
                           ' as particles contained in -c ')
        self.addParamsLine('[ -t <numThreads>  <N=1>  ]   : Number of threads')

        ## examples
        self.addExampleLine('  xmipp_pick_noise -i path/to/mics -c path/to/coords -o path/to/noiseCoords '
                            ' -s 128 -t 2')


    def run(self):
    
        numberOfThreads= self.getIntParam('-t')
        micrographsPath= self.getParam('-i')
        coordsPath= self.getParam('-c')
        boxSize= self.getIntParam('-s')
        numberToPick= self.getIntParam('-n')
        outDir= self.getParam('-o')
        
        micsBaseToFullName= { }
        for micName in os.listdir(micrographsPath):
          if micName.endswith(".mrc") or micName.endswith(".tif"):
            baseName= os.path.basename( micName).split(".")[0]
            micsBaseToFullName[baseName]= os.path.join(micrographsPath, micName)

        I= xmippLib.Image()
        I.read(micsBaseToFullName[baseName])
        micShape= I.getData().shape

        argsList=[]
        for posName in os.listdir(coordsPath):
          if posName.endswith(".pos"):
            baseName= os.path.basename( posName).split(".")[0]
            micName= micsBaseToFullName[baseName]
            posName= os.path.join( coordsPath, posName)
            argsList.append( (baseName, micName, posName, micShape, outDir, numberToPick, boxSize))

        Parallel(n_jobs= numberOfThreads, backend="multiprocessing", verbose=1)(
                    delayed(pickNoiseOneMic)(*arg) for arg in argsList)

   
def pickNoiseOneMic(baseName, mic_fname, posName, mic_shape, outputRoot,
                  extractNoiseNumber, boxSize):

  """ Pick noise from one micrograph 
  """
  
  print("pick noise one mic %s %s %s %s %s %s %s"%(baseName, mic_fname, posName, mic_shape, 
                                                   outputRoot, extractNoiseNumber, boxSize) )

  coords_in_mic_list= readPosCoordsFromFName(posName)
  
  extractNoiseNumber= extractNoiseNumber if extractNoiseNumber!=-1 else  len(coords_in_mic_list)

  min_required_distance= boxSize//2
  currentCoords= np.array(coords_in_mic_list)
  if currentCoords.shape[0]>0:
    good_new_coords= []
    n_good_coords= 0
    for iterNum in range(9999):
      randomCoordsX= np.random.randint( boxSize//2, mic_shape[0]- boxSize//2 , size=extractNoiseNumber*2 )
      randomCoordsY= np.random.randint( boxSize//2, mic_shape[1]- boxSize//2 , size=extractNoiseNumber*2 )
      randomCoords= np.stack( [randomCoordsX, randomCoordsY], axis=1)
      del randomCoordsX, randomCoordsY
      dist_mat= cdist(randomCoords, currentCoords) #dist_mat: i:random_particles   j: currentCoords
      dist_mat= np.min(dist_mat, axis=1)
      g_n_c= randomCoords[ dist_mat>= min_required_distance ]
      n_good_coords+= g_n_c.shape[0]
      good_new_coords.append(g_n_c )
      if n_good_coords>= extractNoiseNumber:
        break
    good_new_coords= np.concatenate(good_new_coords)[:extractNoiseNumber]
    writeCoordsListToPosFname(mic_fname, good_new_coords, outputRoot)
  
if __name__ == '__main__':

    exitCode=ScriptPickNoise().tryRun()
    sys.exit(exitCode)
