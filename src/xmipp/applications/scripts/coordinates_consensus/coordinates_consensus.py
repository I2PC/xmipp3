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
from subprocess import check_call
from math import sqrt

import numpy as np
import xmipp_base as xmipp_base
from enum import Enum

from xmippPyModules.coordinatesTools.coordinatesTools import (readPosCoordsFromFName,
                                                        writeCoordsListToPosFname)

class ConsensusType(Enum):

    def __init__(self, id, message):
        self.id = id
        self.message = message

    AND = -1, 'by_all'
    OR = 1, 'by_at_least_one'
    UNION_INTERSECTIONS = 0, 'by_at_least_two'

class ScriptCordsConsensus(xmipp_base.XmippScript):
    def __init__(self):
      xmipp_base.XmippScript.__init__(self)
        
    def defineParams(self):
      self.addUsageLine('Preprocess all mics in directory')
      ## params
      
      self.addParamsLine('-i <inputFile>   : A file that contains the path to all coordinates files'
                         ' (pos files). In each row, many cordinates files can be provided. Same'
                         ' sampling rate in all files required')

      self.addParamsLine('-s <particleSize>       : particle size in pixels')
      
      self.addParamsLine('-c <consensus>          : How many times need a particle to be selected to be considered'
                         ' as a consensus particle. Set to "by_all" to indicate that it needs to be selected by all'
                         ' algorithms. Set to "by_at_least_two" to indicate that it needs to be selected by at least two algorithms.'
                         ' Set to "by_at_least_one" to indicate that it suffices that only 1 algorithm selects the particle')
      
      self.addParamsLine('-d <diameterTolerance> <F=0.1>  : Distance between 2 coordinates'
                         ' to be considered the same, measured as fraction of particleSize')

      self.addParamsLine('-o <pathToExtractedParticles>  : A path to the directory where consensus coordiantes '
                         ' will be saved')
                                                
      self.addParamsLine('[ -t <numThreads>  <N=1>  ]   : Number of threads')

      ## examples
      self.addExampleLine('  coordinates_consensus -i path/to/inputs/file.txt -d 0.05 -s 128 -t 2 -o path/to/outDir')
      self.addExampleLine('  path/to/inputs/file.txt:\n'
                         '#micName micId pos1 pos2\n'
                         'Runs/import/tmp/010_movie_aligned.mrc 010 Runs/picker1/tmp/010_movie_aligned.pos Runs/picker2/tmp/010_movie_aligned.pos\n'
                         'Runs/import/tmp/110_movie_aligned.mrc 110 Runs/picker1/tmp/110_movie_aligned.pos Runs/picker2/tmp/110_movie_aligned.pos\n'
                         )


    def run(self):
    
      numberOfThreads= self.getIntParam('-t')
      inputFile= self.getParam('-i')
      boxSize= self.getDoubleParam('-s')
      consensusRadius= self.getDoubleParam('-d')
      consensusCriterium = self.getParam('-c')
      outDir= self.getParam('-o')

      argsList=[]
      with open(inputFile) as f:
        for line in f:
          if line.startswith("#"):
            continue
          else:
            lineArray= line.split()
            if len(lineArray)<1:
              continue
            else:
              argsList+=[  ( lineArray, boxSize, consensusRadius, consensusCriterium, outDir)]
              
      Parallel(n_jobs= numberOfThreads, backend="multiprocessing", verbose=1)(
                  delayed(consensusCoordsOneMic)(*arg) for arg in argsList)

   
def consensusCoordsOneMic(coords_files, boxSize, consensusRadius, consensusCriterium, outDir):
  """ Compute consensus of coordiantes for the same micrograph
      @param  coords_files [mic1_p1.pos, mic2_p2.pos,...]
  """
  baseName = os.path.splitext(os.path.basename(coords_files[0]))[0]
  out_name= os.path.join(outDir, baseName+".pos")
  if os.path.isfile(out_name): return
  coords=[]
  Ncoords=0
  n=0
  micIds=set([])
  for fname in coords_files:
    x_y_list, micId= readPosCoordsFromFName(fname, returnAlsoMicId=True)
    if micId is None:
      continue
    micIds.add(micId)
    x_y_array= np.asarray(x_y_list, dtype=int)
    coords.append(x_y_array)
    Ncoords += x_y_array.shape[0]
    n += 1
  if len(micIds)>1:
    raise ValueError("Error, inconsistency in micIds, all files should belong to the same micId: %s"%(str(coords_files)))
  if len(coords)==0:
    return
  allCoords = np.zeros([Ncoords, 2])
  votes = np.zeros(Ncoords)

  #Cast Consensus
  if consensusCriterium == ConsensusType.AND.message:
      consensusCriterium = ConsensusType.AND.id
  elif consensusCriterium == ConsensusType.UNION_INTERSECTIONS.message:
      consensusCriterium = ConsensusType.UNION_INTERSECTIONS.id
  elif consensusCriterium == ConsensusType.OR.message:
      consensusCriterium = ConsensusType.OR.id

  # Add all coordinates of the <first set of coordinates
  N0 = coords[0].shape[0]
  inAllMicrographs = consensusCriterium < 0 or consensusCriterium == len(coords_files)
  if N0 == 0 and inAllMicrographs:
    return
  elif N0 > 0:
    allCoords[0:N0, :] = coords[0]
    votes[0:N0] = 1
  
  consensusNpixels = consensusRadius* boxSize

  # Add the rest of coordinates
  Ncurrent = N0
  for n in range(1, len(coords)):
    for coord in coords[n]:
      if Ncurrent > 0:
        dist = np.sum((coord - allCoords[0:Ncurrent])**2, axis=1)
        imin = np.argmin(dist)
        if sqrt(dist[imin]) < consensusNpixels:
          newCoord = (votes[imin]*allCoords[imin,]+coord)/(votes[imin]+1)
          allCoords[imin,] = newCoord
          votes[imin] += 1
        else:
          allCoords[Ncurrent, :] = coord
          votes[Ncurrent] = 1
          Ncurrent += 1
      else:
        allCoords[Ncurrent, :] = coord
        votes[Ncurrent] = 1
        Ncurrent += 1

  # Select those in the consensus
  if consensusCriterium < 0:
      consensusCriterium = len(coords_files)
  elif consensusCriterium == 0:
      consensusCriterium = 2

  consensusCoords = allCoords[votes >= consensusCriterium, :]
  # Write the consensus file only if there are some coordinates (size > 0)
  if consensusCoords.size>0:
      writeCoordsListToPosFname(out_name, consensusCoords, outputRoot=outDir, micId= micId)
            
if __name__ == '__main__':

    exitCode=ScriptCordsConsensus().tryRun()
    sys.exit(exitCode)
    
