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

from xmipp_base import *

class ScriptExtractParticles(XmippScript):
    def __init__(self):
      XmippScript.__init__(self)
        
    def defineParams(self):
      self.addUsageLine('Preprocess all mics in directory')
      ## params
      
      self.addParamsLine('-i <inputFile>   : A file that contains the path of input micrograph and associated '
                         ' coordinates (pos files)')

      self.addParamsLine('-s <particleSize>            : particle size in pixels')
      
      self.addParamsLine('-d <donwsampleFactor> <D=1>  : Downsamplig factor')

      self.addParamsLine('-o <pathToExtractedParticles>  : A path to the directory where preprocessed extracted '
                         'particles will be saved')
                                                
      self.addParamsLine('[ -t <numThreads>  <N=1>  ]   : Number of threads')

      ## examples
      self.addExampleLine('  xmipp_extract_particles -i path/to/inputs/file.txt -d 4 -s 128 -t 2 -o path/to/outDir')
      self.addExampleLine('  path/to/inputs/file.txt:\n'
                         '#mic coords\n'
                         'Runs/004986_XmippProtScreenDeepConsensus/extra/preProcMics/010_movie_aligned.mrc Runs/004986_XmippProtScreenDeepConsensus/tmp/010_movie_aligned.pos\n'
                         'Runs/004986_XmippProtScreenDeepConsensus/extra/preProcMics/100_movie_aligned.mrc Runs/004986_XmippProtScreenDeepConsensus/tmp/100_movie_aligned.pos\n'
                         'Runs/004986_XmippProtScreenDeepConsensus/extra/preProcMics/107_movie_aligned.mrc Runs/004986_XmippProtScreenDeepConsensus/tmp/107_movie_aligned.pos\n'
                         )


    def run(self):
    
      numberOfThreads= self.getIntParam('-t')
      inputFile= self.getParam('-i')
      boxSize= self.getDoubleParam('-s')
      downFactor= self.getDoubleParam('-d')
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
            elif len(lineArray)!=2:
              raise ValueError("Error, input file bad format. If -c option, it must have 2 cols: 'micFname coordsFname'")
            else:
              mic_fname= lineArray[0]
              mic_basename_split=  os.path.basename(mic_fname).split(".")
              mic_basename= ".".join(mic_basename_split[:-1]) if len(mic_basename_split)>1 else mic_basename_split[0]
              out_name= os.path.join(outDir, mic_basename)
              pos_fname= lineArray[1]
              if not pos_fname.startswith("particles@"):
                pos_fname="particles@"+pos_fname
              argsList+=[  (mic_fname, pos_fname, boxSize, out_name, downFactor)]
              
      Parallel(n_jobs= numberOfThreads, backend="multiprocessing", verbose=1)(
                  delayed(extractPartsOneMic)(*arg) for arg in argsList)

   
def extractPartsOneMic(mic_fname, pos_fname, boxSize, out_name, downFactor=1):
  """ Extract particles from one micrograph
  """
  if os.path.isfile(out_name): return
 
  cmd = "xmipp_micrograph_scissor -i %s --pos %s -o %s --Xdim %d --downsampling %f --fillBorders"%(mic_fname, 
                                                              pos_fname, out_name, boxSize, downFactor)
  print(cmd)
  check_call(cmd, shell=True)

if __name__ == '__main__':

    exitCode=ScriptExtractParticles().tryRun()
    sys.exit(exitCode)
    
