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

from subprocess import check_call

from joblib import delayed, Parallel
import numpy as np
from xmipp_base import *

class ScriptPreproMics(XmippScript):
    def __init__(self):
      XmippScript.__init__(self)
        
    def defineParams(self):
      self.addUsageLine('Preprocess all mics in directory')
      ## params
      
      self.addParamsLine('-i <inputFile>         : A file that contains the path of input micrograph and possibly CTFs')

      self.addParamsLine('-s <samplingRate>            : sampling rate of the micrographs Angstroms/pixel')
      
      self.addParamsLine('-d <donwsampleFactor> <D=1>  : Downsamplig factor')

      self.addParamsLine('-o <pathToProcessesMics>  : A path to the directory where preprocessed micrograph will be saved')
      
      self.addParamsLine('[--invert_contrast ] : Invert micrograph contrast')
      
      self.addParamsLine('[ --phase_flip ] : Apply phase_flipping micrograph contrast')
                                                
      self.addParamsLine('[ -t <numThreads>  <N=1>  ]   : Number of threads')

      ## examples
      self.addExampleLine('  xmipp_preprocess_mics -i path/to/inputs/file.txt -s 1.6 -d 4 -t 2 -o path/to/outDir')
      self.addExampleLine('  path/to/inputs/file.txt:\n'
                         '#mic ctfparams\n'
                         'Runs/004986_XmippProtScreenDeepConsensus/extra/preProcMics/010_movie_aligned.mrc Runs/004986_XmippProtScreenDeepConsensus/tmp/010_movie_aligned.mrc.ctfParam\n'
                         'Runs/004986_XmippProtScreenDeepConsensus/extra/preProcMics/100_movie_aligned.mrc Runs/004986_XmippProtScreenDeepConsensus/tmp/100_movie_aligned.mrc.ctfParam\n'
                         'Runs/004986_XmippProtScreenDeepConsensus/extra/preProcMics/107_movie_aligned.mrc Runs/004986_XmippProtScreenDeepConsensus/tmp/107_movie_aligned.mrc.ctfParam\n'
                         )


    def run(self):
    
      numberOfThreads= self.getIntParam('-t')
      inputFile= self.getParam('-i')
      samplingRate= self.getDoubleParam('-s')
      downFactor= self.getDoubleParam('-d')
      outDir= self.getParam('-o')
      invert_contrast= self.checkParam('--invert_contrast')
      phase_flip= self.checkParam('--phase_flip')

      argsList=[]
      with open(inputFile) as f:
        for line in f:
          if line.startswith("#"):
            continue
          else:
            lineArray= line.split()
            if len(lineArray)<1: continue
            mic_fname= lineArray[0]
            mic_basename=  os.path.basename(mic_fname)
            out_name= os.path.join(outDir, mic_basename)
            ctf_fname= None
            if phase_flip:
              if len(lineArray)!=2:
                raise ValueError("Error, input file bad format. If -c option, it must have 2 cols: 'micFname ctfFname'")
              else:
                ctf_fname= lineArray[1]
            argsList+=[  (mic_fname, samplingRate, out_name, ctf_fname, 
                          invert_contrast, phase_flip, downFactor)]
      Parallel(n_jobs= numberOfThreads, backend="multiprocessing", verbose=1)(
                  delayed(preproOneMic)(*arg) for arg in argsList)

   
def preproOneMic(mic_fname, samplingRate, out_name, ctf_fname=None, invert_contrast=False, phase_flip=False, downFactor=1):
  """ Preprocess one micrograph
  """
  if os.path.isfile(out_name): return
  out_name_tmp= out_name+".tmp.mrc"
  if downFactor != 1:
      cmd = "xmipp_transform_downsample -i %s -o %s --step %f --method fourier" % (mic_fname, out_name_tmp, downFactor)
      print(cmd)
      check_call(cmd, shell=True)
      mic_fname = out_name_tmp
      
  if phase_flip:
      cmd = "xmipp_ctf_phase_flip -i %s -o %s --ctf %s --sampling %f"% (mic_fname, out_name_tmp, ctf_fname, samplingRate*downFactor)
      print(cmd)
      check_call(cmd, shell=True)
      mic_fname = out_name_tmp
      
  cmd = "xmipp_transform_normalize -i %s -o %s --method OldXmipp" % (mic_fname, out_name)
  if invert_contrast:
      cmd += " --invert"
  print(cmd)
  check_call(cmd, shell=True)
  try:
    os.remove(out_name_tmp)
  except:
    pass
if __name__ == '__main__':

    exitCode=ScriptPreproMics().tryRun()
    sys.exit(exitCode)
