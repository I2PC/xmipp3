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

import re
import sys, os

from xmipp_base import *
from xmippPyModules.deepLearningToolkitUtils.utils import *
import numpy as np

import xmippLib


class ScriptDeepDenoising(XmippScript):
  _conda_env=CondaEnvManager.CONDA_DEFAULT_ENVIRON
  def __init__(self):
    XmippScript.__init__(self)

  def defineParams(self):
    self.addUsageLine('Denoise particles Deep Learning. It works in two modes:\n'
                      '* mode 1: Train a neural network using a training dataset of particles and projections. '
                      '* mode 2: Denoise a set of particles using an already trained network')

    self.addParamsLine(' --mode <mode> : "training"|"denoising". Select training or denoising mode')

    self.addParamsLine(' -i <noisyParticles> : Noisy particles .xmd to denoise or train network')
    self.addParamsLine(' [ -p <noisyParticles> ] : Projections .xmd to train network (mandatory) or to evaluate denoising')
    self.addParamsLine(' [ --empty_particles <emptyParticles> ] : Empty particles .xmd to make training more robust, optional')
    self.addParamsLine(' [ -o <denoisedParticles> ] : Denoised particles .xmd. Mandatory when mode=="denoising", ignored otherwise')
    self.addParamsLine(' [ -c <pathToNetworkConfJson>   ]      : A path to a json file that contains the '
                       'arguments required to create/use the network'
                       'see scipion-em-xmipp/xmipp/protocols/protocol_deep_denoising.')

    ## examples
    self.addExampleLine('trainNet net:  xmipp_deep_denoising -n ./netData -i ./params.json')

  def run(self):
    print("running")
    checkIf_tf_keras_installed()
    import json

    netArgsFname = self.getParam('-c')
    assert os.path.isfile(netArgsFname), "Error, %s is not a file"%netArgsFname
    with open(netArgsFname) as f:
      args= json.load(f)

    updateEnviron(args["builder"]["gpuList"])

    dataPathParticles = self.getParam('-i')

    if self.checkParam('-p'):
      dataPathProjections = self.getParam('-p')
    else:
      dataPathProjections = None

    if self.checkParam('--empty_particles'):
      dataPathEmpty = self.getParam('--empty_particles')
    else:
      dataPathEmpty = None

    mode = self.getParam('--mode')
    trainKeyWords = ["train", "training"]
    predictKeyWords = ["predict", "denoising"]

    if args["builder"]["modelType"] == "U-Net":
      from xmippPyModules.deepDenoising.unet import UNET as ModelClass
    elif args["builder"]["modelType"] == "GAN":
      from xmippPyModules.deepDenoising.gan import GAN as ModelClass
    else:
      raise ValueError('modelTypeName must be one of ["GAN", "U-Net"]')

    del args["builder"]["modelType"]
    model = ModelClass(**args["builder"])

    if mode in trainKeyWords:
      print("mode 1: training")
      assert dataPathProjections is not None, "Error, projections must be provided to train the network"
      model.train(args["running"]["lr"], args["running"]["nEpochs"], dataPathParticles,
                  dataPathProjections, dataPathEmpty)

    elif mode in predictKeyWords:
      self.predictDenoised( model, args["builder"]["boxSize"], dataPathParticles, dataPathProjections)
    else:
      raise Exception("Error, --mode must be training or denoising")

  def predictDenoised(self, model,  boxSize, inputParticlesMdName, dataPathProjections=None):
    from scipy.stats import pearsonr

    inputParticlesStackName = re.sub(r"\.xmd$", ".stk", inputParticlesMdName)
    outputParticlesMdName = self.getParam('-o')
    outputParticlesStackName = re.sub(r"\.xmd$", ".stk", outputParticlesMdName)

    #useProjections = not dataPathProjections # problem with this assignment because later on it doesn't enter line 153 if useProjections when in fact there is "input projections to train"
    useProjections = dataPathProjections  

    if useProjections:
      inputProjectionsStackName = re.sub(r"\.xmd$", ".stk", dataPathProjections)
      metadataProjections = xmippLib.MetaData(inputProjectionsStackName)
    else:
      metadataProjections=None

    dimMetadata = getMdSize(inputParticlesMdName)
    xmippLib.createEmptyFile(outputParticlesStackName, boxSize, boxSize, 1, dimMetadata)

    mdNewParticles = xmippLib.MetaData()

    I = xmippLib.Image()
    i = 0
    # yieldPredictions will compute the denoised particles from the particles contained in inputParticlesMdName and
    # it will yield a batch of denoisedParts, inputParts (and projectionParts if inputProjectionsStackName provided)
    for preds, particles, projections in model.yieldPredictions(inputParticlesMdName, metadataProjections):
      newRow = XmippMdRow()
      for pred, particle, projection in zip(preds, particles, projections): # Here we will populate the output stacks
        i += 1
        outputImgpath = ('%06d@' % (i,)) + outputParticlesStackName #denoised image path
        I.setData(np.squeeze(pred))
        I.write(outputImgpath)

        pathNoise = ('%06d@' % (i,)) + inputParticlesStackName #input image path

        newRow.setValue(xmippLib.MDL_IMAGE, outputImgpath)
        newRow.setValue(xmippLib.MDL_IMAGE_ORIGINAL, pathNoise)

        correlations_input_vs_denoised, _ = pearsonr(pred.ravel(), particle.ravel())
        newRow.setValue(xmippLib.MDL_CORR_DENOISED_NOISY, correlations_input_vs_denoised)

        if useProjections:
          pathProj = ('%06d@' % (i,)) + inputProjectionsStackName #projection image path
          newRow.setValue(xmippLib.MDL_IMAGE_REF, pathProj)
          correlations_proj_vs_denoised, _ = pearsonr(pred.ravel(), projection.ravel())
          newRow.setValue(xmippLib.MDL_CORR_DENOISED_PROJECTION, correlations_proj_vs_denoised)

        newRow.addToMd(mdNewParticles)

    mdNewParticles.write('particles@' + outputParticlesMdName, xmippLib.MD_APPEND) #save the particles in the metadata

if __name__ == '__main__':
  exitCode = ScriptDeepDenoising().tryRun()
  sys.exit(exitCode)

