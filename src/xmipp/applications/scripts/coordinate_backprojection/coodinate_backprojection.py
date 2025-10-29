#!/usr/bin/env python3

"""
**************************************************************************
*
* Authors:  Oier Lauzirika Zarrabeitia (olauzirika@cnb.csic.es)
*
* Unidad de  Bioinformatica of Centro Nacional de Biotecnologia, CSIC
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

from typing import Tuple, List, NamedTuple, Iterable, Optional, Union
import numpy as np

from xmipp_base import XmippScript
import xmippLib

class TiltData(NamedTuple):
    projectionMatrix: np.ndarray
    shift: np.ndarray
    coordinates2d: np.ndarray

def _computeProjectionDeltas(references: np.ndarray,
                             data: TiltData ) -> Tuple[np.ndarray, np.ndarray]:
    
    projectionMatrix = data.projectionMatrix
    detections = data.coordinates2d
    projections = (projectionMatrix @ references.T).T + data.shift
    deltas = detections[:,None] - projections[None,:]
    
    detections2 = (projectionMatrix.T @ detections.T).T
    projections2 = (projectionMatrix.T @ projections.T).T
    deltas2 = detections2[:,None] - projections2[None,:]
    
    return deltas, deltas2
    
def _normalizeLogResponsibilities(exponent: np.ndarray,
                                  nIter: int ) -> Tuple[np.ndarray, np.ndarray]:    
    # Apply Sinkhorn-Knopp
    u = np.zeros((exponent.shape[0], 1))
    v = np.zeros((1, exponent.shape[1]))
    tmp = None
    for _ in range(nIter):
        tmp = np.subtract(exponent, v, out=tmp)
        u = np.logaddexp.reduce(tmp, axis=1, keepdims=True, out=u)
        
        tmp = np.subtract(exponent, u, out=tmp)
        v = np.logaddexp.reduce(tmp, axis=0, keepdims=True, out=v)
        v = np.maximum(v, 0.0, out=v) # Don't boost unassigned components
    
    return u, v
        
def _computeGmmResponsibilities(distance2: np.ndarray,
                                sigma2: Union[np.ndarray, float],
                                weights: np.ndarray,
                                d: int,
                                returnLogLikelihood: bool = False,
                                out: Optional[np.ndarray] = None) -> np.ndarray:

    LOG2PI = np.log(2*np.pi)

    exponent = np.multiply(distance2, -0.5 / sigma2, out=out)

    logSigma2 = np.log(sigma2)
    logWeights = np.log(weights)
    logCoefficient = -0.5*d*(LOG2PI + logSigma2)
    logMantissa = logWeights + logCoefficient
    exponent += logMantissa
    
    u, v = _normalizeLogResponsibilities(exponent, 16)
    exponent -= u
    exponent -= v

    gamma = np.exp(exponent, out=exponent) # Aliasing
    if returnLogLikelihood:
        logLikelihood = np.sum(u)
        return gamma, logLikelihood
    else:
        return gamma

class ScriptCoordinateBackProjection(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Coordinate backprojection')
        self.addParamsLine('-i <inputMetadata>          : Path to a metadata file with all the 2D coordinates')
        self.addParamsLine('-a <tsMetadata>             : Path to a metadata file with the tilt-series alignment information')
        self.addParamsLine('-o <outputMetadata>         : Path to a metadata with the reconstructed 3D coordinates')
        self.addParamsLine('-n <numberOfCoords>         : Number of reconstructed 3D coordinates')
        self.addParamsLine('--box <x> <y> <z>           : Box size')
        self.addParamsLine('[--sigma <noise=8.0>]       : Picking noise std deviation estimation')
        
    def run(self):
        inputMetadataFn = self.getParam('-i')
        inputTsAliMetadataFn = self.getParam('-a')
        outputMetadataFn = self.getParam('-o')
        nCoords = self.getIntParam('-n')
        boxSize = (
            self.getIntParam('--box', 0),
            self.getIntParam('--box', 1),
            self.getIntParam('--box', 2)
        )
        sigma = self.getDoubleParam('--sigma')
        
        
        tiltSeriesCoordinates = self.readTiltSeriesCoordinates(inputMetadataFn)
        transforms = self.readTiltSeriesProjectionTransforms(inputTsAliMetadataFn)
        tiltIds = set(tiltSeriesCoordinates.keys()) & set(transforms.keys())
        
        data = []
        for tiltId in tiltIds:
            matrix, shift = transforms[tiltId]
            coordinates2d = np.array(tiltSeriesCoordinates[tiltId])
            data.append(TiltData(matrix, shift, coordinates2d))
        
        positions, effectiveCounts = self.coordinateBackProjection(
            data=data,
            nCoords=nCoords,
            boxSize=boxSize,
            sigma=sigma
        )
        
        outputMd = xmippLib.MetaData()
        for position, effectiveCount in zip(positions, effectiveCounts):
            x, y, z = position
            objId = outputMd.addObject()
            outputMd.setValue(xmippLib.MDL_X, x, objId)
            outputMd.setValue(xmippLib.MDL_Y, y, objId)
            outputMd.setValue(xmippLib.MDL_Z, z, objId)
            outputMd.setValue(xmippLib.MDL_LL, float(effectiveCount), objId)
        outputMd.write(outputMetadataFn)
        
    def readTiltSeriesProjectionTransforms(self, filename):
        result = dict()
        
        md = xmippLib.MetaData(filename)
        for objId in md:
            rot = md.getValue(xmippLib.MDL_ANGLE_ROT, objId) or 0.0
            tilt = md.getValue(xmippLib.MDL_ANGLE_TILT, objId) or 0.0
            psi = md.getValue(xmippLib.MDL_ANGLE_PSI, objId) or 0.0
            shiftX = md.getValue(xmippLib.MDL_SHIFT_X, objId) or 0.0
            shiftY = md.getValue(xmippLib.MDL_SHIFT_Y, objId) or 0.0
            tiltId = md.getValue(xmippLib.MDL_IMAGE_IDX, objId) or objId

            matrix = xmippLib.Euler_angles2matrix(rot, tilt, psi)
            matrix = matrix[:2]
            shift = np.array((shiftX, shiftY))

            result[tiltId] = (matrix, shift)

        return result
    
    def readTiltSeriesCoordinates(self, filename):
        result = dict()
        
        md = xmippLib.MetaData(filename)
        for objId in md:
            x = md.getValue(xmippLib.MDL_X, objId)
            y = md.getValue(xmippLib.MDL_Y, objId)
            tiltId = md.getValue(xmippLib.MDL_IMAGE_IDX, objId)
            
            coord2d = (x, y)
            if tiltId in result:
                result[tiltId].append(coord2d)
            else:
                result[tiltId] = [coord2d]
                
        return result
    
    def coordinateBackProjection(self,
                                 data: List[TiltData], 
                                 nCoords: int,
                                 sigma: float,
                                 boxSize: Tuple[int, int, int] ) -> Tuple[np.ndarray, np.ndarray]:
        EPS = 1e-12
        TOL = 1e-2
        D = 2
        
        boundary = np.array(boxSize) / 2
        positions = np.random.uniform(
            low=-boundary,
            high=boundary,
            size=(nCoords, 3)
        )

        weights = np.full(nCoords, 1/nCoords)
        sigma2 = np.square(sigma)
        
        deltas = None
        backprojectedDeltas = None
        oldLogLikelihood = -np.inf
        while True:
            n = np.zeros(len(positions))
            gradient = np.zeros_like(positions)
            count = 0
            logLikelihoods = np.zeros(len(data))
            for i, tilt in enumerate(data):
                deltas, backprojectedDeltas = _computeProjectionDeltas(
                    positions, 
                    tilt
                )
            
                deltas2 = np.square(deltas, out=deltas) # Aliasing
                distances2 = np.sum(deltas2, axis=2)

                responsibilities, logLikelihood = _computeGmmResponsibilities(
                    distances2, sigma2, weights, D,
                    returnLogLikelihood=True,
                    out=distances2 # Aliasing
                )
                
                backprojectedDeltas *= responsibilities[:,:,None]
   
                n += np.sum(responsibilities, axis=0)
                gradient += np.sum(backprojectedDeltas, axis=0)
                count += len(responsibilities)
                logLikelihoods[i] = logLikelihood
  
            
            gradient /= np.maximum(n[:,None], EPS)
            positions += gradient
            positions = np.clip(positions, -boundary, boundary)
            
            weights = n / count
            weights = np.maximum(weights, EPS)
            
            logLikelihood = np.logaddexp.reduce(logLikelihoods)
            improvement = (logLikelihood - oldLogLikelihood)
            if improvement < TOL:
                break # No improvement
            oldLogLikelihood = logLikelihood
            
        return positions, n
        
if __name__=="__main__":
    ScriptCoordinateBackProjection().tryRun()
