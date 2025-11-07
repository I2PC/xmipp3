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

from typing import Tuple, List, NamedTuple, Optional, Union
import numpy as np

from xmipp_base import XmippScript
import xmippLib

class TiltData(NamedTuple):
    projectionMatrix: np.ndarray
    shift: np.ndarray
    coordinates2d: np.ndarray
        
def _computeGmmResponsibilities(
    distances2: np.ndarray,
    sigma2: Union[np.ndarray, float],
    weights: np.ndarray,
    d: int,
    returnLogLikelihood: bool = False,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    LOG2PI = np.log(2*np.pi)

    # Compute the pairwise distances

    
    exponent = np.multiply(distances2, -0.5 / sigma2, out=out)

    logSigma2 = np.log(sigma2)
    logWeights = np.log(weights)
    logCoefficient = -0.5*d*(LOG2PI + logSigma2)
    logMantissa = logWeights + logCoefficient
    exponent += logMantissa
    
    """
    u, v = _normalizeLogResponsibilities(exponent, 16)
    exponent -= u
    exponent -= v
    """
    
    logNorm = np.logaddexp.reduce(exponent, axis=1, keepdims=True)
    exponent -= logNorm

    gamma = np.exp(exponent, out=exponent) # Aliasing
    if returnLogLikelihood:
        logLikelihood = np.sum(logNorm)
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
        self.addParamsLine('-o <outputMetadata>         : Path to the output metadata with the reconstructed 3D coordinates')
        self.addParamsLine('-n <numberOfCoords>         : Number of reconstructed 3D coordinates')
        self.addParamsLine('--box <x> <y> <z>           : Box size')
        self.addParamsLine('[-t <outputTiltSeries>]     : Path to the output metadata with updated tilt series.')
        self.addParamsLine('[--sigma <noise=8.0>]       : Initial picking coordinate std deviation estimation')
        self.addParamsLine('[--alpha <alpha=0.5>]       : Dirichlet prior on component weights.')
        
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
        alpha = self.getDoubleParam('--alpha')
        
        if self.checkParam('-t'):
            outTsMetadataFn = self.getParam('-t')
        else:
            outTsMetadataFn = None
        
        coord2dMd = xmippLib.MetaData(inputMetadataFn)
        tsMd = xmippLib.MetaData(inputTsAliMetadataFn)
        tiltSeriesCoordinates = self.readTiltSeriesCoordinates(inputMetadataFn)
        transforms = self.readTiltSeriesProjectionTransforms(inputTsAliMetadataFn)
        tiltIds = set(tiltSeriesCoordinates.keys()) & set(transforms.keys())
        
        data = []
        for tiltId in tiltIds:
            matrix, shift = transforms[tiltId]
            coordinates2d = np.array(tiltSeriesCoordinates[tiltId])
            data.append(TiltData(matrix, shift, coordinates2d))
        
        positions, effectiveCounts, sigma = self.coordinateBackProjection(
            data=data,
            nCoords=nCoords,
            boxSize=boxSize,
            sigma=sigma,
            alpha=alpha
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
        
        if outTsMetadataFn is not None:
            tsMd.setColumnValues(xmippLib.MDL_SIGMANOISE, sigma.tolist())
            tsMd.write(outTsMetadataFn)
        
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
    
    def coordinateBackProjection(
        self,
        data: List[TiltData], 
        nCoords: int,
        sigma: float,
        alpha: float,
        boxSize: Tuple[int, int, int] 
    ) -> Tuple[np.ndarray, np.ndarray]:
        EPS = 1e-8
        TOL = 1e-3
        MAX_ITER = 128
        D = 2
        
        boundary = np.array(boxSize) / 2
        positions = np.random.uniform(
            low=-boundary,
            high=boundary,
            size=(nCoords, 3)
        )

        weights = np.full(nCoords, 1/nCoords)
        sigma2 = np.full(len(data), np.square(sigma))
        for _ in range(MAX_ITER):
            n = np.zeros(len(positions))
            backprojections = np.zeros_like(positions)
            count = 0
            matrices = np.zeros((len(positions), 3, 3))
            for i, tilt in enumerate(data):
                detections = tilt.coordinates2d
                shift = tilt.shift
                projectionMatrix = tilt.projectionMatrix
                projectionMatrix2 = projectionMatrix.T @ projectionMatrix

                projections = (projectionMatrix @ positions.T).T + shift
                deltas = detections[:,None] - projections[None,:]
                deltas2 = np.square(deltas, out=deltas)
                distances2 = np.sum(deltas2, axis=2)

                responsibilities = _computeGmmResponsibilities(
                    distances2=distances2,
                    sigma2=sigma2[i],
                    weights=weights,
                    d=D,
                    returnLogLikelihood=False
                )
                
                sigma2[i] = 0.5 * np.sum(responsibilities*distances2) / np.sum(responsibilities)
                
                contribution = responsibilities.sum(axis=0)
                matrices += contribution[:,None,None] * projectionMatrix2
                n += contribution
                
                centeredDetections = detections - tilt.shift
                updatedProjection = np.sum(responsibilities[:,:,None] * centeredDetections[:,None], axis=0)
                backprojections += (projectionMatrix.T @ updatedProjection.T).T
   
                count += len(responsibilities)
            
            n -= 1 - alpha  # Dirichlet prior
            mask = n > EPS
            n = n[mask]
            matrices = matrices[mask]
            backprojections = backprojections[mask]
            
            oldPositions = positions[mask]
            positions = (np.linalg.inv(matrices + EPS*np.eye(3)) @ backprojections[:,:,None]).squeeze()
            weights = n / n.sum()
            
            delta = np.mean(np.linalg.norm(oldPositions - positions, axis=-1))
            if delta < TOL:
                break
        
        return positions, n, np.sqrt(sigma2)
        
if __name__=="__main__":
    ScriptCoordinateBackProjection().tryRun()
