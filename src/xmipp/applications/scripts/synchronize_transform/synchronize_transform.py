#!/usr/bin/env python

# ***************************************************************************
# * Authors:    Jose Luis Vilas Prieto (jlvilas@cnb.csic.es) 
#               Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
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
# *  e-mail address 'xmipp@cnb.csic.es'
# ***************************************************************************/

from typing import List, Optional, Tuple
    
from xmipp_base import XmippScript
import xmippLib
import itertools
import numpy as np
import scipy.sparse

import matplotlib.pyplot as plt


class ScriptSynchronizeTransform(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addParamsLine(' -i <pairwise>          : Pairwise alignment metadata')
        self.addParamsLine(' -o <alignment>         : Output alignment metadata')
        self.addParamsLine(' [--error <error>]      : Output error metadata (optional)')

    def run(self):
        inputFn = self.getParam('-i')
        outputFn = self.getParam('-o')
        if self.checkParam('--error'):
            errorFn = self.getParam('--error')
        else:
            errorFn = None

        inputMd = xmippLib.MetaData(inputFn)
        pairs, rotations, shifts, weights = self._readPairwiseAlignments(inputMd)
        ids, indices = np.unique(pairs, return_inverse=True)
        indices = indices.reshape(pairs.shape)
        
        n = len(ids)
        synchronizedRotations, rotErrors = self._synchronizeRotations(indices, n, rotations, weights)
        synchronizedShifts, shiftErrors = self._synchronizeShifts(indices, n, synchronizedRotations, shifts)
        
        if errorFn is not None:
            inputMd.setColumnValues(xmippLib.MDL_ANGLE_DIFF, rotErrors.tolist())
            inputMd.setColumnValues(xmippLib.MDL_SHIFT_DIFF, shiftErrors.tolist())
            inputMd.write(errorFn)
            
        outputMd = self._writeAlignments(ids, synchronizedRotations, synchronizedShifts)
        outputMd.write(outputFn)
            
    def _readPairwiseAlignments(self, md: xmippLib.MetaData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = md.size()
        ids = np.empty((n, 2), dtype=np.int64)
        rotations = np.empty((n, 3, 3))
        shifts = np.empty((n, 3))
        weights = np.empty(n)
        
        for i, objId in enumerate(md):
            id0 = md.getValue(xmippLib.MDL_REF, objId)
            id1 = md.getValue(xmippLib.MDL_IDX, objId)
            rot = md.getValue(xmippLib.MDL_ANGLE_ROT, objId)
            tilt = md.getValue(xmippLib.MDL_ANGLE_TILT, objId)
            psi = md.getValue(xmippLib.MDL_ANGLE_PSI, objId)
            x = md.getValue(xmippLib.MDL_SHIFT_X, objId)
            y = md.getValue(xmippLib.MDL_SHIFT_Y, objId)
            z = md.getValue(xmippLib.MDL_SHIFT_Z, objId)
            weight = md.getValue(xmippLib.MDL_WEIGHT, objId)
            
            rotation = xmippLib.Euler_angles2matrix(rot, tilt, psi)
            shift =  -(rotation.T @ np.array([x, y, z])) # Determine why
            
            ids[i] = (id0, id1)
            rotations[i] = rotation
            shifts[i] = shift
            weights[i] = weight
    
        return ids, rotations, shifts, weights
    
    def _writeAlignments(self, ids: np.ndarray, rotations: np.ndarray, shifts: np.ndarray) -> xmippLib.MetaData:
        md = xmippLib.MetaData()
        
        for itemId, rotation, shift in zip(ids, rotations, shifts):
            rot, tilt, psi = xmippLib.Euler_matrix2angles(rotation)
            x, y, z = shift
            
            objId = md.addObject()
            md.setValue(xmippLib.MDL_ITEM_ID, int(itemId), objId)
            md.setValue(xmippLib.MDL_ANGLE_ROT, float(rot), objId)
            md.setValue(xmippLib.MDL_ANGLE_TILT, float(tilt), objId)
            md.setValue(xmippLib.MDL_ANGLE_PSI, float(psi), objId)
            md.setValue(xmippLib.MDL_SHIFT_X, float(x), objId)
            md.setValue(xmippLib.MDL_SHIFT_Y, float(y), objId)
            md.setValue(xmippLib.MDL_SHIFT_Z, float(z), objId)
            
        return md
    
    def _synchronizeRotations(self, indices: np.ndarray, n: int, rotations: np.ndarray, weights: np.ndarray) -> np.ndarray:
        D = 3
        
        pairwise = scipy.sparse.lil_array((D*n, )*2)
        for (index0, index1), rotation, weight in zip(indices, rotations, weights):
            start0 = D*index0
            end0 = start0 + D
            start1 = D*index1
            end1 = start1 + D
            
            rotation = weight*rotation
            pairwise[start0:end0, start1:end1] = rotation
            pairwise[start1:end1, start0:end0] = rotation.T

        EYE = np.eye(D)
        for i in range(n):
            start = D*i
            end = start + D
            pairwise[start:end, start:end] = EYE

        pairwise = pairwise.tocsr()
        
        D2 = 2*D+1
        result = np.random.randn(n, D, D2)
        u, _, vt = np.linalg.svd(result, full_matrices=False)
        result = u @ vt
        x = result.reshape(n*D, D2)

        MAX_ITER = 16384
        EPS = 1e-6
        lastObjective = -np.inf
        for _ in range(MAX_ITER):
            y = pairwise @ x
            objective = np.dot(x.ravel(), y.ravel()) # tr(x.T @ y)
            
            improvement = objective - lastObjective
            if improvement < EPS:
                break

            u, _, vt = np.linalg.svd(y.reshape(n, D, D2), full_matrices=False)
            result = u @ vt
            x = result.reshape(n*D, D2)
            
            lastObjective = objective
        
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        w = np.square(s)
        w = w[:3]
        x = u[:,:3]
        
        result = x.reshape(n, D, D)

        u, _, vt = np.linalg.svd(result, full_matrices=False)
        result = u @ vt
        
        sign = np.linalg.det(result)
        result[:,:,2] *= sign[:,None]
            
        errors = np.empty(len(indices))
        for i, (index0, index1), rotation in zip(itertools.count(), indices, rotations):
            delta = result[index0].T @ rotation @ result[index1]
            error = np.degrees(np.arccos((np.trace(delta) - 1) / 2))
            errors[i] = error
        
        return result, errors

    def _synchronizeShifts(self, indices: np.ndarray, n: int, synchronizedRotations: np.ndarray, shifts: np.ndarray) -> np.ndarray:
        nCols = 3*n
        nRows = 3*len(indices)

        desing = scipy.sparse.lil_array((nRows, nCols))
        y = np.empty(nRows)
        for i, (index0, index1), shift in zip(itertools.count(), indices, shifts):
            startRow = 3*i
            endRow = startRow + 3
            start0 = 3*index0
            end0 = start0 + 3
            start1 = 3*index1
            end1 = start1 + 3
            
            desing[startRow:endRow,start0:end0] = -synchronizedRotations[index1] @ synchronizedRotations[index0].T
            desing[startRow:endRow,start1:end1] = np.eye(3)
            y[startRow:endRow] = shift
        desing = desing.tocoo()
        
        solution = scipy.sparse.linalg.lsqr(desing, y)
        x = solution[0].reshape((n, 3))
        
        predicted = desing @ solution[0]
        deltas = predicted - y
        deltas = deltas.reshape(len(indices), 3)
        errors = np.square(deltas).sum(axis=1)
        
        return x, errors
    
if __name__ == '__main__':
    ScriptSynchronizeTransform().tryRun()
