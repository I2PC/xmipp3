#!/usr/bin/env python3

#/***************************************************************************
# *
# * Authors:    Carlos Oscar Sorzano coss@cnb.csic.es
# *
# * CSIC
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
#  ***************************************************************************/


import math
import sys
import time
import numpy as np
from sklearn.cluster import KMeans
import jax
import xmippLib
from xmipp_script import XmippScript

import sys
import numpy as np
import jax
import xmippLib
from xmipp_script import XmippScript

from xmippPyModules.deepEmbed import (
    DT_IMAGE,
    setup_gpu,
    make_preprocess_context,
    preprocess_images,
    load_train_state,
    make_embedder,
    load_centroids,
    assign_to_centroids,
)

class ScriptDeepEmbedTrain(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Train a deep center model')
        ## params
        self.addParamsLine(' -i <metadata>                : xmd file with the list of images')
        self.addParamsLine(' --imodel <fnModel>           : Model filename')
        self.addParamsLine('--icentroids <fn>             : Output file for centroids (.npy or .npz)')
        self.addParamsLine('-o <fnXmd>                    : Output metadata')
        self.addParamsLine('[--gpu <id=0>]                : GPU Id')
        self.addParamsLine('[--imgSize <Xdim=64>]         : Prediction image size')
        self.addParamsLine('[--batchSize <s=1024>]        : Batch size')

    def run(self):
        fnXmd = self.getParam("-i")
        fnModel = self.getParam("--imodel")
        fnCentroids = self.getParam("--icentroids")
        fnOut = self.getParam("-o")

        batch_size = int(self.getParam("--batchSize"))
        XdimOut = int(self.getParam("--imgSize"))
        filterBankK = 5
        gpuId = self.getParam("--gpu")

        setup_gpu(gpuId)

        mdIn = xmippLib.MetaData(fnXmd)
        fnImgs = mdIn.getColumnValues(xmippLib.MDL_IMAGE)
        N = len(fnImgs)

        centroids = load_centroids(fnCentroids, normalize=True)
        embeddingDim = centroids.shape[1]
        centroids_dev = jax.device_put(centroids)

        preprocess_ctx = make_preprocess_context(XdimOut, K=filterBankK)

        _, state = load_train_state(
            fnModel,
            XdimOut=XdimOut,
            embedding_dim=embeddingDim,
            K=filterBankK,
            lr=3e-4,
            rng=jax.random.PRNGKey(0),
        )
        embedder = make_embedder(state.apply_fn)

        labels_all = np.empty(N, dtype=np.int32)
        scores_all = np.empty(N, dtype=np.float32)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_files = fnImgs[start:end]

            Xnp = preprocess_images(batch_files, preprocess_ctx, dtype=np.float32)
            Xdev = jax.device_put(Xnp).astype(DT_IMAGE)

            emb = embedder(state.params, Xdev)
            lab, sco = assign_to_centroids(emb, centroids_dev)

            labels_all[start:end] = np.asarray(jax.device_get(lab), dtype=np.int32)
            scores_all[start:end] = np.asarray(jax.device_get(sco), dtype=np.float32)

            print(f"[{end:7d}/{N:7d}] done")

        mdOut = xmippLib.MetaData(fnXmd)
        for i, objId in enumerate(mdOut):
            mdOut.setValue(xmippLib.MDL_REF, int(labels_all[i]), objId)

        mdOut.write(fnOut)

if __name__ == '__main__':
    exitCode = ScriptDeepEmbedTrain().tryRun()
    sys.exit(exitCode)
