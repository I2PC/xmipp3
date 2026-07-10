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
import os

import xmippLib
from xmipp_script import XmippScript

from xmippPyModules.deepEmbed import (
    setup_gpu,
    TripletNet,
    create_train_state,
    save_train_state,
    get_xmipp_preloaded_array,
    precompute_original_grid,
    make_centered_warper_multichan,
    make_batcher,
    run_epoch,
    train_step,
    sample_embeddings
)

class ScriptDeepEmbedTrain(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Train a deep center model')
        ## params
        self.addParamsLine(' -i <metadata>                : xmd file with the list of images')
        self.addParamsLine(' --omodel <fnModel>           : Model filename')
        self.addParamsLine('[--batchSize <N=8>]           : Batch size')
        self.addParamsLine('[--imgSize <Xdim=64>]         : Training image size')
        self.addParamsLine('[--gpu <id=0>]                : GPU Id')
        self.addParamsLine('[--learningRate <lr=0.0001>]  : Learning rate')
        self.addParamsLine('[--maxEpochs <N=100>]         : Max. Epochs')
        self.addParamsLine('[--sigmaShift <s=10>]         : Std.Dev. of the simulated shifts')
        self.addParamsLine('[--embeddingDim <s=128>]      : Embedding dimension')
        self.addParamsLine('[--embeddingPoints <N=100000>]: Embedding points')
        self.addParamsLine('[--embeddingK <K=100>]        : Embedding clusters')
        self.addParamsLine('--ocentroids <fn>             : Output file for centroids (.npy or .npz)')

    def run(self):
        fnXmd = self.getParam("-i")
        fnModel = self.getParam("--omodel")
        maxEpochs = int(self.getParam("--maxEpochs"))
        batch_size = int(self.getParam("--batchSize"))
        XdimOut = int(self.getParam("--imgSize"))
        gpuId = self.getParam("--gpu")
        learning_rate = float(self.getParam("--learningRate"))
        sigma_shift = float(self.getParam("--sigmaShift"))
        embeddingDim = int(self.getParam("--embeddingDim"))
        embeddingPoints = int(self.getParam("--embeddingPoints"))
        embeddingK = int(self.getParam("--embeddingK"))
        fnCentroids = self.getParam("--ocentroids")

        setup_gpu(gpuId)

        devices = jax.devices(backend="gpu")
        env_name = os.environ.get("CONDA_DEFAULT_ENV")
        env_prefix = os.environ.get("CONDA_PREFIX")

        assert len(devices) > 0

        print("Conda environment: %s", env_name)
        print("Conda prefix: %s", env_prefix)
        print(f"Visible devices in JAX: {devices}")

        mdExp = xmippLib.MetaData(fnXmd)
        fnImgs = mdExp.getColumnValues(xmippLib.MDL_IMAGE)

        model = TripletNet(d=embeddingDim)
        state = create_train_state(jax.random.PRNGKey(0), model, XdimOut, 5, learning_rate)

        pre_dev = get_xmipp_preloaded_array(fnImgs, XdimOut, K=5)
        coords, cx, cy = precompute_original_grid(XdimOut, XdimOut)
        warper = make_centered_warper_multichan(coords, cx, cy)

        key = jax.random.PRNGKey(42)
        next_triplet, step = make_batcher(warper, batch_size, pre_dev.shape[0], sigma_shift)
        margin = 0.2

        key, batch = next_triplet(pre_dev, key)
        state, _ = train_step(state, batch, margin=margin)
        jax.block_until_ready(state.params)

        N = int(pre_dev.shape[0])
        steps_per_epoch = math.ceil(N / batch_size)

        t0 = time.time()
        for epoch in range(1, maxEpochs + 1):
            state, key, loss_sum, dap_sum, dan_sum, viol_sum = run_epoch(
                step, pre_dev, state, key, steps_per_epoch, margin
            )

            loss_avg = float(loss_sum / steps_per_epoch)
            dap_avg = float(dap_sum / steps_per_epoch)
            dan_avg = float(dan_sum / steps_per_epoch)
            viol_avg = float(viol_sum / steps_per_epoch)

            dt = time.time() - t0
            print(f"[epoch {epoch:4d}/{maxEpochs}] loss={loss_avg:.8f}  "
                  f"d_ap={dap_avg:.8f}  d_an={dan_avg:.8f}  "
                  f"viol%={100 * viol_avg:.1f}  ({dt:.1f}s/epoch)")
            t0 = time.time()

        save_train_state(state, fnModel)

        print("Generating synthetic embeddings for centroid estimation...")
        emb_key = jax.random.PRNGKey(12345)
        Xemb = sample_embeddings(pre_dev, state, next_triplet, emb_key, embeddingPoints, batch_size)

        print(f"Running K-means with K={embeddingK} on {Xemb.shape[0]} points...")
        kmeans = KMeans(
            n_clusters=embeddingK,
            init="k-means++",
            n_init=10,
            max_iter=100,
            random_state=0,
        )
        kmeans.fit(Xemb)
        centroids = kmeans.cluster_centers_.astype(np.float32)
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)

        np.save(fnCentroids, centroids)

if __name__ == '__main__':
    exitCode = ScriptDeepEmbedTrain().tryRun()
    sys.exit(exitCode)
