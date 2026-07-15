#!/usr/bin/env python3

#/***************************************************************************
# *
# * Authors:    
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
import os

import xmippLib
from xmipp_script import XmippScript


class ScriptClusterEmbeddingsKmeans(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Train a deep center model')
        ## params
        self.addParamsLine(' -i <metadata>                : Input latents')
        self.addParamsLine('--olabels <fn>             : Output path for assigned labels')
        self.addParamsLine('[--embeddingK <K=20>]         : Embedding clusters')

    def run(self):
        fnLatents = self.getParam("-i")
        embeddingK = int(self.getParam("--embeddingK"))
        fnCentroids = self.getParam("--olabels")

        Xemb = np.loadtxt(fnLatents)

        print(f"Running K-means with K={embeddingK} on {Xemb.shape[0]} points...")
        kmeans = KMeans(
            n_clusters=embeddingK,
            init="k-means++",
            n_init=10,
            max_iter=100,
            random_state=0,
        )
        kmeans.fit(Xemb)
        labels = kmeans.labels_
        
        np.savetxt(fnCentroids, labels)

if __name__ == '__main__':
    exitCode = ScriptClusterEmbeddingsKmeans().tryRun()
    sys.exit(exitCode)
