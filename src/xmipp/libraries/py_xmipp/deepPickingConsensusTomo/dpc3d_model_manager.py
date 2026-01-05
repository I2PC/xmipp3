#!/usr/bin/env python3
""""
**************************************************************************
*
* Authors:  Mikel Iceta Tena (miceta@cnb.csic.es)
* 
*
* Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
* Initial version: jan 2025
**************************************************************************
"""
from dpc3d_data_manager import * # For the data loader
from dpc3d_model_definitions import * # For the architecture skeletons
from dpc3d_parameter_manager import * # For estimating NN parameters
from dpc3d_model_definitions import build_unet_3d

import tensorflow as tf
from tensorflow.keras import mixed_precision

# Memory management tricks
mixed_precision.set_global_policy('mixed_float16')

class ModelManDPC3D:
    def __init__(self, model, data_manager, params):
        """
        model: tf.keras.Model instance (UNet3D / classifier)
        data_manager: instance of DataManager
        params: dict or ParameterManager object
        """
        self.model = model
        self.data_manager = data_manager
        self.params = params
        self.batch_size = params.get("batch_size", 4)
        self.infer_batch_size = params.get("infer_batch_size", 8)
        self.aggregation_method = params.get("aggregation", "topk")  # max/topk/mean

    # ------------------------------
    # Training
    # ------------------------------
    def train(self, train_volumes, train_labels, val_volumes=None, val_labels=None,
              epochs=10, steps_per_epoch=100):
        """
        Training loop for patch-level classification
        """
        ds_train = self.data_manager.get_train_dataset(train_volumes, train_labels)
        
        if val_volumes is not None:
            ds_val = self.data_manager.get_train_dataset(val_volumes, val_labels)
        else:
            ds_val = None

        self.model.fit(
            ds_train,
            validation_data=ds_val,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )

    # ------------------------------
    # Patch-level inference for one patch
    # ------------------------------
    def predict_patch(self, patch):
        """
        patch: numpy array (Z,X,Y) or (Z,X,Y,1)
        returns: probability
        """
        if patch.ndim == 3:
            patch = patch[..., None]  # add channel dim
        patch = np.expand_dims(patch, 0)  # batch dim
        prob = self.model.predict(patch, batch_size=1)
        return float(prob[0, 0])

    # ------------------------------
    # Volume-level inference with tiling + aggregation
    # ------------------------------
    def infer_volume(self, volume):
        """
        volume: full 3D volume
        returns:
            volume_score: aggregated probability
            scores: list of patch probabilities
            coords: list of patch coordinates
        """
        vol = self.data_manager.normalize(volume)

        # Generate patches
        coords = self.data_manager.generate_patch_grid(vol.shape)
        patches = [self.data_manager.extract_patch(vol, c) for c in coords]
        patches = np.stack(patches)[..., None]

        # Predict in batches
        scores = []
        for i in range(0, len(patches), self.infer_batch_size):
            batch = patches[i:i+self.infer_batch_size]
            batch_scores = self.model.predict(batch, batch_size=len(batch))
            scores.extend(batch_scores[:, 0])
        scores = np.array(scores)

        # Aggregate patch scores
        volume_score = self.aggregate(scores)

        return volume_score, scores, coords

    # ------------------------------
    # Aggregation function
    # ------------------------------
    def aggregate(self, scores, topk=3):
        if self.aggregation_method == "max":
            return np.max(scores)
        elif self.aggregation_method == "mean":
            return np.mean(scores)
        elif self.aggregation_method == "topk":
            k = min(topk, len(scores))
            return np.mean(np.sort(scores)[-k:])
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    # ------------------------------
    # Save / Load
    # ------------------------------
    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)