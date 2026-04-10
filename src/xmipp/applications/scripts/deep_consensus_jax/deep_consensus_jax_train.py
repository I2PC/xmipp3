# **************************************************************************
# *
# * Authors:  Mikel Iceta (miceta@cnb.csic.es) 2026
# * Authors:  Carlos Oscar Sorzano (coss@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

import math, sys, time, os

import numpy as np
import jax
import jax.numpy as jnp
import optax

from flax.training import train_state, checkpoints

import xmippLib
from xmipp_script import XmippScript

from xmippPyModules.deepJaxConsensus import (
    setup_gpu,
    CryoCNNet,
    create_lr_schedule,
    compute_auc,
    xmipp_batch_generator,
    xmipp_preload_all,
    train_step,
    eval_step
)

class ScriptDeepConsensusJaxTrain(XmippScript):
    """
    Script to train a deep consensus network using JAX.
    """
    def __init__(self):
        XmippScript.__init__(self)

    def define_params(self):
        self.addUsageLine('Train a DeepScreen model using JAX')
        ## Parameters
        self.addParamsLine(' --pos <metadata> : xmd of true particle images')
        self.addParamsLine(' --neg <metadata> : xmd of false particle images')
        self.addParamsLine(' --omodel <fnModel> : output FOLDER for the trained model')
        self.addParamsLine('[--batchSize <N=64>] : Batch size')
        self.addParamsLine('[--imgSize <Xdim=128>] : Image size for training')
        self.addParamsLine('[--gpu <id=0>] : GPU ID to use')

        self.addParamsLine('[--learningRate <LR=0.0001>] : Learning rate for the optimizer')
        self.addParamsLine('[--maxEpochs <N=100>] : Number of training epochs')
        self.addParamsLine('[--nCores <N=16>] : Number of CPU cores for data loading')
        self.addParamsLine('[--augment] : Whether to apply data augmentation (default: False)')
        self.addParamsLine('[--bands <K=4>] : Number of frequency bands for the model')
    
    def run(self):
        fnModel = self.getParam("--omodel")
        maxEpochs = int(self.getParam("--maxEpochs"))
        batchSize = int(self.getParam("--batchSize")
        XDimOut = int(self.getParam("--imgSize"))
        gpuId = self.getParam("--gpu")
        learningRate = float(self.getParam("--learningRate"))
        nCores = int(self.getParam("--nCores"))
        augment = False
        if self.checkParam("--augment"):
            augment = True
        bands = 4
        if self.checkParam("--bands"):
            bands = int(self.getParam("--bands"))

        setup_gpu(gpuId)

        fnDict = dict()

        # Only POS and NEG are needed in training mode
        # For scoring check deep_consensus_jax_predict.py
        posXmdPath = self.getParam("--pos")
        mdPos = xmippLib.MetaData(posXmdPath)
        pos = mdPos.getColumnValues(xmippLib.MDL_IMAGE)
        nPos = len(mdPos)
        fnDict["positive"] = pos

        negXmdPath = self.getParam("--neg")
        mdNeg = xmippLib.MetaData(negXmdPath)
        neg = mdNeg.getColumnValues(xmippLib.MDL_IMAGE)
        nNeg = len(mdNeg)
        fnDict["negative"] = neg

        nTotal = nPos + nNeg

        # Set up the model, optimizer and state
        model = CryoCNNet()
        rng = jax.random.PRNGKey(0)
        dummy = jnp.ones((1, XDimOut, XDimOut, bands+2), dtype=jnp.float32)
        # Init model with dummy data to get the parameter shapes
        params = model.init(rng, dummy, training=True)["params"]
        lr_schedule = create_lr_schedule(learningRate, warmup_steps=500, total_steps=10000)
        opt = optax.adamw(lr_schedule)

        class TrainState(train_state.TrainState):
            dropout_rng: jax.Array
        
        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=opt,
            dropout_rng=rng
        )

        num_steps = math.ceil(nTotal / batchSize)

        train_gen = xmipp_train_batch_generator(fn_image_lists=fnDict, 
                                                batch_size=batchSize,
                                                XdimOut=XDimOut,
                                                K=bands,
                                                n_cores=nCores,
                                                augment=augment
                                                ) 
            
        #train_gen_prefetch = prefetch_to_device(train_gen, 2) # batches
        train_gen_prefetch = train_gen

        val_gen = xmipp_val_batch_generator(fn_image_lists=fnDict, 
                                                batch_size=batchSize,
                                                XdimOut=XDimOut,
                                                K=bands,
                                                n_cores=nCores
                                                ) 
        #val_gen_prefetch = prefetch_to_device(val_gen, 2) # batches
        val_gen_prefetch = val_gen  

        # TRAIN LOOP
        val_interval = 500 # Validate every 500 steps
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for step, batch in zip(range(num_steps), train_gen_prefetch):
            # Train
            state, train_loss, logits = train_step(state, batch)

            # Regular logs
            if step % 100 == 0:
                probs = jax.nn.sigmoid(logits)
                print(f"Step {step}/{num_steps}, Loss: {loss:.4f}, Mean prob: {probs.mean():.4f}")

            # Validation every N steps
            if step % val_interval == 0 and step > 0:
            
                val_losses = []
                all_probs = []
                all_labels = []

                for val_batch in val_gen_prefetch:
                    images, labels = val_batch
                    loss, probs = val_step(state.params, val_batch)
                    
                    val_losses.append(loss)
                    all_probs.append(probs.reshape(-1))
                    all_labels.append(labels.reshape(-1))
                
                val_loss = jnp.mean(jnp.stack(val_losses))
                probs = jnp.concatenate(all_probs)
                labels = jnp.concatenate(all_labels)

                # METRICS
                auc = compute_auc(probs, labels)
                preds = (probs > 0.5).astype(jnp.float32)
                acc = jnp.mean(preds == labels)

                print(f"[val] step {step} | loss {val_loss:.4f} | acc {acc:.3f} | AUC {auc:.3f}")

                # Check for EARLY STOP
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    checkpoints.save_checkpoint(
                        ckpt_dir="./checkpoints",
                        target=state,
                        step=step,
                        overwrite=True
                    )

                    print(f"New best model at step {step} with val loss {val_loss:.4f}")
                else:
                    patience_counter += 1
                
                if patience_coutner >= patience:
                    print(f"Early stopping at step {step} with best val loss {best_val_loss:.4f}")
                    break


if __name__ == "__main__":
    exitCode = ScriptDeepConsensusJaxTrain().tryRun()
    sys.exit(exitCode)
