# **************************************************************************
# *
# * Authors:  Ruben Sanchez (rsanchez@cnb.csic.es), April 2017
# * Authors:  Mikel Iceta (miceta@cnb.csic.es), March 2026
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
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf

MODEL_DEPTH = 6
DROPOUT_PROB = 0.1
DESIRED_INPUT_SIZE = 64
def main_network(input_shape, nData, l2RegStrength=1e-3, num_labels=1, resizeSize=None):
  '''
    input_shape: tuple:int,  ( height, width, nChanns )
    num_labels: int. Generally 1
    learningRate: float 
    int nData Expected data size (used to select model size)
  '''

  input_size = resizeSize if resizeSize is not None else DESIRED_INPUT_SIZE
    
  print("Model depth: %d"%MODEL_DEPTH)
  if input_shape!=(input_size, input_size, 1):
    network_input= keras.layers.Input(shape = (None, None, 1))
    #TODO: (nonenonenone) para tener un nº de canales (filter bank COSS)
    assert K.backend() == 'tensorflow', 'Resize_bicubic_layer is compatible only with tensorflow'
    # tf.image.resize_images was removed in TF 2.x; use tf.image.resize instead
    network= keras.layers.Lambda(lambda x: tf.image.resize(x, (input_size, input_size), method='bicubic'),
                                 name="resize_tf")(network_input)
  else:
    network_input = keras.layers.Input(shape = input_shape) 
    network = network_input

  for i in range(1, MODEL_DEPTH+1):
    network = keras.layers.Conv2D(filters=32, kernel_size=3, activation=None, padding='same')
    network = keras.layers.BatchNormalization(axis=-1)(network)
    network = keras.layers.Activation('relu')(network)
    #network = keras.layers.SpatialDropout2D(0.05)(network)
    if i != MODEL_DEPTH:
      network = keras.layers.MaxPooling2D(pool_size=2, strides=2)(network)

  network = keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(network)
  network = keras.layers.Flatten()(network)
  network = keras.layers.Dropout(DROPOUT_PROB)(network)

  network = keras.layers.Dense(2**9, activation='relu')(network)
  network = keras.layers.Dropout(DROPOUT_PROB)(network)
  # Ensure final logits are computed in float32 for numerical stability when using mixed precision
  activation = 'sigmoid' if num_labels == 1 else 'softmax'
  y_pred = keras.layers.Dense(num_labels, activation=activation, dtype='float32')(network)
  
  model = keras.models.Model(inputs=network_input, outputs=y_pred)
  
  # Use TF2 optimizer signature (learning_rate instead of lr)
  optimizer = lambda learningRate: keras.optimizers.Adam(learning_rate=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

  return model, optimizer
