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

MODEL_DEPTH= 4
DROPOUT_KEEP_PROB= 0.5
DESIRED_INPUT_SIZE=256
def main_network(input_shape, nData, l2RegStrength=1e-5, num_labels=2):
  '''
    input_shape: tuple:int,  ( height, width, nChanns )
    num_labels: int. Generally 2
    learningRate: float 
    int nData Expected data size (used to select model size)
  '''

  if nData<1500:
    nFiltersInit=0
  elif 1500<=nData<20000:
    nFiltersInit=1
  else:
    nFiltersInit=2
    
  print("Model depth: %d"%MODEL_DEPTH)
  if input_shape!=(DESIRED_INPUT_SIZE,DESIRED_INPUT_SIZE, 1):
    network_input= keras.layers.Input(shape= (None, None, input_shape[-1]))
    assert K.backend() == 'tensorflow', 'Resize_bicubic_layer is compatible only with tensorflow'
    # tf.image.resize_images was removed in TF 2.x; use tf.image.resize instead
    network= keras.layers.Lambda(lambda x: tf.image.resize(x, (DESIRED_INPUT_SIZE, DESIRED_INPUT_SIZE), method='bicubic'),
                                 name="resize_tf")(network_input)
  else:
    network_input= keras.layers.Input(shape= input_shape) 
    network= network_input

  for i in range(1, MODEL_DEPTH+1):
    network= keras.layers.Conv2D(2**(nFiltersInit+i), max(3, 30//2**i), activation='relu',  padding='same',
                                                kernel_regularizer= keras.regularizers.l2(l2RegStrength) )(network)
    network= keras.layers.Conv2D(2**(nFiltersInit+i), max(3, 30//2**i), activation='linear',  padding='same',
                                                kernel_regularizer= keras.regularizers.l2(l2RegStrength) )(network)
    network= keras.layers.BatchNormalization()(network)
    network= keras.layers.Activation('relu')(network)
    if i!=MODEL_DEPTH:
      network= keras.layers.MaxPooling2D(pool_size= max(2, 7-(2*(i-1))), strides=2, padding='same')(network)

  network= keras.layers.AveragePooling2D(pool_size=4, strides=2, padding='same')(network)
  network= keras.layers.Flatten()(network)

  network= keras.layers.Dense(2**9, activation='relu',
                                kernel_regularizer= keras.regularizers.l2(l2RegStrength))(network)
  network= keras.layers.Dropout(1-DROPOUT_KEEP_PROB)(network)
  y_pred= keras.layers.Dense(num_labels, activation='softmax')(network)
  
  model = keras.models.Model(inputs=network_input, outputs=y_pred)
  
  # Use TF2 optimizer signature (learning_rate instead of lr)
  optimizer= lambda learningRate: keras.optimizers.Adam(learning_rate=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

  return model, optimizer

