'''
rsanchez@cnb.csic.es

'''
import keras
from keras import backend as K

MODEL_DEPTH= 4
DROPOUT_KEEP_PROB= 0.5
DESIRED_INPUT_SIZE=128
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
    assert keras.backend.backend() == 'tensorflow', 'Resize_bicubic_layer is compatible only with tensorflow'
    network= keras.layers.Lambda( lambda x: K.tf.image.resize_images(x, (DESIRED_INPUT_SIZE, DESIRED_INPUT_SIZE)),
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
  y_pred= keras.layers.Dense(num_labels, activation='softmax')(network),
  
  model = keras.models.Model(inputs=network_input, outputs=y_pred)
  
  optimizer= lambda learningRate: keras.optimizers.Adam(lr= learningRate, beta_1=0.9, beta_2=0.999,epsilon=1e-8)

  return model, optimizer

