#!/usr/bin/env python3
# **************************************************************************
# *
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
# *  e-mail address 'jmdelarosa@cnb.csic.es'
# *
# **************************************************************************

import numpy as np

from keras import Sequential, backend as K
from keras.applications import VGG16
from keras.layers import Flatten, Dense, Dropout

import xmippLib as xmipp
import xmipp_base


def crop(x, output_shape, shifts_):
    """
    Extract from image 'x' all possible crops consistent with 'output_shape' and 'shifts'
    :param x: an np.array with ndim = 2 representing a grayscale image
    :param output_shape: A list / tuple / np.array with ndim = 2 representing [nb_rows, nb_cols]
    :param shifts: A list / tuple / np.array with ndim = 2 representing the shifts in x and y axes for cropping the
    image ([x_shift, y_shift])
    :return: A list of np.arrays representing all possible crops consistent with the arguments. The np.arrays do not
    include the channels information and therefore they have ndim = 2.
    """
    assert len(shifts_) == 2, "Invalid shift lengths"
    assert shifts_[0] > 0 and shifts_[1] > 0, "Negative shifts"
    # copy input argument
    shifts = [shifts_[0], shifts_[1]]

    nb_segments_i = np.floor(x.shape[0] / float(output_shape[0]))
    nb_segments_j = np.floor(x.shape[1] / float(output_shape[1]))
    if nb_segments_i > 1:
        delta_i_max = int(np.floor((x.shape[0] - output_shape[0]) / float(nb_segments_i - 1.0)))
        if shifts[0] > delta_i_max:
            shifts[0] = delta_i_max
    else:
        shifts[0] = 0
    if nb_segments_j > 1:
        delta_j_max = int(np.floor((x.shape[1] - output_shape[1]) / float(nb_segments_j - 1.0)))
        if shifts[1] > delta_j_max:
            shifts[1] = delta_j_max
    else:
        shifts[1] = 0
    start_i = 0
    end_i = start_i + output_shape[0]
    Xs = []
    for i in range(int(nb_segments_i)):
        start_j = 0
        end_j = start_j + output_shape[1]
        for j in range(int(nb_segments_j)):
            assert end_i <= x.shape[0], "end_i >= x_shape[0]"
            assert end_j <= x.shape[1], "end_j >= x_shape[1]"
            Xs.append(x[start_i:end_i, start_j:end_j])
            start_j += shifts[1]
            end_j = start_j + output_shape[1]
        start_i += shifts[0]
        end_i = start_i + output_shape[0]
    return Xs


def particle_size_cnn(input_shape):
    #input_shape = list(input_shape[:-1]) + [3]
    model = Sequential()
    # Force data to be RGB
    # grayscale image to RGB image, as used in VGG16: FIXME: this results in topological order issues
    output_shape = list(input_shape[:-1]) + [3]
    # model.add(keras.layers.Lambda(lambda x: K.repeat_elements(x, 3, axis=-1),
    #                               input_shape=input_shape,
    #                               output_shape=output_shape,
    #                               name='repeat_channels')
    #           )
    # # TODO output_shape = input_shape
    # Load pretrained VGG16
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=output_shape)
    conv_base.trainable = False
    model.add(conv_base)

    # Custom top layer for regression
    model.add(Flatten(name='flatten'))
    model.add(Dense(32, activation='relu', name='dense'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation=None, name='regression'))

    return model


def get_conv_base(model):
    conv_base_list = [x for x in model.layers if x.name == 'vgg16']
    assert len(conv_base_list) == 1, 'Could not find base CNN model'
    return conv_base_list[0]


def set_trainable_last_conv_layer(model):
    conv_base = get_conv_base(model)
    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False


def set_not_trainable_last_conv_layer(model):
    conv_base = get_conv_base(model)
    conv_base.trainable = False
    for layer in conv_base.layers:
        layer.trainable = False


def save_weights(model, filepath, overwrite=True):
    # Set everything to trainable to save and load all weights
    conv_base = get_conv_base(model)
    restore_to_non_trainable = not conv_base.trainable
    set_trainable_last_conv_layer(model)
    model.save_weights(filepath, overwrite)
    if restore_to_non_trainable:
        set_not_trainable_last_conv_layer(model)


def load_weights(model, filepath, by_name=False, skip_mismatch=False, reshape=False):
    conv_base = get_conv_base(model)
    restore_to_non_trainable = not conv_base.trainable
    set_trainable_last_conv_layer(model)
    model.load_weights(filepath, by_name, skip_mismatch, reshape)
    if restore_to_non_trainable:
        set_not_trainable_last_conv_layer(model)


def get_image_config(img_size):
    config = {
        "img_rows": img_size,
        "img_cols": img_size,
        "data_format": K.image_data_format()
    }
    if config["data_format"] == 'channels_first':
        config["input_shape"] = (1, config["img_rows"], config["img_cols"])
    else:
        config["input_shape"] = (config["img_rows"], config["img_cols"], 1)
    return config


class ScriptParticleBoxsize(xmipp_base.XmippScript):
    _conda_env = xmipp_base.CondaEnvManager.CONDA_DEFAULT_ENVIRON

    def __init__(self):
        xmipp_base.XmippScript.__init__(self)
        self.particle_boxsize = None

    def defineParams(self):
        self.addUsageLine('.')
        ## params
        self.addParamsLine('--img_size <IMG_SIZE> : Number of rows of the image. The image is assumed to have the same number ' +
                             'of rows and columns')
        self.addParamsLine('--weights <WEIGHS_FILE>   : Name of the weights file to be used for prediction')
        self.addParamsLine('--micrographs <MICS_FILE>                 : Path to a TXT file that contains the complete paths of the micrographies that '
                                'should be processed')
        self.addParamsLine('--output <FILE>                          : Path to a TXT file that will contain the particle boxsize predicted for each '
                            'of the micrographs')
        ## examples
        ## TODO
        ## self.addExampleLine('   xmipp_particle_boxsize -p "Images/*xmp" -o all_images.sel -isstack')

    def run(self):
        # type: () -> object
        img_size = self.getParam('--img_size')
        weights = self.getParam('--weights')
        micrographs = self.getParam('--micrographs')
        output_file = self.getParam('--output')

        config = get_image_config(int(img_size))

        # Default shifts for cropping the images. TODO: add as arguments?
        shift = np.max([config["img_rows"], 1])
        shifts = np.repeat(shift, 2)


        def to_rgb(x):
            return np.repeat(
                x.reshape(*(config["img_rows"], config["img_cols"], 1)),
                3, axis=-1
            )

        model = particle_size_cnn(config["input_shape"])
        load_weights(model, weights)

        predictions = []
        with open(micrographs, 'r') as micrographs_file:
            for micrograph_path in micrographs_file:
                micrograph_path = micrograph_path.rstrip('\n')
                img = xmipp.Image(micrograph_path).getData()
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                img = (img - np.mean(img)) * 255.0  # Vgg16 is not standardized

                Xs = crop(img, (config["img_rows"], config["img_cols"]), shifts)
                # reshape to match the number of channels and add an extra dimension
                # to account for the batch size
                Xs = [np.expand_dims(to_rgb(x), axis=0) for x in Xs]
                micrograph_predictions = [float(model.predict(x)) for x in Xs]
                predictions.append(np.median(micrograph_predictions))
        self.particle_boxsize = int(np.round(np.median(predictions)))
        with open(output_file, 'w') as fp:
            fp.write(str(self.particle_boxsize) + '\n')

if __name__ == '__main__':
    ScriptParticleBoxsize().tryRun()
