#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""

import mrcfile
import argparse
import sys, os
import numpy as np
import torch
from xmippPyModules.pcaAlignFunction.assessment import *

torch.cuda.is_available()
torch.cuda.current_device()
cuda = torch.device('cuda:0')


def read_images(mrcfilename):

    with mrcfile.open(mrcfilename, permissive=True) as f:
         emImages = f.data.astype(np.float32).copy()
    return emImages 

def save_images(data, outfilename):
    data = data.astype('float32')
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(data)
        
        
def signal_to_noise_statistic(images, radius):
    
    #create circular mask
    dim = images.size(dim=1)
    center = dim // 2
    
    if not radius:
        radius = dim // 2
        
    y, x = torch.meshgrid(torch.arange(dim) - center, torch.arange(dim) - center, indexing='ij')
    dist = torch.sqrt(x**2 + y**2)

    mask = dist <= radius
    mask = mask.float()    
      
    inv_mask = torch.logical_not(mask)
    
    mask = mask.bool()#.to(cuda)
    inv_mask = inv_mask.bool()#.to(cuda)
    
    pixels_in_mask = torch.masked_select(images, mask.unsqueeze(0))
    pixels_out_mask = torch.masked_select(images, inv_mask.unsqueeze(0))
    
    mean_value_in = torch.mean(pixels_in_mask)
    std_value_in = torch.std(pixels_in_mask)
    
    mean_value_out = torch.mean(pixels_out_mask)
    std_value_out = torch.std(pixels_out_mask)
    
    return(mean_value_in, std_value_in, mean_value_out, std_value_out)


def apply_scale(prjImages, expImages, radius):
    
    prjMeanSignal, prjStdSignal, prjMeanNoise, prjStdNoise = signal_to_noise_statistic(prjImages, radius)
    # print(prjMeanSignal, prjStdSignal, prjMeanNoise, prjStdNoise)
    expMeanSignal, expStdSignal, expMeanNoise, expStdNoise = signal_to_noise_statistic(expImages, radius)
    # print(expMeanSignal, expStdSignal, expMeanNoise, expStdNoise)
    
    a = prjStdSignal**2
    denom = torch.abs(expStdSignal**2 - expStdNoise**2)
    if denom == 0:
        denom = 0.000000001
    a = torch.sqrt(a/denom)
    b = prjMeanSignal - a*(expMeanSignal-expMeanNoise)
    print(a,b)
      
    prjImages = (prjImages - b)/a
    
    return prjImages
  
       
if __name__=="__main__":
          
    examples = """
    Examples:
    for scale leveling:
      preprocess.py -o references_scale.mrcs -i exp_file.mrcs -r ref_file.mrcs -rad 80
    for convert star to xmd
      preprocess.py -o xmipp_file.xmd -s star_relion.star --convert
    for create mrcs stack
      preprocess.py -o stack.mrcs --s star_relion.star --create_stack
    """
    parser = argparse.ArgumentParser(prog='preprocess_images.py', description="Program used for multiple purposes. "
                                     " It can be used to convert the Relion star file to Xmipp xmd format, "
                                     " create the mrcs stack from the star file, and scale the reference particles "
                                     " when they are generated from a volume that has not been reconstructed using Xmipp.",
                                     epilog = examples, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # common arguments
    required_args_group = parser.add_argument_group('required arguments')
    required_args_group.add_argument("-o", "--output", help="File output", required=True)
    
    # scale_leveling
    scale_leveling_group = parser.add_argument_group('scale_leveling', 'Arguments for scale leveling')
    scale_leveling_group.add_argument("-i", "--exp", help="input mrcs file for experimental images. It is necessary for scale leveling")
    scale_leveling_group.add_argument("-r", "--ref", help="input mrcs file for reference images")
    scale_leveling_group.add_argument("-rad", "--radius", type=float, help="Radius of the circular mask that will be used to define the background area (in pixels)")
    scale_leveling_group.add_argument("-b", "--batch", type=int, default=5000, help="Number of experimental images for the statistics. (default = 5000)")

    
    # convert_star
    convert_star_group = parser.add_argument_group('convert_star or create_stack', 'Arguments for converting star to xmd or create mrcs stack')
    convert_star_group.add_argument("-s", "--star", help="input star file")
    convert_star_group.add_argument("--convert", action="store_true", help="Convert Relion star to Xmipp xmd")
    convert_star_group.add_argument("--create_stack", action="store_true", help="Create mrcs stack from star file")
    convert_star_group.add_argument("--random_angles", action="store_true", help="Create xmd with random angles")   
    
    args = parser.parse_args()
    
    expFile = args.exp  
    prjFile = args.ref
    output = args.output
    radius =  args.radius
    batch = args.batch
    star = args.star
    create_stack = args.create_stack
    convert = args.convert
    random = args.random_angles

    if prjFile:
        #Read Images
        # mmap = mrcfile.mmap(expFile, permissive=True)
        # nExp = mmap.data.shape[0]
        prjImages = read_images(prjFile) 
        
        #convert ref images to tensor 
        tref= torch.from_numpy(prjImages).float().to("cpu")
        del(prjImages)
    
        # batch = min(batch, nExp)
        
        print("Scaling particles")
        Texp_numpy = np.load(expFile)
        Texp = torch.from_numpy(Texp_numpy)
        # Texp = torch.from_numpy(mmap.data[:batch].astype(np.float32)).to("cpu")
        tref = apply_scale(tref, Texp, radius)
        # del(Texp)
        
            #save preprocess images
        save_images(tref.numpy(), output)
        
    if convert:
        assess = evaluation()
        assess.convertRelionStarToXmd(star, output)
        
    if create_stack:
        assess = evaluation()
        print("Creating mrc stack")
        assess.createStack(star, output)
        
    if random:
        assess = evaluation()
        print("Generating XMD with random angles")
        assess.initRandomStar(star, output)














