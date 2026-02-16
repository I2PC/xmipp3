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
from xmippPyModules.alignPcaFunctions.assessment import *
from xmippPyModules.alignPcaFunctions.bnb_gpu import *

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


#Initial angles
def get_alignPCA_vinit_angles(class_averages, inplane_std=0.5, tilt_std=0.3, jitter_std=0.15):
    """
    Genera orientaciones iniciales y devuelve ángulos de Euler (Rot, Tilt, Psi) en grados.
    
    Args:
        class_averages: Tensor [N, H, W]
    Returns:
        angles: Tensor [N, 3] donde cada fila es [rot, tilt, psi] en grados.
    """
    N, H, W = class_averages.shape
    device = class_averages.device

    # 1. PCA 2D basada en momentos de inercia
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    coords = torch.stack([x, y], dim=-1).reshape(-1, 2)
    
    imgs = class_averages - class_averages.mean(dim=(-2, -1), keepdim=True)
    weights = torch.relu(imgs).reshape(N, -1) + 1e-8
    weights = weights / weights.sum(dim=-1, keepdim=True)
    
    cov = torch.einsum('ni,ix,iy->nxy', weights, coords, coords)
    _, eigvecs = torch.linalg.eigh(cov) 

    base_rots = torch.eye(3, device=device).repeat(N, 1, 1)
    base_rots[:, 0:2, 0:2] = eigvecs

    # 2. Romper simetrías (In-plane, Tilt y Jitter so3)
    theta = torch.randn(N, device=device) * inplane_std
    Rz = rotation_matrix_z(theta)
    
    tilt_rand = torch.randn(N, device=device) * tilt_std
    Rx = rotation_matrix_x(tilt_rand)
    
    # Jitter en el álgebra so(3)
    eps = torch.randn(N, 3, device=device) * jitter_std
    jitter_rot = torch.matrix_exp(hat(eps))

    # 3. Rotación global (Diversidad de la semilla)
    q = torch.randn(4, device=device)
    q = q / q.norm()
    w, x_q, y_q, z_q = q
    global_rot = torch.tensor([
        [1 - 2*y_q**2 - 2*z_q**2, 2*x_q*y_q - 2*z_q*w,     2*x_q*z_q + 2*y_q*w],
        [2*x_q*y_q + 2*z_q*w,     1 - 2*x_q**2 - 2*z_q**2, 2*y_q*z_q - 2*x_q*w],
        [2*x_q*z_q - 2*y_q*w,     2*y_q*z_q + 2*x_q*w,     1 - 2*x_q**2 - 2*y_q**2]
    ], device=device)

    # Matriz final: Global * Jitter * Tilt * InPlane * PCA
    R = global_rot @ jitter_rot @ Rx @ Rz @ base_rots
    R = R.transpose(1,2)

    # 4. Extracción de ángulos de Euler (Convención ZYZ: Rot, Tilt, Psi)
    tilt = torch.acos(R[:, 2, 2].clamp(-1.0, 1.0))
    
    rot = torch.atan2(R[:, 1, 2], R[:, 0, 2])
    
    psi = torch.atan2(R[:, 2, 1], -R[:, 2, 0])

    # Convertir a grados para el archivo .star
    angles = torch.stack([rot, tilt, psi], dim=-1)
    return torch.rad2deg(angles)

# --- Funciones auxiliares ya optimizadas ---
def hat(v):
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    O = torch.zeros_like(vx)
    return torch.stack([
        torch.stack([ O, -vz,  vy], dim=-1),
        torch.stack([ vz,  O, -vx], dim=-1),
        torch.stack([-vy,  vx,  O], dim=-1),
    ], dim=-2)

def rotation_matrix_x(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    O, I = torch.zeros_like(theta), torch.ones_like(theta)
    return torch.stack([
        torch.stack([I, O, O], dim=-1),
        torch.stack([O, c,-s], dim=-1),
        torch.stack([O, s, c], dim=-1),
    ], dim=-2)

def rotation_matrix_z(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    O, I = torch.zeros_like(theta), torch.ones_like(theta)
    return torch.stack([
        torch.stack([ c,-s, O], dim=-1),
        torch.stack([ s, c, O], dim=-1),
        torch.stack([ O, O, I], dim=-1),
    ], dim=-2)
  
       
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
    required_args_group.add_argument("-i", "--exp", help="input mrcs file for experimental images.")
    required_args_group.add_argument("-o", "--output", help="File output", required=True)
    
    # scale_leveling
    scale_leveling_group = parser.add_argument_group('scale_leveling', 'Arguments for scale leveling')
    # scale_leveling_group.add_argument("-i", "--exp", help="input mrcs file for experimental images. It is necessary for scale leveling")
    scale_leveling_group.add_argument("-r", "--ref", help="input mrcs file for reference images")
    scale_leveling_group.add_argument("-rad", "--radius", type=float, help="Radius of the circular mask that will be used to define the background area (in pixels)")
    scale_leveling_group.add_argument("-b", "--batch", type=int, default=5000, help="Number of experimental images for the statistics. (default = 5000)")

    
    # convert_star
    convert_star_group = parser.add_argument_group('convert_star or create_stack', 'Arguments for converting star to xmd or create mrcs stack')
    convert_star_group.add_argument("-s", "--star", help="input star file")
    convert_star_group.add_argument("--convert", action="store_true", help="Convert Relion star to Xmipp xmd")
    convert_star_group.add_argument("--create_stack", action="store_true", help="Create mrcs stack from star file")
    convert_star_group.add_argument("--random_angles", action="store_true", help="Create xmd with random angles") 
    convert_star_group.add_argument("--initial_angles", action="store_true", help="Create xmd with initial angles using pca") 
    
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
    initial_angles = args.initial_angles

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
        
    if initial_angles:
        nBand = 1
        bnb = BnBgpu(nBand)
        print("Generating XMD with initial angles")
        expImages = read_images(expFile)
        texp = torch.from_numpy(expImages).float().to("cuda")
        radius = 60
        texp = texp * bnb.create_mask(texp, radius)
        del(expImages)
        initAngles = get_alignPCA_vinit_angles(texp)
        initAngles_numpy = initAngles.detach().cpu().numpy()
        assess = evaluation()
        assess.initPcaAnglesStar(initAngles_numpy, star, output)
        














