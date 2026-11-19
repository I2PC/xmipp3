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
from xmippPyModules.alignPcaFunctions.pca_gpu import *
from xmippPyModules.alignPcaFunctions.bnb_gpu import *
from xmippPyModules.alignPcaFunctions.reconstruct_gpu import *
from xmippPyModules.alignPcaFunctions.assessment import *
from xmippPyModules.alignPcaFunctions.init_angle_generation import *
import starfile
# from builtins import iter
# from _weakref import ref


def read_images(mrcfilename):

    with mrcfile.open(mrcfilename, permissive=True) as f:
        return f.data.astype(np.float32)
    
def load_vol(file_path, device='cuda'):

    with mrcfile.open(file_path, mode='r') as mrc:
        data = mrc.data.astype(np.float32)
        vol = torch.from_numpy(data).to(device)
        vol = (vol - vol.mean()) / vol.std()
        
    return vol

def save_vol(volume, filename, psize):
    vol_np = volume.detach().cpu().numpy().astype('float32')
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(vol_np)
        mrc.voxel_size = psize
        
  
       
if __name__=="__main__":
      
    parser = argparse.ArgumentParser(description="align images")
    parser.add_argument("-i", "--exp", help="input mrc file for experimental images", required=True)
    parser.add_argument("-s", "--sartExp", help="star file for experimental images", required=True)
    
    
    args = parser.parse_args()
    
    expFile = args.exp
    expStar = args.sartExp
           
    torch.cuda.is_available()
    torch.cuda.current_device()
    cuda = torch.device('cuda') 
    
    initializer = CommonLinesInitializer(image_size=128, pixel_size = 1.911, device=cuda)
    assess = evaluation()
    
    
    star = starfile.read(expStar)
    expImages = read_images(expFile)
    texp = torch.from_numpy(expImages).to(cuda)
    
      # Normalizar nombres de columnas
    star.columns = [str(c).lstrip("_") for c in star.columns]

    # Extraer ángulos
    angle_columns = ["anglePsi", "angleRot", "angleTilt"]
    angles = star[angle_columns].to_numpy(dtype=np.float32)

    # Convertir a matrices de rotación
    # rotations = assess.determineR(angles)
    rotations = assess.euler_zyz_to_matrix(psi=angles[:, 0], rot=angles[:, 1], tilt=angles[:, 2])
    
    diagnose = initializer.diagnose_orientations(texp, rotations, outlier_threshold=2.0, symmetry='D2')
    
    
    scores = diagnose["outlier_scores"].detach().cpu().numpy()
    median_res = diagnose["average_median_residual"].detach().cpu().numpy()
    mean_res = diagnose["average_mean_residual"].detach().cpu().numpy()
    bad_frac = diagnose["bad_pair_fraction"].detach().cpu().numpy()
    confidence = diagnose["average_mean_confidence"].detach().cpu().numpy()
    suspects = diagnose["suspect_mask"].detach().cpu().numpy()
    
    print("\nÍndice | Score | Mediana | Media | Bad pairs | Confianza | Estado")
    for i in range(len(scores)):
        print(
            f"{i:5d} | {scores[i]:8.3f} | {median_res[i]:8.4f} | "
            f"{mean_res[i]:8.4f} | {bad_frac[i]:8.3f} | "
            f"{confidence[i]:8.4f} | "
            f"{'SOSPECHOSO' if suspects[i] else 'OK'}"
        )
    
     
    
    
    