#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""
import numpy as np
import pandas as pd
import starfile
import torch

class evaluation:
    
    def __init__(self):
        torch.cuda.is_available()
        torch.cuda.current_device()
        self.cuda = torch.device('cuda:0')
    
        #for experimental images with starfile module
    def updateExpStar(self, expStar, refClas, angle, shiftVec, output, dist):
        
        star = starfile.read(expStar)
        new = output + "_images.star"  
    
        # star.loc[:,"ref"] = refClas.int()+1
        
        star.loc[:, "ref"] = refClas.int().cpu().numpy() + 1
        star["ref"] = star["ref"].astype(int)
        
        # star.loc[:, "dist"] = dist.cpu().numpy().astype(float)
        
        dist_array = dist.detach().cpu().numpy()
        dist_array = np.nan_to_num(dist_array, nan=0.0, posinf=1e6, neginf=-1e6)
        star["cost"] = dist_array.astype(np.float64)

        columns = ["anglePsi", "angleRot", "angleTilt", "shiftX", "shiftY", "shiftZ"]
        for column in columns:
            # if column not in star.columns:
            star[column] = 0.0
    
        star.loc[:, "anglePsi"] = angle.astype(float)
        star.loc[:, "shiftX"] = shiftVec[:, 0].cpu().numpy().astype(float)
        star.loc[:, "shiftY"] = shiftVec[:, 1].cpu().numpy().astype(float)
    
        # starfile.write(star, new, overwrite=True)
        starfile.write({"particles": star}, new, overwrite=True)
        
        
        
    def createClassesStar(self, numClasses, classFile, count, output):
        
        data = {"ref": range(1, numClasses + 1),
                "image": [f"{i:06d}@{classFile}" for i in range(1, numClasses + 1)],
                "classCount": count}
        
        df = pd.DataFrame(data)

        newStar = output + "_classes.star" 
        starfile.write(df, newStar, overwrite=True)
        
 


 