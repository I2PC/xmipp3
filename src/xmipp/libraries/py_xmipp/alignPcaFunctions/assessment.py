#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""
import numpy as np
import starfile
import mrcfile
import torch
import os
import concurrent.futures


class evaluation:
    
    def __init__(self):
        torch.cuda.is_available()
        torch.cuda.current_device()
        self.cuda = torch.device('cuda:0')
        
    #for experimental images with starfile module
    def getAngle(self, prjStar):
        
        star=starfile.read(prjStar)
        self.angle_triplet = []
        
        for i in range(len(star)):
            
            phi = star["anglePsi"][i]
            rot = star["angleRot"][i]
            tilt = star["angleTilt"][i]
            angles = (phi, rot, tilt)
            self.angle_triplet.append(angles)

        return self.angle_triplet
    
    
    def getShifts(self, expStar, nExp):
        
        self.expShifts = torch.zeros(nExp, 2, device = self.cuda)       
        star=starfile.read(expStar)
        
        for i in range(nExp):
            
            shX = star["shiftX"][i]
            shY = star["shiftY"][i]           
            self.expShifts[i] = torch.tensor((shX, shY)).view(1,2) 

        return self.expShifts
    
    
    def getPosition(self, expStar, nExp):
    
        self.expPosition = torch.zeros(nExp, 3, device = self.cuda)       
        star=starfile.read(expStar)
        
        for i in range(nExp):
            
            rot = star["anglePsi"][i]
            shX = star["shiftX"][i]
            shY = star["shiftY"][i]           
            self.expPosition[i] = torch.tensor((rot, shX, shY)).view(1,3) 

        return self.expPosition
    
    
    def getShiftsRelion(self, expStar, sampling, nExp):
        
        self.expShifts = torch.zeros(nExp, 2, device = self.cuda)
        
        star=starfile.read(expStar)
        
        for i in range(nExp):
            
            shX = round(star["particles"]["rlnOriginXAngst"][i]/sampling)
            shY = round(star["particles"]["rlnOriginYAngst"][i]/sampling)
            
            self.expShifts[i] = torch.tensor((shX, shY)).view(1,2) 

        return self.expShifts
    
    
        #for experimental images with starfile module      
        
    def writeExpStar(self, prjStar, expStar, matchPair, shiftVec, nExp, apply_shifts, output):
        
        matchPair = matchPair.cpu().numpy()
        
        self.getAngle(prjStar)
        if apply_shifts:
            self.getShifts(expStar, nExp)
            expShifts = self.expShifts.cpu().numpy()
        star = starfile.read(expStar)
        new = output  
    
        # Adjustment of Psi angles
        psi_adjusted = matchPair[:, 3]
        psi_adjusted = np.where(psi_adjusted > 180, psi_adjusted - 360, psi_adjusted)
    
        columns = ["anglePsi", "angleRot", "angleTilt", "shiftX", "shiftY", "shiftZ"]
        for column in columns:
            if column not in star.columns:
                star[column] = 0.0
    
        # Updating columns in the dataframe
        angle_triplet = np.array(self.angle_triplet)
        shiftVec = np.array(shiftVec)
        star.loc[:, "anglePsi"] = psi_adjusted + angle_triplet[matchPair[:, 1].astype(int), 0]
        star.loc[:, "angleRot"] = angle_triplet[matchPair[:, 1].astype(int), 1]
        star.loc[:, "angleTilt"] = angle_triplet[matchPair[:, 1].astype(int), 2]
    
        if apply_shifts:
            star.loc[:, "shiftX"] = shiftVec[matchPair[:, 4].astype(int), 0] + expShifts[:, 0]
            star.loc[:, "shiftY"] = shiftVec[matchPair[:, 4].astype(int), 1] + expShifts[:, 1]
        else:
            star.loc[:, "shiftX"] = shiftVec[matchPair[:, 4].astype(int), 0]
            star.loc[:, "shiftY"] = shiftVec[matchPair[:, 4].astype(int), 1]
    
        starfile.write(star, new)
        
        
    def writeExpStarClass(self, prjStar, expStar, matchPair, shiftVec, nExp, apply_shifts, output):
        
        matchPair = matchPair.cpu().numpy()

        
        self.getAngle(prjStar)
        if apply_shifts:
            self.getShifts(expStar, nExp)
            expShifts = self.expShifts.cpu().numpy()
        star = starfile.read(expStar)
        new = output  
    
        # Adjustment of Psi angles
        psi_adjusted = -matchPair[:, 3]
        psi_adjusted = np.where(psi_adjusted > 180, psi_adjusted - 360, psi_adjusted)
    
        columns = ["anglePsi", "angleRot", "angleTilt", "shiftX", "shiftY", "shiftZ", "class"]
        for column in columns:
            if column not in star.columns:
                star[column] = 0.0
    
        # Updating columns in the dataframe
        angle_triplet = np.array(self.angle_triplet)
        shiftVec = np.array(shiftVec)
        star.loc[:, "anglePsi"] = psi_adjusted + angle_triplet[matchPair[:, 1].astype(int), 0]
        star.loc[:, "angleRot"] = angle_triplet[matchPair[:, 1].astype(int), 1]
        star.loc[:, "angleTilt"] = angle_triplet[matchPair[:, 1].astype(int), 2]
    
        if apply_shifts:
            star.loc[:, "shiftX"] = -shiftVec[matchPair[:, 4].astype(int), 0] + expShifts[:, 0]
            star.loc[:, "shiftY"] = -hiftVec[matchPair[:, 4].astype(int), 1] + expShifts[:, 1]
        else:
            star.loc[:, "shiftX"] = -shiftVec[matchPair[:, 4].astype(int), 0]
            star.loc[:, "shiftY"] = -shiftVec[matchPair[:, 4].astype(int), 1]
    
        star.loc[:, "class"] = matchPair[:, 6]#.astype(int)
        star["class"] = star["class"].astype(int)
        starfile.write(star, new)
        
        
    def writeExpStar_minScore(self, prjStar, expStar, matchPair, shiftVec, nExp, apply_shifts, output):
        
        matchPair = matchPair.cpu().numpy()
        
        self.getAngle(prjStar)
        if apply_shifts:
            self.getShifts(expStar, nExp)
            expShifts = self.expShifts.cpu().numpy()
        star = starfile.read(expStar)
        new = output 
        
        #Detewrmine shifts
        indices = torch.tensor(matchPair[:, 4], dtype=torch.long) 
        shiftVec = torch.tensor(shiftVec, dtype=torch.float64)
        angle = torch.tensor(matchPair[:, 3], dtype=torch.float64)
        
        
        shift_x = shiftVec[indices, 0]
        shift_y = shiftVec[indices, 1]
        center = torch.tensor([0, 0])
        newShiftX, newShiftY = self.inverse_transform_shift(angle, shift_x, shift_y, center) 
    
        # Adjustment of Psi angles
        psi_adjusted = -matchPair[:, 3]    
        psi_adjusted = np.where(psi_adjusted > 180, psi_adjusted - 360, psi_adjusted)
        # psi_adjusted = matchPair[:, 3]    
        # psi_adjusted = np.where(psi_adjusted > 180, psi_adjusted - 360, psi_adjusted)
              
    
        columns = ["anglePsi", "angleRot", "angleTilt", "shiftX", "shiftY", "shiftZ", "sel"]
        for column in columns:
            if column not in star.columns:
                star[column] = 0.0
    
        # Updating columns in the dataframe
        angle_triplet = np.array(self.angle_triplet)
        shiftVec = np.array(shiftVec)
        star.loc[:, "anglePsi"] = psi_adjusted + angle_triplet[matchPair[:, 1].astype(int), 0]
        star.loc[:, "angleRot"] = angle_triplet[matchPair[:, 1].astype(int), 1]
        star.loc[:, "angleTilt"] = angle_triplet[matchPair[:, 1].astype(int), 2]
    
        if apply_shifts:
            star.loc[:, "shiftX"] = -shiftVec[matchPair[:, 4].astype(int), 0] + expShifts[:, 0]
            star.loc[:, "shiftY"] = -shiftVec[matchPair[:, 4].astype(int), 1] + expShifts[:, 1]
        else:
            # star.loc[:, "shiftX"] = -shiftVec[matchPair[:, 4].astype(int), 0]
            # star.loc[:, "shiftY"] = -shiftVec[matchPair[:, 4].astype(int), 1]
            star.loc[:, "shiftX"] = newShiftX.cpu().numpy()
            star.loc[:, "shiftY"] = newShiftY.cpu().numpy()

    
        star.loc[:, "sel"] = matchPair[:, 5]#.astype(np.int32)
        star["sel"] = star["sel"].astype(int)
        
        #score
        star.loc[:, "score"] = matchPair[:, 2]
        
        starfile.write(star, new)
        
        
        
        
    def inverse_transform_shift(self, angle, shift_x, shift_y, center):
        
        angle = angle.to(dtype=torch.float64)
        shift_x = shift_x.to(dtype=torch.float64)
        shift_y = shift_y.to(dtype=torch.float64)
        center = center.to(dtype=torch.float64)

        theta = torch.deg2rad(angle)  
        cos_a, sin_a = torch.cos(theta), torch.sin(theta)
        
        # center = center.unsqueeze(0).expand(angle.shape[0], -1)       
        # cx, cy = center[:, 0], center[:, 1]
        neg_shift_x, neg_shift_y = -shift_x, -shift_y
    
        # new_shift_x = cos_a * neg_shift_x - sin_a * neg_shift_y + cx * (1 - cos_a) + cy * sin_a
        # new_shift_y = sin_a * neg_shift_x + cos_a * neg_shift_y + cy * (1 - cos_a) - cx * sin_a
        
        new_shift_x = cos_a * neg_shift_x - sin_a * neg_shift_y
        new_shift_y = sin_a * neg_shift_x + cos_a * neg_shift_y
        
        return new_shift_x, new_shift_y
    
        
    def writeExpStarRelion(self, prjStar, expStar, matchPair, shiftVec, sampling, nExp, apply_shifts, output):
        
        matchPair = matchPair.cpu().numpy()
        
        self.getAngle(prjStar)
        self.getShiftsRelion2(expStar, sampling, nExp)
        self.expShifts = self.expShifts.cpu().numpy()
        star=starfile.read(expStar)
        new = output #+ "newStar_exp.star"
        
        #Initializing columns
        star["particles"]["rlnAnglePsi"] = 0.0
        star["particles"]["rlnAngleRot"] = 0.0
        star["particles"]["rlnAngleTilt"] = 0.0
        star["particles"]["rlnOriginXAngst"] = 0.0
        star["particles"]["rlnOriginYAngst"] = 0.0
        star["particles"]["rlnOriginZAngst"] = 0.0

        for i in range(len(star)):
                                
            id = int(matchPair[i][1])
            new_psi = matchPair[i][3]
            posS = int(matchPair[i][4])           
            new_shiftX = float(shiftVec[posS][0])
            new_shiftY = float(shiftVec[posS][1])
            
            if(new_psi < 180):
                new_psi = new_psi
            else:
                new_psi = new_psi - 360

            star["particles"].at[i, "rlnAnglePsi"] = new_psi + self.angle_triplet[id][0]
            star["particles"].at[i, "rlnAngleRot"] = self.angle_triplet[id][1]
            star["particles"].at[i, "rlnAngleTilt"] = self.angle_triplet[id][2]
            
            if apply_shifts:
                star["particles"].at[i, "rlnOriginXAngst"] = (new_shiftX + self.expShifts[i][0])*sampling
                star["particles"].at[i, "rlnOriginYAngst"] = (new_shiftY + self.expShifts[i][1])*sampling
            else:
                star["particles"][i, "rlnOriginXAngst"] = new_shiftX*sampling
                star["particles"][i, "rlnOriginYAngst"] = new_shiftY*sampling 
            
        #priors  
        star["particles"].loc[:,"rlnOriginXPriorAngst"] = star["particles"]["rlnOriginXAngst"] 
        star["particles"].loc[:,"rlnOriginYPriorAngst"] = star["particles"]["rlnOriginYAngst"]
          
        star["particles"].loc[:,"rlnAnglePsiPrior"] = star["particles"]["rlnAnglePsi"] 
        star["particles"].loc[:,"rlnAngleRotPrior"] = star["particles"]["rlnAngleRot"]
        star["particles"].loc[:,"rlnAngleTiltPrior"] = star["particles"]["rlnAngleTilt"]
           
        starfile.write(star, new)
    
   
    def convertRelionStarToXmd(self, relionStar, output):
        star=starfile.read(relionStar)
        
        dict = {
                'rlnOriginXAngst': 'shiftX',
                'rlnOriginYAngst': 'shiftY',
                'rlnCoordinateX': 'xcoor',
                'rlnCoordinateY': 'ycoor',            
                'rlnAnglePsi': 'anglePsi',
                'rlnAngleRot': 'angleRot',             
                'rlnAngleTilt': 'angleTilt',
                'rlnDefocusU': 'ctfDefocusU',                
                'rlnDefocusV': 'ctfDefocusV',
                'rlnDefocusAngle': 'ctfDefocusAngle',
                'rlnCtfMaxResolution': 'ctfCritMaxFreq',            
                'rlnCtfFigureOfMerit': 'ctfCritFitting',            
                'rlnImageName': 'image'                
            }
                    
        sampling = star['optics']['rlnImagePixelSize'] 
        
        star['data'] = star['particles']
        del star['particles']
        
        id = range(1, len(star['data'])+1)
        star['data']['itemId'] = id
        
        star['data']['rlnOriginXAngst'] = star['data']['rlnOriginXAngst']/float(sampling)
        star['data']['rlnOriginYAngst'] = star['data']['rlnOriginYAngst']/float(sampling)
        
        star['data']['ctfVoltage'] = float(star['optics']['rlnVoltage'])
        star['data']['ctfSphericalAberration'] = float(star['optics']['rlnSphericalAberration'])
        star['data']['ctfQ0'] = float(star['optics']['rlnAmplitudeContrast'])
        star['data']['enabled'] = 1
        star['data']['flip'] = 0
        
        for relion, xmipp in dict.items():
            if relion in star['data'].columns:
                star['data'].rename(columns={relion: xmipp}, inplace=True)
        
        del star['optics']
        star = star['data'].drop(columns=star['data'].filter(regex='^rln', axis=1))
        
        starfile.write(star, output)
     
        
    def createStack(self, relionStar, output):
        star = starfile.read(relionStar)
        rln_image_name = star['particles']['rlnImageName']
        
        batch_mrc = []
        
        for line in rln_image_name:
            image_num, mrc_filename = line.split('@')
              
            with mrcfile.open(mrc_filename, permissive=True) as mrcs:
                image = mrcs.data[int(image_num)-1]
                
            batch_mrc.append(image.astype(np.float32))
        
        batch_mrc = np.stack(batch_mrc)
        
        # Save images
        with mrcfile.new(output, overwrite=True) as mrc_out:
            mrc_out.set_data(batch_mrc)
            
 
        #For random angle to generate initial random volume with classes
            
    # Generate random angles
    def generate_random_angles(self, num_images, angle_range=(-180, 180)):
        self.anglesRot = np.random.uniform(angle_range[0], angle_range[1], num_images)
        self.anglesTilt = np.random.uniform(angle_range[0], angle_range[1], num_images)
        return self.anglesRot, self.anglesTilt 
        
    #for experimental images with starfile module
    def initRandomStar(self, expXMD, outXMD):
        
        star = starfile.read(expXMD) 
        
        num_images = len(star)
        
        columns = ["anglePsi", "angleRot", "angleTilt", "shiftX", "shiftY", "shiftZ"]
        for column in columns:
            if column not in star.columns:
                star[column] = 0.0
        
        anglesRot, anglesTilt = self.generate_random_angles(num_images)
        
        star.loc[:, "anglePsi"] = 0.0
        star.loc[:, "angleRot"] = anglesRot
        star.loc[:, "angleTilt"] = anglesTilt  
   
        starfile.write(star, outXMD, overwrite=True)
 
            
            
            
            