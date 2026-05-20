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
        
    
    def initPcaAnglesStar(self, angles, expXMD, outXMD):
        
        star = starfile.read(expXMD) 
                
        columns = ["anglePsi", "angleRot", "angleTilt", "shiftX", "shiftY", "shiftZ"]
        for column in columns:
            if column not in star.columns:
                star[column] = 0.0
                
        star.loc[:, "angleRot"] = angles[:, 0]
        star.loc[:, "angleTilt"] = angles[:, 1]
        star.loc[:, "anglePsi"] = angles[:, 2]
   
        starfile.write(star, outXMD, overwrite=True)       
        


    def estimatePose_v0(self, angle_triplet, expStar, matchPair_raw, shiftVec, nExp, apply_shifts):

        # 1. Máscara y Filtrado (Columna 5 es el flag)
        mask = matchPair_raw[:, 5] > 0.5
        valid_indices = torch.where(mask)[0] # Mantenemos índices en GPU si es posible
        
        num_valid = valid_indices.size(0)
        if num_valid == 0:
            return valid_indices, torch.empty((0, 3, 3), device=self.cuda), \
                   torch.empty((0,), device=self.cuda), torch.empty((0,), device=self.cuda)
    
        # Filtramos matchPair (usamos una versión local para cálculos de ángulos)
        matchPair = matchPair_raw[mask] 
    
        # 2. Cálculos de Shifts Relativos (Inverse Transform)
        indices_shift = matchPair[:, 4].long() 
        shiftVec_tensor = torch.as_tensor(shiftVec, dtype=torch.float32, device=self.cuda)
        angle_match = matchPair[:, 3].float()
        
        # Obtenemos los shifts X e Y correspondientes a cada match
        s_x = shiftVec_tensor[indices_shift, 0]
        s_y = shiftVec_tensor[indices_shift, 1]
        center = torch.tensor([0, 0], dtype=torch.float32, device=self.cuda)
        
        # Aplicamos la transformación inversa
        newShiftX, newShiftY = self.inverse_transform_shift(angle_match, s_x, s_y, center)
    
        # 3. Ángulos Finales y Matriz de Rotación
        # Ajuste de Psi (en GPU para evitar saltos a CPU)
        psi_adjusted = -matchPair[:, 3]
        psi_adjusted = torch.where(psi_adjusted > 180, psi_adjusted - 360, psi_adjusted)
        
        angle_triplet = torch.where(angle_triplet > 180, angle_triplet - 360, angle_triplet)
        angle_triplet_gpu = torch.as_tensor(angle_triplet, dtype=torch.float32, device=self.cuda)
        idx_triplet = matchPair[:, 1].long()
        
        # Convención ZYZ: a=Rot, b=Tilt, c=Psi
        # f_rot = angle_triplet_gpu[idx_triplet, 0]
        # f_tilt = angle_triplet_gpu[idx_triplet, 1]
        # f_psi = psi_adjusted + angle_triplet_gpu[idx_triplet, 2]
        f_psi = psi_adjusted + angle_triplet_gpu[idx_triplet, 0]
        f_rot = angle_triplet_gpu[idx_triplet, 1]
        f_tilt = angle_triplet_gpu[idx_triplet, 2]
        print(f_psi, f_rot, f_tilt)
    
        # Construcción de la matriz R (ZYZ)
        a, b, c = torch.deg2rad(f_rot), torch.deg2rad(f_tilt), torch.deg2rad(f_psi)
        ca, sa = torch.cos(a), torch.sin(a)
        cb, sb = torch.cos(b), torch.sin(b)
        cc, sc = torch.cos(c), torch.sin(c)
    
        R = torch.zeros((num_valid, 3, 3), device=self.cuda)
        R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = ca*cb*cc - sa*sc, -ca*cb*sc - sa*cc, ca*sb
        R[:, 1, 0], R[:, 1, 1], R[:, 1, 2] = sa*cb*cc + ca*sc, -sa*cb*sc + ca*cc, sa*sb
        R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = -sb*cc, sb*sc, cb
    
        # 4. Shifts Finales (Filtrados y en GPU)
        if apply_shifts:
            self.getShifts(expStar, nExp) 
            expShifts_filt = self.expShifts[mask] 
            
            shiftX_final = -s_x + expShifts_filt[:, 0]
            shiftY_final = -s_y + expShifts_filt[:, 1]
        else:
            shiftX_final = newShiftX
            shiftY_final = newShiftY
            
        shifts_final = torch.stack([shiftX_final, shiftY_final], dim=1)
    
        return valid_indices, R, shifts_final
    
    
    @torch.no_grad()
    def euler_zyz_to_matrix(self, psi, rot, tilt, degrees=True):
        """
        Convención:
            R = Rz(rot) @ Ry(tilt) @ Rz(psi)
            
        Parámetros:
        ----------
        psi, rot, tilt : Tensor o array-like

        Devuelve
        --------
        R : (N,3,3)
            Matrices de rotación.
        """
    
        psi = torch.as_tensor(psi, dtype=torch.float32, device=self.cuda)
        rot = torch.as_tensor(rot, dtype=torch.float32, device=self.cuda)
        tilt = torch.as_tensor(tilt, dtype=torch.float32, device=self.cuda)
    
        if degrees:
            psi  = torch.deg2rad(psi)
            rot  = torch.deg2rad(rot)
            tilt = torch.deg2rad(tilt)
    
        ca = torch.cos(rot)
        sa = torch.sin(rot)
    
        cb = torch.cos(tilt)
        sb = torch.sin(tilt)
    
        cc = torch.cos(psi)
        sc = torch.sin(psi)
    
        n = psi.shape[0]
    
        R = torch.empty((n, 3, 3), dtype=torch.float32, device=self.cuda)
    
        # Fila 1
        R[:, 0, 0] = ca * cb * cc - sa * sc
        R[:, 0, 1] = -ca * cb * sc - sa * cc
        R[:, 0, 2] = ca * sb
    
        # Fila 2
        R[:, 1, 0] = sa * cb * cc + ca * sc
        R[:, 1, 1] = -sa * cb * sc + ca * cc
        R[:, 1, 2] = sa * sb
    
        # Fila 3
        R[:, 2, 0] = -sb * cc
        R[:, 2, 1] = sb * sc
        R[:, 2, 2] = cb
    
        return R
    
    
    
    def estimatePose(self, angle_triplet, expStar, matchPair_raw, shiftVec, nExp, apply_shifts, filter_matches=True):
        """
        Estima la pose final de las partículas filtrando por el flag de mejor match.
        
        Arumentos:
            angle_triplet: Tensor (N_ref, 3) con [rot, tilt, psi] de la librería.
            matchPair_raw: Tensor (N_exp, 6) donde la col 5 es el flag de mejor match.
            shiftVec: Array/Tensor con los vectores de traslación de la búsqueda.
            apply_shifts: Booleano para decidir qué corrección de shift aplicar.
        """
        
        # 1. Filtramos solo los mejores matches (Flag en columna 5)
        if filter_matches:
            mask = matchPair_raw[:, 5] > 0.5
            valid_indices = torch.where(mask)[0]
        else:
            mask = torch.ones(matchPair_raw.shape[0], dtype=torch.bool, device=matchPair_raw.device)
            valid_indices = torch.arange(matchPair_raw.shape[0], device=matchPair_raw.device)
            
        
        num_valid = valid_indices.size(0)
        if num_valid == 0:
            return (valid_indices, 
                    torch.empty((0, 3, 3), device=self.cuda), 
                    torch.empty((0, 2), device=self.cuda))
    
        # Filtramos los datos necesarios
        matchPair = matchPair_raw[mask]
        
        # 2. Gestión de SHIFTS
        indices_shift = matchPair[:, 4].long() 
        shiftVec_tensor = torch.as_tensor(shiftVec, dtype=torch.float32, device=self.cuda)
        
        s_x = shiftVec_tensor[indices_shift, 0]
        s_y = shiftVec_tensor[indices_shift, 1]
        
        if apply_shifts:
            # Si aplicamos shifts externos (de expStar)
            self.getShifts(expStar, nExp) 
            expShifts_filt = self.expShifts[mask] 
            shiftX_final = -s_x + expShifts_filt[:, 0]
            shiftY_final = -s_y + expShifts_filt[:, 1]
        else:
            # Usamos la transformación inversa de los shifts de búsqueda
            angle_match = matchPair[:, 3].float()
            center = torch.tensor([0, 0], dtype=torch.float32, device=self.cuda)
            shiftX_final, shiftY_final = self.inverse_transform_shift(angle_match, s_x, s_y, center)
        
        shifts_final = torch.stack([shiftX_final, shiftY_final], dim=1)
    
        # 3. Cálculo de ÁNGULOS (Convención ZYZ coherente con generate_library)
        # psi_adjusted viene del matching (giro in-plane)
        psi_adjusted = -matchPair[:, 3]
        psi_adjusted = torch.where(psi_adjusted > 180, psi_adjusted - 360, psi_adjusted)
        # psi_adjusted = torch.where(psi_adjusted > 180, 360 -psi_adjusted, psi_adjusted)
        
        # Obtenemos los ángulos base de la librería [rot, tilt, psi]
        idx_triplet = matchPair[:, 1].long()
        lib_angles = torch.as_tensor(angle_triplet, dtype=torch.float32, device=self.cuda)[idx_triplet]
        
        # Mapeo de ángulos para la fórmula ZYZ:
        # alpha (a) = ROT, beta (b) = TILT, gamma (c) = PSI_LIB + PSI_MATCH
        f_rot   = lib_angles[:, 1]
        f_tilt  = lib_angles[:, 2] 
        f_psi   = lib_angles[:, 0] + psi_adjusted
    
        # 4. Construcción analítica de la Matriz R
        R = self.euler_zyz_to_matrix(
            psi=f_psi,
            rot=f_rot,
            tilt=f_tilt,
            degrees=True
        )
    
        return valid_indices, R, shifts_final   
    
    
    def determineR(self, angle_triplet):

        lib_angles = torch.as_tensor(angle_triplet, dtype=torch.float32, device=self.cuda)
        
        # Mapeo de ángulos para la fórmula ZYZ:
        # alpha (a) = ROT, beta (b) = TILT, gamma (c) = PSI_LIB + PSI_MATCH
        f_rot   = lib_angles[:, 1]
        f_tilt  = lib_angles[:, 2]
        f_psi   = lib_angles[:, 0] 
        # print(f_psi, f_rot, f_tilt)
    
        # 4. Construcción analítica de la Matriz R
        a, b, c = torch.deg2rad(f_rot), torch.deg2rad(f_tilt), torch.deg2rad(f_psi)
        ca, sa = torch.cos(a), torch.sin(a)
        cb, sb = torch.cos(b), torch.sin(b)
        cc, sc = torch.cos(c), torch.sin(c)
    
        R = torch.zeros((lib_angles.shape[0], 3, 3), device=self.cuda)
        
        # Fila 1
        R[:, 0, 0] = ca * cb * cc - sa * sc
        R[:, 0, 1] = -ca * cb * sc - sa * cc
        R[:, 0, 2] = ca * sb
        
        # Fila 2
        R[:, 1, 0] = sa * cb * cc + ca * sc
        R[:, 1, 1] = -sa * cb * sc + ca * cc
        R[:, 1, 2] = sa * sb
        
        # Fila 3
        R[:, 2, 0] = -sb * cc
        R[:, 2, 1] = sb * sc
        R[:, 2, 2] = cb
    
        return R  
         
            
            
            
            