#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""
import numpy as np
import torch
from torch.nn.functional import normalize
import time

class PCAgpu:
    
    def __init__(self, nBand):
        
        self.nBand = nBand 
         
        torch.cuda.is_available()
        torch.cuda.current_device()
        self.cuda = torch.device('cuda:0')
         
               
    def first_mean(self, firstBands, firstSet):        
        self.mean = torch.sum(firstBands , 0)/firstSet           
        return self.mean
    
    
    def first_variance (self, firstBands, firstSet):
        
        self.first_mean(firstBands, firstSet)
        
        self.firstCenter = firstBands - self.mean.repeat(firstSet,1)
        self.var = torch.square(self.firstCenter)
        self.var = torch.sum(self.var , 0)/firstSet  

        return(self.mean, self.var)   
    
    
    def first_covariance(self, firstBands, firstSet):
       
        self.first_variance(firstBands, firstSet)

        self.covariance = torch.zeros(self.mean.size(dim=0), self.mean.size(dim=0), device = self.cuda)
        
        self.covariance =torch.cov(firstBands.T)

        return self.covariance, self.mean, self.var
    
    
    def first_eigenvector(self, firstBands, firstSet, num_eig):
        
        self.first_covariance(firstBands, firstSet)
                      
        self.vals, self.vecs = torch.linalg.eigh(self.covariance)
        self.vals = torch.flip(self.vals,[0])
        self.vecs = torch.flip(self.vecs,[1])
                
        return self.mean, self.var, self.vals, self.vecs
                    
               
    #mean = (n/n+1)*mean[0] + (1/n+1)*image    
    def mean_update(self, band, mean, nIm):    
        
        self.meanUp = [torch.zeros(mean[n].size(dim=0), device = self.cuda) for n in range(self.nBand)]
        
        for n in range(self.nBand):        
            self.meanUp[n] = nIm * mean[n]
            self.meanUp[n] += band[n]
            self.meanUp[n] /= nIm+1  

        return self.meanUp   
 
    
    def var_update(self, band, mean, var, nIm):  
        
        self.varUp = [torch.zeros(mean[n].size(dim=0), device = self.cuda) for n in range(self.nBand)]    
       
        for n in range(self.nBand):        
            self.varUp[n] = nIm * var[n]
            self.varUp[n] += (band[n]-mean[n])**2
            self.varUp[n] /= nIm+1
        
        return self.varUp 
    
    
    #phi = (image - mean)T * eigenvectors
    def phiProjTrain(self, band, mean, vecs):

        self.phi = [torch.zeros(mean[n].size(dim=0), device = self.cuda) for n in range(self.nBand)] 
        
        for n in range(self.nBand):
            
            center = band[n] - mean[n]                     
            self.phi[n] = torch.mm(center.reshape(1,mean[n].size(dim=0)) , vecs[n])
            
        return self.phi
    
    
    def phiProj(self, band, vecs):
        
        self.proj = [torch.zeros(band[n].size(dim=0), device = self.cuda) for n in range(self.nBand)]
        
        for n in range(self.nBand):
            
            self.proj[n] = torch.mm(band[n].reshape(1,band[n].size(dim=0)) , vecs[n])
 
        return self.proj
    
    #eigenvalue = eigenvalue[0] +const(phi²-eigenvalue[0])
    #eigenvalue = eigenvalue[0]*(1-const) + const*phi²
    def eigenvalue_update(self, vals, phi, gamma):
        
        self.eigval = [torch.zeros(phi[n].size(dim=1), device = self.cuda) for n in range(self.nBand)]
    
        for n in range(self.nBand): 
            
           temp1 = vals[n]*(1-float(gamma))
           temp2 = (phi[n] * phi[n]) * float(gamma)
           temp1 = temp1.reshape(1,phi[n].size(dim=1))
           self.eigval[n] = temp1 + temp2

        return self.eigval

                
    def eigenvector_update(self, band, vecs, phi, mean, gamma, num_eig):
    
        self.vecs_update = [torch.zeros((x.shape[0],x.shape[1]), device = self.cuda) for x in vecs]
    
        for n in range(self.nBand):            
            
            temp1 = torch.zeros(mean[n].shape[0], device = self.cuda)
            temp2 = torch.full( (vecs[n].shape[0], num_eig[n]), 0.0, device = self.cuda)
            temp3 = torch.zeros(vecs[n].shape[0], device = self.cuda)
                
            temp1 = band[n] - mean[n] 
            temp1 = temp1.reshape(temp1.size(dim=0),1)
            aux = gamma * phi[n]            
            temp1 = torch.matmul(temp1 , aux) 
                        
            temp2 = phi[n] * vecs[n]
            temp2 = aux * temp2 
                                    
            temp3 = torch.cumsum(torch.mul(phi[n], vecs[n]), dim=1)
            temp3 = aux * 2 * temp3
    
            self.vecs_update[n] =  vecs[n] + temp1 - temp2 - temp3               
            self.vecs_update[n] = normalize(self.vecs_update[n])
    
        return self.vecs_update
    
    
    def error(self, eigvalue, variance, per_eig):
        
        self.eigs = torch.zeros(self.nBand, device=self.cuda)
        self.perc = torch.zeros(self.nBand, device=self.cuda)
        self.error = torch.zeros(self.nBand, device=self.cuda)
        
        for n in range(self.nBand):
            
            accum_vals = torch.cumsum(eigvalue[n],dim=1)
            variance_total = torch.sum(variance[n])
            error_vect = accum_vals/variance_total
            self.eigs[n] = (error_vect >= per_eig).nonzero()[0][1]
            self.perc[n] = ((self.eigs[n]+1)/variance[n].size(dim=0))*100
            self.error[n] = error_vect[0][int(self.eigs[n])]
            
        return(self.eigs, self.perc, self.error)
    
    
    def errorVariance(self, eigvalue, variance, per_eig):
        
        self.eigs  = torch.zeros(self.nBand, device=self.cuda, dtype=torch.long)
        self.perc  = torch.zeros(self.nBand, device=self.cuda)
        self.error = torch.zeros(self.nBand, device=self.cuda)

        for n in range(self.nBand):
            vals = eigvalue[n]         
            var  = variance[n]          
            N = var.numel()
            sum_var = var.sum()
            
            accum = torch.cumsum(vals, dim=1) / sum_var

            # if per_eig >= 1:
            #     k = N - 1
            # else:
            #     k = torch.searchsorted(accum.squeeze(0), per_eig).clamp(max=N-1)
            k = torch.searchsorted(accum.squeeze(0), per_eig).clamp(max=N-1)

            self.eigs[n]  = k
            self.perc[n]  = (k + 1) / N * 100
            self.error[n] = accum[0, k] 
            
        return self.eigs, self.perc, self.error
                  
    
    # def batchPCA(self, band, coef, firstSet, eigTotal):
    def batchPCA(self, band, coef, firstSet, eigTotal):
        
        print("-----batch PCA for initializing-----")
        self.Bmean = [torch.zeros(coef[n], device = self.cuda) for n in range(self.nBand)]
        self.Bvar = [torch.zeros(coef[n], device = self.cuda) for n in range(self.nBand)]
        self.Bvals = [torch.zeros(coef[n], device = self.cuda) for n in range(self.nBand)]
        self.Bvecs = [torch.zeros((coef[n], coef[n]), device = self.cuda)for n in range(self.nBand)]
                  
        for n in range(self.nBand):
            
            self.first_eigenvector(band[n][:firstSet], firstSet, eigTotal[n])  
            self.Bmean[n], self.Bvar[n], self.Bvals[n], self.Bvecs[n] = self.mean, self.var,  self.vals, self.vecs

        return(self.Bmean, self.Bvar, self.Bvals, self.Bvecs)
    
    
       
    def trainingPCAonline(self, band, coef, per_eig, batchPCA):

        if batchPCA: 
            firstSet = band[0].shape[0]
        else:   
            firstSet = 3 * int(torch.ceil(0.8 * coef[-1]))
            
        eigTotal = torch.zeros(self.nBand, dtype=int)
        for n in range(self.nBand):
            # eigTotal[n] = int(np.ceil(0.8 * coef[n]))
            eigTotal[n] = coef[n]
        
        print("Batch PCA")
        self.batchPCA(band, coef, firstSet, eigTotal)
        
        if batchPCA:
            for n in range(self.nBand):
                self.Bvals[n] = self.Bvals[n].view(1, coef)
        else:
       
            print("-----Training PCA-----")
            nProj = firstSet+1
            
            for i in range(nProj, band[0].size(dim=0)): 
            
                iMband = [band[n][i] for n in range(self.nBand)]
            
                gamma = 1/np.sqrt(i)                    
                self.mean_update(iMband, self.Bmean, i) 
                self.var_update( iMband, self.meanUp, self.Bvar, i)
                self.phiProjTrain(iMband, self.meanUp, self.Bvecs)
                self.eigenvalue_update(self.Bvals, self.phi, gamma)
                self.eigenvector_update(iMband, self.Bvecs, self.phi, self.meanUp, gamma, eigTotal)
                self.Bmean = self.meanUp
                self.Bvar = self.varUp
                self.Bvals =  self.eigval
                self.Bvecs = self.vecs_update

        
        # self.error(self.Bvals, self.Bvar, per_eig)
        self.errorVariance(self.Bvals, self.Bvar, per_eig)
        for n in range(self.nBand):
            #Using truncation and no error
            # trunc = self.Bvecs[0].size(dim=1)*per_eig
            # print("eigenvector %s ---- percentage %s" %(int(self.eigs[n]+1), "{:.2f}".format(self.perc[n])))
            #self.Bvecs[n] = self.Bvecs[n][:,:(int(trunc+1))]
            
            #Reshaping Eigenvectors
            print(f"eigenvector {int(self.eigs[n]+1)} ---- percentage {self.perc[n]:.2f}")
            
            self.Bvecs[n] = self.Bvecs[n][:,:(int(self.eigs[n]+1))]
            print(self.Bvecs[n].shape, flush=True)

        return(self.Bmean, self.Bvals, self.Bvecs)
    
 
