#!/usr/bin/env python3
# *****************************************************************************
# *
# * Authors:     J.L. Vilas (jlvilas@cnb.csic.es) [1]
# *
# * [1] Centro Nacional de Biotecnologia, CSIC, Spain
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 3 of the License, or
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
# *****************************************************************************

import sys
import os
import mrcfile

import numpy as np
import re
import scipy as sp
import tigre
from tigre.utilities.im3Dnorm import im3DNORM

import xmippLib
from xmipp_base import XmippScript


class TomogramReconstruction(XmippScript):
    _conda_env="xtomo_tigre" 

    def __init__(self):

        XmippScript.__init__(self)
        
        # Command line parameters
        self.fnTs          = None
        self.fnAngles      = None
        self.thickness     = None
        self.libRec        = None
        self.method        = None
        self.fnOut         = None
        self.normalizeTi   = None
        self.backprojector = None
        self.gpuId         = 0
        
        # Global parameters
        self.xdim = None
        self.ydim = None
        

    #  --------------------- DEFINE PARAMS -----------------------------
    def defineParams(self):
        # Description
        self.addUsageLine('This program provides a variety of algorithms to reconstruct tomogram from a set of tilt series.\n'
                          'The program will make use of a tilt series file (--tiltseries) which contains the tilt images and a set of angles given in a metadata file (--angles).\n'
                          'The reconstruction algorithms can be selectec with (--method) and are grouped in families:\n\n'
                          '\t F1 - Exact algorithms:\n'
                          '\t\t 1) WBP or FBP    - (default) Weighted back projection or filtered back projection.\n'
                          '\t\t 2) FDK           - Feldkamp-Davis-Kress algorithm.\n'
                          '\t\t\t\t  See L. A. Feldkamp, L. C. Davis, and J. W. Kress,  Practical cone-beam algorithm. JOSA (1984)\n\n'
                          
                          '\t F2 - Gradient-based algorithms:\n'
                          '\t\t 3) SIRT          - Simultaneus iterative reconstruction technique.\n'
                          '\t\t\t\t  See P. Gilbert, Iterative methods for the three-dimensional reconstruction of an object from projections, Journal of Theoretical Biology, 35,1, 105-107, (1972)\n'
                          '\t\t\t\t  See A.C. Kak and M. Slaney, Principles of Computerized Tomographic Imaging, Society of Industrial and Applied Mathematics, (2001)'
                          '\t\t 4) SART          - Simultaneous algebraic reconstruction technique\n'
                          '\t\t\t\t  See A. Andersen, A. Kak, Simultaneous Algebraic Reconstruction Technique (SART): A Superior Implementation of ART. Ultrasonic Imaging. 6,1, 81-94 (1984)\n'
                          '\t\t 5) OSSART        - Ordered Subset Simultaneous Algebraic Reconstruction Technique\n'
                          '\t\t\t\t  See Y. Censor and T. Elfving. Block-iterative algorithms with diagonally scaled oblique projections for the linear feasibility problem SIAM J. Matrix Anal. Appl. 24 40–58 (2002)\n.' 
                          '\t\t\t\t  See Ge, and J. Ming,  Ordered-subset simultaneous algebraic reconstruction techniques (OS-SART), Journal of X-Ray Science and Technology, 12, 3, 169-177, (2004)\n'
                          '\t\t 6) ASD-POCS      - Adaptative Steepest Descent with Projections Onto the Convex Set.\n'
                          '\t\t\t\t  See E.Y. Sidky, X. Pan, Image reconstruction in circular cone-beam computed tomography by constrained, total-variation minimization. Phys Med Biol, 53, 4777 (2008).\n'
                          '\t\t 7) OS-ASD-POCS   - Oriented Subsets version of ASD_POCS\n'
                          '\t\t\t\t  See E.Y. Sidky, X. Pan, Image reconstruction in circular cone-beam computed tomography by constrained, total-variation minimization. Phys Med Biol, 53, 4777 (2008).\n'
                          '\t\t 8) PCSD          - Projection-controlled steepest descent method\n'
                          '\t\t\t\t  See L. Liu, W. Lin, M. Jin. Reconstruction of sparse-view X-ray computed tomography using adaptive iterative algorithms.Computers in Biology and Medicine, 56, 97-106 (2015).\n'
                          '\t\t 9) AwPCSD        - Adaptive Weighted version of PCSD\n'
                          '\t\t\t\t  See M. Lohvithee, A. Biguri, M. Soleimani, Parameter selection in limited data cone-beam CT reconstruction using edge-preserving total variation algorithms, Physics in Medicine and Biology, (2017).\n'
                          '\t\t 10) AwASD-POCS   - Adaptive Weighted TV (edge preserving) version of ASD-POCS\n\n'
                          
                          '\t F3 - Krylov subspace algorithms:\n'
                          '\t\t 11) CGLS         - Conjugate Gradient Least Squares (CGLS) algorithm\n'
                          '\t\t\t\t  See A. Bjorck, Numerical Methods for Least Squares Problems, SIAM, Philadelphia, (1996)\n'
                          '\t\t 12) LSQR         - Least Squares QR factorization\n'
                          '\t\t\t\t  See C.C. Paige and M.A. Saunders, A robust incomplete factorization preconditioner for positive deﬁnite matrices, ACM Trans. Math. Software, 8 (1982)\n'
                          '\t\t 13) LSMR         - Least Squares MINRES\n'
                          '\t\t\t\t  See D.C.L. Fong, and  M. Saunders, LSMR: an iterative algorithm for sparse least-squares problems, SIAM J. Sci. Comput. 33, 2950–2991, (2011)\n'
                          '\t\t 14) hybrid-LSQR  - Hybrid Least Squares QR factorization\n'
                          '\t\t\t\t  See reference is needed\n'
                          '\t\t 15) AB-GMRES     - AB-generalized minimal residual method\n'
                          '\t\t\t\t  See K. Hayami, J.F. Yin, and T. Ito, GMRES methods for least squares problems, SIAM J. Matrix Anal. Appl., 31, 2400–2430 (2010)\n'
                          '\t\t\t\t  See Per c. Hansen, K. Hayami, K. Morikuni, GMRES methods for tomographic reconstruction with an unmatched back projector, Journal of Computational and Applied Mathematics, 413, 114352, (2022)\n'
                          '\t\t 16) BA-GMRES     - BA-generalized minimal residual method\n'
                          '\t\t\t\t  See K. Hayami, J.F. Yin, and T. Ito, GMRES methods for least squares problems, SIAM J. Matrix Anal. Appl., 31, 2400–2430 (2010)\n'
                          '\t\t\t\t  See Per c. Hansen, K. Hayami, K. Morikuni, GMRES methods for tomographic reconstruction with an unmatched back projector, Journal of Computational and Applied Mathematics, 413, 114352, (2022)\n'
                          '\t\t 17) IRN-TV-CGLS  - Using a binary mask\n'
                          '\t\t\t\t  See S. Gazzola, M.E. Kilmer, J.G Nagy, O.Semerci and E.L Miller, An inner–outer iterative method for edge preservation in image restoration and reconstruction. Inverse Problems, 36(2020)\n\n'
                          
                          '\t F4 - Statistical reconstruction:\n'
                          '\t\t 18) MLEM         - Maximum likelihood Expectation maximization\n'
                          '\t\t\t\t See J. M. Ollinger. Maximum-likelihood reconstruction of transmission images in emission computed tomography via the EM algorithm, IEEE Transactions on Medical Imaging, 12, 89-101, (1994)\n\n'
                          
                          '\t F5 - Variational methods:\n'
                          '\t\t 19) FISTA        - Fast iterative shrinkage thresholding algorithm.\n'
                          '\t\t\t\t  See Q. Xu, D. Yang, J. Tan, A. Sawatzky, M.A. Anastasio, Accelerated fast iterative shrinkage thresholding algorithms for sparsity-regularized cone-beam CT image reconstruction. Med Phys. 43,4, 1849-1872 (2016)\n'
                          '\t\t 20) SART-TV      - Simultaneous algebraic reconstruction technique with total variation\n'
                          '\t\t\t\t  See  H. Yu and G. Wang, A soft-threshold filtering approach for reconstruction from a limited number of projections, Phys. Med. Biol. 55, 13, 3905–3916 (2010)\n')
        
        # Params
        self.addParamsLine(' --tiltseries <fnTs>                  : Tilt series file. It must be a .mrc, .mrcs, .ali, or .st file')
        self.addParamsLine(' --angles <fnAngles>                  : Tlt file with the list of angles of the tilt series. ')
        self.addParamsLine(' --thickness <thickness>              : Thickness in pixels of the reconstructed tomogram')
        self.addParamsLine(' [--iter <iterations>]                : Number of iterations for the reconstructoin algorithm. Only used by SIRT (default 20), SART (20), OSSART(20), MLEM (500)')
        self.addParamsLine(' [--lambda <lmbda>]                   : Hyperparameter. The update will be multiplied by this number every iteration, to make the steps bigger or smaller. Default: lmbda=1.0 for all algorithms except for irn-tv-cgls for which lmbda=5.0')
        self.addParamsLine(' [--lambdared <lambdared>]            : Reduction multiplier for the hyperparameter. lambda=lambda*lambdared every iterations, so the steps can be smaller the further the update. Default=0.99')
        self.addParamsLine(' [--order <order>]                    : (Used by ossart)  Chooses the subset ordering strategy. Options are  1) ordered : uses them in the input order, but divided. 2) random: orders them randomply. 3) angularDistance: chooses the next subset with the biggest angular distance with the ones used (default)')
        
        self.addParamsLine(' [--alpha <alpha>]                    : Defines the TV hyperparameter. default is 0.002. However the paper mentions 0.2 as good choice')
        self.addParamsLine(' [--tviter <tviter>]                  : Defines the amount of TV iterations for algorithms that make use of a tvdenoising step in their iterations (OS-SART-TV, SART-TV, ASD_POCS, AWASD_POCS, FISTA. Default: 20 for SART-TV, 100 for FISTA. ')
        self.addParamsLine(' [--tvlambda <tvlambda>]              : Multiplier for TV. Default: 50 for SART-TV. 20 for FISTA.')
        self.addParamsLine(' [--alpha_red <alpha_red>]            : Defines the reduction rate of the TV hyperparameter')
        self.addParamsLine(' [--ratio <ratio>]                    : The maximum allowed image/TV update ration. If the TV update changes the image more than this, the parameter will be reduced. Default is 0.95')
        self.addParamsLine(' [--backprojector <backprojector>]    : This is the used algorithm for backprojecting at each iteration. ')
        
        self.addParamsLine(' [--blocksize <blocksize>]            : (Used by ossart) Sets the projection block size used simultaneously. If BlockSize = 1 OS-SART becomes SART and if  BlockSize = size(angles,2) then OS-SART becomes SIRT. Default is 20.')
        self.addParamsLine(' [--method <method>]                  : List of reconstruction algorithms. See the full list in the description.')

        self.addParamsLine(' [--filter <filterToApply> ]          : List of filters for FDK or FBP: *ram_lak  (default). * shepp_logan. * cosine. * hamming. *hann. The choice of filter will modify the noise and some discreatization errors, depending on which is chosen.')
        self.addParamsLine(' [--normalize <normalizeTi>]          : Preprocess the set of images to present zero mean and unit standard deviation')
        self.addParamsLine(' -o <fnOut>                           : Output filename of the tomogram.')
        self.addParamsLine(' [--gpu <gpuId>]                      : GPU Ids to be use in the image processing. (by default gpu 0) If this parameter is not set, the gpu 0 will be used') 
        
        # Examples       
        self.addExampleLine('xmipp_tomogram_reconstruction --tiltseries ts.mrc --angles angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method wbp --filter ram_lak')
        self.addExampleLine('xmipp_tomogram_reconstruction --tiltseries ts.mrc --angles angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method fdk --filter hamming')
        self.addExampleLine('xmipp_tomogram_reconstruction --tiltseries ts.mrc --angles angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method sirt --iter 20')
        self.addExampleLine('xmipp_tomogram_reconstruction --tiltseries ts.mrc --angles angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method sart')
        self.addExampleLine('xmipp_tomogram_reconstruction --tiltseries ts.mrc --angles angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method ossart')
        self.addExampleLine('xmipp_tomogram_reconstruction --tiltseries ts.mrc --angles angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method lsmr')
        self.addExampleLine('xmipp_tomogram_reconstruction --tiltseries ts.mrc --angles angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method asd-pocs')
        self.addExampleLine('xmipp_tomogram_reconstruction --tiltseries ts.mrc --angles angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method os-asd-pocs')
        self.addExampleLine('xmipp_tomogram_reconstruction --tiltseries ts.mrc --angles angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method awasd-pocs')
        self.addExampleLine('xmipp_tomogram_reconstruction --tiltseries ts.mrc --angles angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method irn-tv-cgls')
        self.addExampleLine('xmipp_tomogram_reconstruction --tiltseries ts.mrc --angles angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method ab-gmres')
        self.addExampleLine('xmipp_tomogram_reconstruction --tiltseries ts.mrc --angles angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method ab-gmres')
        self.addExampleLine('xmipp_tomogram_reconstruction --tiltseries ts.mrc --angles angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method hybrid_lsqr')
        self.addExampleLine('xmipp_tomogram_reconstruction --tiltseries ts.mrc --angles angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method lsqr')
        self.addExampleLine('xmipp_tomogram_reconstruction --tiltseries ts.mrc --angles angles.xmd --thickness 300 --gpu 0 -o tomogram.mrc --method cgls')
        

    def readInputParams(self):
        '''
        In this function the parameters are read. For for information about their use see the help
        '''

        self.fnTs          = self.getParam('--tiltseries')
        self.fnAngles      = self.getParam('--angles')
        self.thickness     = self.getIntParam('--thickness')
        self.method        = self.fixMethod(self.getParam('--method').lower())
        self.fnOut         = self.getParam('-o')
        self.normalizeTi   = self.getParam('--normalize').lower()
        self.gpuId         = self.getIntParam('--gpu') if self.checkParam('--gpu') else 0
        self.useTigreInterpolation = False
        
        if self.method == 'wbp':
            self.method = 'fbp'
        
        if self.method == 'fdk' or self.method == 'fbp':
            self.filterToApply = self.getParam('--filter').lower() if self.checkParam('--filter') else 'ram_lak'
            self.checkFilter(self.filterToApply, self.method)
            
        if self.method == 'sirt' or self.method == 'sart' or self.method == 'ossart':
            self.iterations    = self.getIntParam('--iter') if self.checkParam('--iter') else 20
            self.lmbda         = self.getDoubleParam('--lambda') if self.checkParam('--lambda') else 1.0
            self.lambdared     = self.getDoubleParam('--lambdared') if self.checkParam('--lambdared') else 0.999
            self.qualmeas      = ["RMSE", "SSD"]
            
        if self.method == 'ossart':
            self.blocksize     = self.getIntParam('--blocksize') if self.checkParam('--blocksize') else 10
            self.order         = self.getParam('--order') if self.checkParam('--order') else "random"
        
        if self.method == 'cgls' or self.method == 'lsqr' or self.method == 'lsmr' or self.method == 'hybridlsqr' or self.method == 'abgmres' or self.method == 'bagmres':
            self.iterations    = self.getIntParam('--iter') if self.checkParam('--iter') else 30
            
        if self.method == 'abgmres' or self.method == 'bagmres':
            self.backprojector = self.getParam('--backprojector') if self.checkParam('--backprojector') else None
             
        if self.method == 'asdpocs':
            self.iterations = self.getIntParam('--iter') if self.checkParam('--iter') else 30
            
        if self.method == 'lsmr' or self.method == 'asdpocs' or self.method == 'osasdpocs':
            self.lmbda         = self.getDoubleParam('--lambda') if self.checkParam('--lambda') else 1.0
            
        if self.method == 'asdpocs' or self.method == 'osasdpocs' or self.method == 'awasdpocs':
            self.alpha         = self.getDoubleParam('--alpha') if self.checkParam('--alpha') else 0.002
            self.tviter        = self.getIntParam('--tviter') if self.checkParam('--tviter') else 25
            self.lambdared     = self.getDoubleParam('--lambdared') if self.checkParam('--lambdared') else 0.9999
            self.alpha_red     = self.getDoubleParam('--alpha_red') if self.checkParam('--alpha_red') else 0.95
            self.ratio         = self.getDoubleParam('--ratio') if self.checkParam('--ratio') else 0.94
        
        if self.method == 'pcsd' or self.method == 'awpcsd':
            self.iterations    = self.getIntParam('--iter') if self.checkParam('--iter') else 20
            
        if self.method == 'osasdpocs':
            self.blocksize     = self.getIntParam('--blocksize') if self.checkParam('--blocksize') else 10
            self.iterations    = self.getIntParam('--iter') if self.checkParam('--iter') else 20

        if self.method == 'awasdpocs':
            self.blocksize     = self.getIntParam('--blocksize') if self.checkParam('--blocksize') else 20

        if self.method == 'irntvcgls':
            self.lmbda         = self.getDoubleParam('--lambda') if self.checkParam('--lambda') else 5.0
            self.iterations    = self.getIntParam('--iter') if self.checkParam('--iter') else 10
            self.niter_outer   = 2
            
        if self.method == 'fista':
            self.iterations    = self.getIntParam('--iter') if self.checkParam('--iter') else 100
            self.tviter        = self.getIntParam('--tviter') if self.checkParam('--tviter') else 100
            self.tvlambda      = self.getIntParam('--tvlambda') if self.checkParam('--tvlambda') else 20
        
        if self.method == 'sarttv':
            self.iterations    = self.getIntParam('--iter') if self.checkParam('--iter') else 30
            self.tviter        = self.getIntParam('--tviter') if self.checkParam('--tviter') else 50
            self.tvlambda      = self.getIntParam('--tvlambda') if self.checkParam('--tvlambda') else 50
            self.alpha_red    = self.getDoubleParam('--alpha_red') if self.checkParam('--alpha_red') else 0.95
            
        
        if self.method == 'mlem':
            self.iterations    = self.getIntParam('--iter') if self.checkParam('--iter') else 500
            
            

    def run(self):
        print('Starting ...')
        from scipy.spatial.transform import Rotation as R
        
        self.readInputParams()
        
        ts = self.readTiltSeries()
        print(os.path.splitext(self.fnAngles)[1])

        rotAngles = None
        offSets = None

        if os.path.splitext(self.fnAngles)[1] == '.tlt':
            tiltAngles = self.readTltFile(self.fnAngles)
            self.useTigreInterpolation= False
        else:
            self.useTigreInterpolation=True
            mdTs = xmippLib.MetaData(os.path.expanduser(self.fnAngles))
            tiltAngles = []
            rotAngles = []
            offSets = []
            for objId in mdTs:
                tilt = mdTs.getValue(xmippLib.MDL_ANGLE_TILT, objId)

                if mdTs.containsLabel(xmippLib.MDL_ANGLE_ROT):
                    rot = mdTs.getValue(xmippLib.MDL_ANGLE_ROT, objId)
                else:
                    rot = 0.0

                if mdTs.containsLabel(xmippLib.MDL_SHIFT_X):
                    sx = mdTs.getValue(xmippLib.MDL_SHIFT_X, objId)
                else:
                    sx = 0.0

                if mdTs.containsLabel(xmippLib.MDL_SHIFT_Y):
                    sy = mdTs.getValue(xmippLib.MDL_SHIFT_Y, objId)
                else:
                    sy = 0.0
                offSets.append([-sy,-sx])
                rotAngles.append([-rot, 0.0, 0.0])
                
                # Given ZYZ Euler angles
                zyz_angles = [tilt, rot, 0.0]  # replace with actual values

                # Create a rotation object from ZYZ Euler angles
                r_zyz = R.from_euler('ZYZ', zyz_angles, degrees=True)

                # Convert the rotation to ZXZ Euler angles
                zxz_angles = r_zyz.as_euler('ZXZ', degrees=True)
                tiltAngles.append(zxz_angles)

            tiltAngles = np.vstack(tiltAngles)
            offSets    = np.vstack(offSets)

        tiltAngles  = np.array(tiltAngles) * np.pi/180.0
        self.tigreReconstruction(ts, tiltAngles, rotAngles=None, offSets=offSets)

    def fixMethod(self, method):
        
        method = method.replace("-","")
        method = method.replace("_","")
        
        return method


    def readTltFile(self, fnTlt):
        try:
            with open(fnTlt, 'r') as tlt:
                lines = tlt.readlines()
    
                # Regular expression for matching numbers in the first column
                number_pattern = re.compile(r'^\s*([\d.-]+)')
    
                # Try to extract the first column using a comma or whitespace as the delimiter
                first_column = []
                for line in lines:
                    match = number_pattern.match(line)
                    if match:
                        first_column.append(float(match.group(1)))
    
                return np.array(first_column)
    
        except IOError:
            raise FileNotFoundError("Error parsing the tlt file")
    
    
    def fixFilter(self, candidate, method):
        if method == 'fdk':
            listFilters = ["ram_lak", "shepp_logan", "cosine", "hamming", "hann"]
            if '-' in candidate:
                candidate = candidate.replace('-', '_')
                
        elif method == 'fbp':
            listFilters = ["ram-lak", "shepp-logan", "cosine", "hamming", "hann"]
            if '_' in candidate:
                candidate = candidate.replace('_', '-')
        else:
            raise Exception('The selected filter presents a problem, please check the --filter')
        
        return candidate, listFilters
    
    def checkFilter(self, candidate, method):
        candidate, listFilters = self.fixFilter(candidate, method)

        if candidate in listFilters:
            self.filterToApply = listFilters[listFilters.index(candidate)]
        else:
            raise Exception('The selected filter does not exist, please check the --filter flag')
        print(self.filterToApply)

    def getGPUs(self):
        return str(self.gpuId)

    
    def tigreReconstruction(self, ts, tiltAngles, rotAngles=None, offSets=None):
        import tigre.algorithms as algs
        from tigre.utilities import gpu

        gpuids = gpu.GpuIds()
        gpuids.devices = [int(self.getGPUs())]

        geo = tigre.geometry(mode="parallel", nVoxel=np.array([self.xdim, self.ydim, self.thickness]))
        if self.useTigreInterpolation:
            if rotAngles is not None:
                geo.rotDetector = rotAngles

            if offSets is not None:
                geo.offDetector = offSets  # Offset of image from origin


        
        # EXACT ALGORITHMS
        if self.method == 'fbp': 
            reconstruction = algs.fbp(ts, geo, tiltAngles, filter=self.filterToApply, noneg=False, gpuids=gpuids)
     
        elif self.method == 'fdk':
            reconstruction = algs.fdk(ts, geo, tiltAngles, filter=self.filterToApply, noneg=False, gpuids=gpuids)
        
        # GRADIENT ALGORITHMS
        elif self.method =='sirt':
            reconstruction = algs.sirt(ts, geo, tiltAngles, self.iterations, lmbda=self.lmbda, lmbda_red=self.lambdared, verbose=True, noneg=False, gpuids=gpuids) #, Quameasopts=self.qualmeas, computel2=True)
            
        elif self.method == 'sart':
            reconstruction = algs.sart(ts, geo, tiltAngles, self.iterations, lmbda=self.lmbda, lmbda_red=self.lambdared, verbose=True,  noneg=False, gpuids=gpuids)#, Quameasopts=self.qualmeas, computel2=True)
            
        elif self.method == 'ossart':
            reconstruction = algs.ossart(ts, geo, tiltAngles, self.iterations, lmbda=self.lmbda, lmbda_red=self.lambdared, verbose=True,  noneg=False, blocksize=self.blocksize, OrderStrategy=self.order, gpuids=gpuids)#, Quameasopts=self.qualmeas, computel2=True)
       
        # KRYLOV ALGORITHMS     
        elif self.method == 'cgls':
            reconstruction = algs.cgls(ts, geo, tiltAngles, self.iterations, noneg=False, gpuids=gpuids)#, computel2=True)
            
        elif self.method == 'lsqr':
            reconstruction = algs.lsqr(ts, geo, tiltAngles, self.iterations,  noneg=False, gpuids=gpuids)#, computel2=True)
            
        elif self.method == 'lsmr':
            reconstruction = algs.lsmr(ts, geo, tiltAngles, self.iterations, lmbda=self.lmbda,  noneg=False, gpuids=gpuids)#, computel2=True, 
            
        elif self.method == 'hybridlsqr':
            reconstruction = algs.hybrid_lsqr(ts, geo, tiltAngles, self.iterations,  noneg=False, gpuids=gpuids)#, computel2=True)
            
        elif self.method == 'abgmres':   
            if self.backprojector is None:
                reconstruction = algs.ab_gmres(ts, geo, tiltAngles, self.iterations,  noneg=False, gpuids=gpuids)#, computel2=True)
            else:
                reconstruction = algs.ab_gmres(ts, geo, tiltAngles, self.iterations, backprojector="FDK",  noneg=False, gpuids=gpuids)#, computel2=True)
            
        elif self.method == 'bagmres':
            if self.backprojector is None:
                reconstruction = algs.ba_gmres(ts, geo, tiltAngles, self.iterations,  noneg=False)#, computel2=True)
            else:
                reconstruction = algs.ba_gmres(ts, geo, tiltAngles, self.iterations, backprojector="FDK",  noneg=False, gpuids=gpuids)#, computel2=True)
            
        elif self.method == 'asdpocs':
            epsilon = im3DNORM(tigre.Ax(algs.fdk(ts, geo, tiltAngles), geo, tiltAngles) - ts, 2) * 0.15
            reconstruction = algs.asd_pocs(ts, geo, tiltAngles, self.iterations, tviter=self.tviter, maxl2err=epsilon, 
                                           alpha=self.alpha, lmbda=self.lmbda, lmbda_red=self.lambdared, rmax=self.ratio, verbose=True,  noneg=False, gpuids=gpuids)
            
        elif self.method == 'osasdpocs':
            epsilon = im3DNORM(tigre.Ax(algs.fdk(ts, geo, tiltAngles), geo, tiltAngles) - ts, 2) * 0.15
            reconstruction = algs.os_asd_pocs(ts, geo, tiltAngles, self.iterations, tviter=self.tviter, maxl2err=epsilon,
                                               alpha=self.alpha, lmbda=self.lmbda, lmbda_red=self.lambdared, rmax=self.ratio, 
                                               verbose=True, blocksize=self.blocksize,  noneg=False, gpuids=gpuids)
                       
        elif self.method == 'awasdpocs':
            epsilon = im3DNORM(tigre.Ax(algs.fdk(ts, geo, tiltAngles), geo, tiltAngles) - ts, 2) * 0.15
            reconstruction = algs.awasd_pocs(ts, geo, tiltAngles, self.iterations, tviter=self.tviter, maxl2err=epsilon, 
                                             alpha=self.alpha, lmbda=self.lmbda, lmbda_red=self.lambdared, rmax=self.ratio, verbose=True, 
                                             delta=np.array([-0.005]), noneg=False, gpuids=gpuids)
            
        elif self.method == 'irntvcgls':
            reconstruction = algs.irn_tv_cgls(ts, geo, tiltAngles, self.iterations, lmbda=self.lmbda, niter_outer=self.niter_outer, noneg=False, gpuids=gpuids)
        
        elif self.method == 'fista':
            reconstruction = algs.fista(ts, geo, tiltAngles, self.iterations, tviter=self.tviter, tvlambda=self.tvlambda, noneg=False, gpuids=gpuids)
            
        elif self.method == 'sarttv':
            reconstruction = algs.sart_tv(ts, geo, tiltAngles, self.iterations, tvlambda=self.tvlambda, tviter=self.tviter, noneg=False, gpuids=gpuids)
            
        elif self.method == 'mlem':
            reconstruction = algs.mlem(ts, geo, tiltAngles, self.iterations, noneg=False, gpuids=gpuids)
        
        elif self.method == 'pcsd':
            reconstruction = algs.pcsd(ts, geo, tiltAngles, self.iterations, noneg=False, gpuids=gpuids)
            
        elif self.method == 'awpcsd':
            reconstruction = algs.aw_pcsd(ts, geo, tiltAngles, self.iterations, noneg=False, gpuids=gpuids)
            
        else:
            raise  Exception('The selected reconstruction method does not exist, check the --method flag')


        with mrcfile.new(self.fnOut, overwrite=True) as mrc:
            mrc.set_data(np.transpose(reconstruction, (2, 0, 1)))
        

    def getMethod(self):
        
        if self.method == 'FDK': 
            fdkFilters = ["ram-lak", "shepp-logan", "cosine", "hamming", "hann"]
        
            # Check if the word is in the list
            if self.filterToApply in fdkFilters:
                print("%s is in the list in the list of filters." % self.filterToApply)
            else:
                print("%s is not in the list." % self.filterToApply)
                
            return self.method, self.filterToApply
        
        elif self.method == 'sirt':
            return self.method, self.filterToApply
            

    def readTiltSeries(self):
        
        #Get tilt series
        ts_aux = mrcfile.read(self.fnTs)
        
        #Define volume
        dims = np.shape(ts_aux)
        self.xdim = dims[2]
        self.ydim = dims[1]
        self.nimages = dims[0]
        
        stdTi = []
        meanTi = []
        if self.normalizeTi == 'standard':
            for i in range(0, self.nimages):
                ti = ts_aux[i,:,:]
                stdTi = np.std(ti)
                meanTi = np.mean(ti)
                ti = (ti-meanTi)/stdTi
                ts_aux[i,:,:] = ti

        return ts_aux.astype(np.float32)

if __name__ == '__main__':
    '''
    scipion xmipp_tomogram_tigre_reconstrucion -g 0
    '''
    exitCode=TomogramReconstruction().tryRun()
    sys.exit(exitCode)