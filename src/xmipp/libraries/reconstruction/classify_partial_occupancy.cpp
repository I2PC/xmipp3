/***************************************************************************
 *
 * Authors:     Federico P. de Isidro-Gomez (federico.pdeisidro@astx.com)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

 #include "classify_partial_occupancy.h"
 #include "core/transformations.h"
 #include "core/metadata_extension.h"
 #include "core/multidim_array.h"
 #include "core/xmipp_image_extension.h"
 #include "core/xmipp_image_generic.h"
 #include "core/xmipp_image_base.h"
 #include "core/xmipp_fft.h"
 #include "core/xmipp_fftw.h"
 #include "core/linear_system_helper.h"
 #include "core/xmipp_error.h"
 #include "data/projection.h"
 #include "data/mask.h"
 #include "data/filters.h"
 #include "data/morphology.h"
 #include "data/unitCell.h"
 #include <iostream>
 #include <string>
 #include <sstream>
 #include "data/image_operate.h"
 #include <iostream>
 #include <cstdlib>
 #include <vector>
 #include <utility>
 #include <chrono>



// I/O methods ===================================================================
void ProgClassifyPartialOccupancy::readParams()
{
	XmippMetadataProgram::readParams();
 	fnVolR = getParam("--ref");
	fnMaskRoi=getParam("--mask_roi");
	padFourier = getDoubleParam("--padding");
	realSpaceProjector = checkParam("--realSpaceProjection");

	if (checkParam("--noise_est"))
	{
		fnNoiseEst = getParam("--noise_est");
		computeNoiseEstimation = false;
	}
	else if (checkParam("--noise_est_particles"))
	{
		numParticlesNoiseEst = getIntParam("--noise_est_particles");

		if (checkParam("--mask_protein"))
		{
			fnMaskProtein=getParam("--mask_protein");
		}
		else
		{
			REPORT_ERROR(ERR_IO, "Mask protein not provided> mandatory for noise estimation.");
		}
		computeNoiseEstimation = true;
	}

	if (checkParam("--unitcell"))
	{
		symmetryProvided = true;
		uc_sym = getParam("--unitcell");
	}
	else
	{
		symmetryProvided = false;
	}
	
}

void ProgClassifyPartialOccupancy::show() const
{
    if (!verbose)
        return;
	std::cout
	<< "Input particles:\t" << fn_in << std::endl
	<< "Reference volume:\t" << fnVolR << std::endl
	<< "Mask of the protein region:\t" << fnMaskProtein << std::endl
	<< "Mask of the region of interest to keep or subtract:\t" << fnMaskRoi << std::endl
	<< "Padding factor:\t" << padFourier << std::endl
	<< "Output particles:\t" << fn_out << std::endl;
}

void ProgClassifyPartialOccupancy::defineParams()
{
	//Usage
    addUsageLine("This algorithm classify a set of particles based on the presence of signal in a particular location of the specimen. \
				  The input particles must be projection subtraction keeping only the density of interes as in xmipp_subtract_projection, \
				  since subtraction parameters are relevant for calculation. Masks are expected to be binary. The algorithm is sensitive to \
				  the noise estimation quality, whixh is recomended to be calculated previosuly as in xmipp_subtract_projection due to the \
				  computational burden.");

    //Parameters
	XmippMetadataProgram::defineParams();
    addParamsLine("--ref <volume>\t						: Reference volume to subtract");
    addParamsLine("--mask_roi <mask_roi=\"\">     		: 3D mask for region of interest to keep or subtract, no mask implies subtraction of whole images");

	addParamsLine("--noise_est <noise_est=\"\">			: Previously calculated noise estimation for likelihood calculation.");
	addParamsLine("or --noise_est_particles <n=5000>    : Number of particles to calculate the noise estimation if it is not previously calculated. \
										                  The computational burden of this operation is significative, especially if a high number of particles is processed.");
	addParamsLine("[--mask_protein <mask_protein=\"\">]	: 3D mask for region of the specimen. Only required to calculate noise estimation.");

	addParamsLine("[--unitcell <sym>] 					: Extract a unit cell from volume for frequency profilling. Recomended if the specimen presents symmetry \
														  and in particular if the reference volume is recosntructed as this.");

	addParamsLine("[--realSpaceProjection]				: Project volume in real space to avoid Fourier artifacts");
	addParamsLine("[--padding <p=2>]					: Padding factor for Fourier projector");

	// Example
    addExampleLine("A typical use is:", false);
    addExampleLine("xmipp_classify_partial_occupancy -i input_particles.xmd --ref input_map.mrc --mask_roi mask_roi_vol.mrc --mask_protein mask_protein_vol.mrc \
													 -o output_particles.xmd --noise_est noise_estimation.mrc");
}

 void ProgClassifyPartialOccupancy::readParticle(const MDRow &r) 
 {
	r.getValueOrDefault(MDL_IMAGE, fnImgI, "no_filename");
	I.read(fnImgI);
	I().setXmippOrigin();
 }

 void ProgClassifyPartialOccupancy::writeParticle(MDRow &rowOut, double avg, double std, double zScore) 
 {
	#ifdef DEBUG_WRITE_PARICLE
	std::cout << "-------------- writing output particle" << std::endl;
	std::cout << "Final ll_I: " << avg << " --- Final ll_IsubP: " << std << " --- Final delta_ll: " << zScore << std::endl;
	std::cout << "fnImgI: " << fnImgI << std::endl;
	std::cout << "-------------- " << std::endl;
	#endif

	rowOut.setValue(MDL_IMAGE, fnImgI); 
	rowOut.setValue(MDL_AVG, avg); 
	rowOut.setValue(MDL_STDDEV, std); 
	rowOut.setValue(MDL_ZSCORE, zScore); 
 }


// Main methods ===================================================================
void ProgClassifyPartialOccupancy::preProcess() 
{
	// Read input volume, mask and particles metadata
	show();
	V.read(fnVolR);
	V().setXmippOrigin();

	// Read ROI mask
	vMaskRoi.read(fnMaskRoi);
	vMaskRoi().setXmippOrigin();

	// Apply ROI mask to reference volume (***NO SE SI ESTO TIENE SENTIDO O NO, TAMOS PROBANDO)
	V() *= vMaskRoi();

	// Create 2D circular mask to avoid edge artifacts after wrapping
	V.getDimensions(Xdim, Ydim, Zdim, Ndim);

	I().initZeros((int)Ydim, (int)Xdim);  //*** dimensions should be read from particles

	// Initialize projectors
	double cutFreq = 0.5;

	if (rank==0)
	{
			std::cout << "-------Charactizing frequencies-------" << std::endl;
			// Extract unit cell
			if (symmetryProvided)
			{
				uc_rmax = Xdim / 2;	// Assume square volume
				uc_x_origin = Xdim / 2;
				uc_y_origin = Ydim / 2;
				uc_z_origin = Zdim / 2;

				unitCellExtraction();
			}
			else
			{
				V_unitcell = V;
			}
			
			// Extract frequencies
			frequencyCharacterization();
			std::cout << "-------Frequencies characterized-------" << std::endl;

			if (computeNoiseEstimation)
			{
				// Read Protein mask
				vMaskP.read(fnMaskProtein);
				vMaskP().setXmippOrigin();

				std::cout << "-------Estimating noise -------" << std::endl;
				noiseEstimation();
				std::cout << "-------Noise estimated -------" << std::endl;
			}
			else
			{
				powerNoise.read(fnNoiseEst);
				std::cout << "Noise estimation read from: " << fnNoiseEst << std::endl;
				
				#ifdef DEBUG_OUTPUT_FILES
				size_t lastindex = fn_out.find_last_of(".");
				std::string debugFileFn = fn_out.substr(0, lastindex) + "_noisePower.mrc";
				powerNoise.write(debugFileFn);
				#endif
			}
			


		if (!realSpaceProjector)
		{
			// Initialize Fourier projectors
			std::cout << "-------Initializing projectors-------" << std::endl;

			projector = new FourierProjector(V(), padFourier, cutFreq, xmipp_transformation::BSPLINE3);
			std::cout << "Volume ---> FourierProjector(V(),"<<padFourier<<","<<cutFreq<<","<<xmipp_transformation::BSPLINE3<<");"<< std::endl;

			std::cout << "-------Projectors initialized-------" << std::endl;
		}
	}
	else
	{
		if (!realSpaceProjector)
		{
			projector = new FourierProjector(padFourier, cutFreq, xmipp_transformation::BSPLINE3);
		}
	}
 }

void ProgClassifyPartialOccupancy::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
 { 
	// Project volume and process projections 
	const auto sizeI = (int)XSIZE(I());

	processParticle(rowIn, sizeI);

	// Build projected ROI mask. Mask projection is always calculated in real space
	projectVolume(vMaskRoi(), PmaskRoi, sizeI, sizeI, part_angles.rot, part_angles.tilt, part_angles.psi, &roffset);

	// Apply binarization to projected mask, DO NOT NEEDED BECAUSE PROJECTING IN REAL SPACE
	// M = binarizeMask(PmaskRoi);

	#ifdef DEBUG_OUTPUT_FILES
	size_t dotPos = fnImgOut.find_last_of('.');
	I.write(fnImgOut.substr(0, dotPos) + "_I.mrcs");
	P.write(fnImgOut.substr(0, dotPos) + "_P.mrcs");
	PmaskRoi.write(fnImgOut.substr(0, dotPos) + "_PmaskRoi.mrcs");
	// M.write(fnImgOut.substr(0, dotPos) + "_PmaskRoi_Norm.mrcs");
	#endif

	double ll_I = 0;
	double ll_IsubP = 0;

	compareRegions(ll_I, ll_IsubP, fnImgOut);

	writeParticle(rowOut, ll_I, ll_IsubP, (ll_I-ll_IsubP)); 
}

void ProgClassifyPartialOccupancy::processParticle(const MDRow &rowprocess, int sizeImg) 
{	
	// Read metadata information for projection
	readParticle(rowprocess);
	rowprocess.getValueOrDefault(MDL_ANGLE_ROT, part_angles.rot, 0);
	rowprocess.getValueOrDefault(MDL_ANGLE_TILT, part_angles.tilt, 0);
	rowprocess.getValueOrDefault(MDL_ANGLE_PSI, part_angles.psi, 0);
	
	roffset.initZeros(2);
	rowprocess.getValueOrDefault(MDL_SHIFT_X, roffset(0), 0);
	rowprocess.getValueOrDefault(MDL_SHIFT_Y, roffset(1), 0);
	roffset *= -1;

	rowprocess.getValue(MDL_SUBTRACTION_B, adjustParams.b); 
	rowprocess.getValue(MDL_SUBTRACTION_BETA0, adjustParams.b0); 
	rowprocess.getValue(MDL_SUBTRACTION_BETA1, adjustParams.b1); 
	
	// Project volume + apply translation
	if (realSpaceProjector)
	{
		projectVolume(V(), P, sizeImg, sizeImg, part_angles.rot, part_angles.tilt, part_angles.psi, &roffset);
	}
	else
	{
		projectVolume(*projector, P, sizeImg, sizeImg, part_angles.rot, part_angles.tilt, part_angles.psi, ctfImage);
		selfTranslate(xmipp_transformation::LINEAR, P(), roffset, xmipp_transformation::WRAP);
	}
}

void ProgClassifyPartialOccupancy::finishProcessing()
{
	if (allow_time_bar && verbose && !single_image)
        progress_bar(time_bar_size);
    writeOutput();
}


// Core methods ===================================================================
void ProgClassifyPartialOccupancy::unitCellExtraction()
{
	UnitCell UC(uc_sym, uc_rmin, uc_rmax, uc_expandFactor, uc_offset, uc_sampling, uc_x_origin, uc_y_origin, uc_z_origin);
	// UC.maskUnitCell(V, V_unitcell);
}


void ProgClassifyPartialOccupancy::frequencyCharacterization()
{
	// Calculate FT
	MultidimArray< std::complex<double> > V_ft; // Volume FT
    MultidimArray<double> V_ft_mod; // Volume FT modulus

	FourierTransformer fcTransformer;
	fcTransformer.FourierTransform(V_unitcell(), V_ft, false);

	// FT dimensions
	int Xdim_ft = XSIZE(V_ft);
	int Ydim_ft = YSIZE(V_ft);
	int Zdim_ft = ZSIZE(V_ft);
	int Ndim_ft = NSIZE(V_ft);

	if (Zdim_ft == 1)
	{
		Zdim_ft = Ndim_ft;
	}

	int maxRadius = std::min(Xdim_ft, std::min(Ydim_ft, Zdim_ft));	// Restric analysis to Nyquist

	#ifdef DEBUG_FREQUENCY_PROFILE
	std::cout << "FFT map dimensions: " << std::endl;  
	std::cout << "FT Xdim " << Xdim_ft << std::endl;
	std::cout << "FT Ydim " << Ydim_ft << std::endl;
	std::cout << "FT zSize " << Zdim_ft << std::endl;
	std::cout << "FT nSize " << Ndim_ft << std::endl;
	std::cout << "maxRadius " << maxRadius << std::endl;
	#endif

	// Construct frequency map and initialize the frequency vectors
	MultidimArray<double> freqMap;
	Matrix1D<double> freq_fourier_x;
	Matrix1D<double> freq_fourier_y;
	Matrix1D<double> freq_fourier_z;

	freq_fourier_x.initZeros(Xdim_ft);
	freq_fourier_y.initZeros(Ydim_ft);
	freq_fourier_z.initZeros(Zdim_ft);

	double u;	// u is the frequency

	// Defining frequency components. First element should be 0, it is set as the smallest number to avoid singularities
	VEC_ELEM(freq_fourier_z,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<Zdim_ft; ++k){
		FFT_IDX2DIGFREQ(k,Zdim, u);
		VEC_ELEM(freq_fourier_z, k) = u;
	}

	VEC_ELEM(freq_fourier_y,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<Ydim_ft; ++k){
		FFT_IDX2DIGFREQ(k,Ydim, u);
		VEC_ELEM(freq_fourier_y, k) = u;
	}

	VEC_ELEM(freq_fourier_x,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<Xdim_ft; ++k){
		FFT_IDX2DIGFREQ(k,Xdim, u);
		VEC_ELEM(freq_fourier_x, k) = u;
	}

	//Initializing map with frequencies
	freqMap.resizeNoCopy(V_ft);

	// Directional frequencies along each direction
	double uz;
	double uy;
	double ux;
	double uz2;
	double uz2y2;
	long n=0;
	int idx = 0;

	for(size_t k=0; k<Zdim_ft; ++k)
	{
		uz = VEC_ELEM(freq_fourier_z, k);
		uz2 = uz*uz;
		
		for(size_t i=0; i<Ydim_ft; ++i)
		{
			uy = VEC_ELEM(freq_fourier_y, i);
			uz2y2 = uz2 + uy*uy;

			for(size_t j=0; j<Xdim_ft; ++j)
			{
				ux = VEC_ELEM(freq_fourier_x, j);
				ux = sqrt(uz2y2 + ux*ux);

				idx = (int) round(ux * Xdim);
				DIRECT_MULTIDIM_ELEM(freqMap,n) = idx;

				++n;
			}
		}
	}

	// Calculate radial average
	std::vector<double> radialCounter_FT;

	V_ft_mod.initZeros(Zdim_ft, Ydim_ft, Xdim_ft);
	radialAvgVolFT.resize(maxRadius, 0);
	radialCounter_FT.resize(maxRadius, 0);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V_ft)
	{
		double value_mod  = sqrt((DIRECT_MULTIDIM_ELEM(V_ft,n) * std::conj(DIRECT_MULTIDIM_ELEM(V_ft,n))).real());

		DIRECT_MULTIDIM_ELEM(V_ft_mod,n)  = value_mod;
		
		if(DIRECT_MULTIDIM_ELEM(freqMap,n) < maxRadius)
		{
			radialAvgVolFT[(int)(DIRECT_MULTIDIM_ELEM(freqMap,n))]  += value_mod;
			radialCounter_FT[(int)(DIRECT_MULTIDIM_ELEM(freqMap,n))] += 1;
		}
	}

	for(size_t i = 0; i < radialCounter_FT.size(); i++)
	{
		radialAvgVolFT[i] /= radialCounter_FT[i];
	}

	// Calculate minimum modulus for frequency 
	auto maxElement = std::max_element(radialAvgVolFT.begin(), radialAvgVolFT.end());
	thrModuleFT = 0.5*static_cast<double>(*maxElement);
	maxModuleFT = static_cast<double>(*maxElement);

	//Construct particle frequency map (2D)
	// Reuse freq_fourier_x and freq_fourier_y vectors

	//Initializing particle map with frequencies
	particleFreqMap.initZeros(Ydim_ft, Xdim_ft);

	// Directional frequencies along each direction
	double uy2;
	n=0;
	idx = 0;

	for(size_t i=0; i<Ydim_ft; ++i)
	{
		uy = VEC_ELEM(freq_fourier_y, i);
		uy2 = uy*uy;

		for(size_t j=0; j<Xdim_ft; ++j)
		{
			ux = VEC_ELEM(freq_fourier_x, j);
			ux = sqrt(uy2 + ux*ux);

			idx = (int) round(ux * Xdim);
			DIRECT_MULTIDIM_ELEM(particleFreqMap,n) = idx;

			++n;
		}
	}

	#ifdef DEBUG_OUTPUT_FILES
	// Save FT maps
	size_t lastindex = fn_out.find_last_of(".");
	std::string rawname = fn_out.substr(0, lastindex);
	Image<double> saveImage;
	
	std::string debugFileFn = rawname + "_freqMap.mrc";
	saveImage() = freqMap;
	saveImage.write(debugFileFn);

	debugFileFn = rawname + "_particleFreqMap.mrc";
	saveImage() = particleFreqMap;
	saveImage.write(debugFileFn);

	debugFileFn = rawname + "_FT_mod.mrc";
	saveImage() = V_ft_mod;
	saveImage.write(debugFileFn);

	// Save output metadata
	MetaDataVec md;
	size_t id;

	for(size_t i = 0; i < radialCounter_FT.size(); i++)
	{
		id = md.addObject();
		md.setValue(MDL_AVG, radialAvgVolFT[i],      id);
		md.setValue(MDL_X,   radialCounter_FT[i],  id);
	}

	std::string outputMD = rawname + "_FT_profile.xmd";
	md.write(outputMD);
	#endif

	std::cout << "Frenquency profiling estimated. Minimum modulus value: " << thrModuleFT << std::endl;
}

void ProgClassifyPartialOccupancy::noiseEstimation()
{
	auto t1 = std::chrono::high_resolution_clock::now();

	MetaData &mdIn = *getInputMd();

    srand(time(0)); // Seed for random number generation
    int maxX = Xdim - cropSize;
    int maxY = Ydim - cropSize;
    int minX = cropSize;
    int minY = cropSize;

	double scallignFactor = (Xdim * Ydim) / (cropSize * cropSize);

    bool invalidRegion;
	size_t processedParticles = 0;

    MultidimArray< double > noiseCrop;
	powerNoise().initZeros((int)Ydim, (int)Xdim/2 +1);

	// Iterate particles
	for (const auto& r : mdIn)
	{
		#ifdef DEBUG_NOISE_CALCULATION
		std::cout << "Estimating noise from particle " << processedParticles + 1 << std::endl;
		#endif

		r.getValueOrDefault(MDL_IMAGE, fnImgI, "no_filename");
		I.read(fnImgI);
		I().setXmippOrigin();

		r.getValueOrDefault(MDL_ANGLE_ROT, part_angles.rot, 0);
		r.getValueOrDefault(MDL_ANGLE_TILT, part_angles.tilt, 0);
		r.getValueOrDefault(MDL_ANGLE_PSI, part_angles.psi, 0);
		roffset.initZeros(2);
		r.getValueOrDefault(MDL_SHIFT_X, roffset(0), 0);
		r.getValueOrDefault(MDL_SHIFT_Y, roffset(1), 0);
		roffset *= -1;

		projectVolume(vMaskP(), PmaskProtein, Xdim, Ydim, part_angles.rot, part_angles.tilt, part_angles.psi, &roffset);
		projectVolume(vMaskRoi(), PmaskRoi, Xdim, Ydim, part_angles.rot, part_angles.tilt, part_angles.psi, &roffset);

		#ifdef DEBUG_OUTPUT_FILES
		size_t dotPos = fn_out.find_last_of('.');
		FileName outputPath = fn_out.substr(0, dotPos) + "_PmaskProtein.mrcs";
		outputPath.compose(processedParticles + 1, outputPath);
		PmaskProtein.write(outputPath);
		#endif

		// Optimize noise calulation: search for random regions that fall in the square that circunscribe the 
		// region that contain protein. We avoid generating random numbers in invalid regions.
		if(processedParticles < numberParticlesForBoundaryDetermination)
		{
			for (int i = 0; i < (int)Ydim; ++i) 
			{
				for (int j = 0; j < (int)Xdim; ++j) 
				{
					if (DIRECT_A2D_ELEM(PmaskProtein(), i, j) > 0) 
					{
						if (j < minX) minX = j;
						if (j > maxX) maxX = j;
						if (i < minY) minY = i;
						if (i > maxY) maxY = i;
					}
				}
			}
		}

		do {
			invalidRegion = false;
			noiseCrop.initZeros((int)Ydim, (int)Xdim);

			int x = minX + rand() % (maxX - minX + 1);
			int y = minY + rand() % (maxY - minY + 1);

			for (size_t i = 0; i < cropSize; i++)
			{
				for (size_t j = 0; j < cropSize; j++)
				{
					#ifdef DEBUG_NOISE_CALCULATION
					std::cout << "--------------------------------------------------" << std::endl;
					std::cout << "x  " << x << " y " << y  << std::endl;
					std::cout << "i " << i << " j  " << j  << std::endl;
					std::cout << "y + i  " << y + i << " x + j " << x + j << std::endl;
					std::cout << "(Ydim/2) " << (Ydim/2) << " (Xdim/2) " << (Xdim/2) << std::endl;
					std::cout << "(Ydim/2) - (cropSize/2) + i  " << (Ydim/2) - (cropSize/2) + i << " (Xdim/2) - (cropSize/2) + j " << (Xdim/2) - (cropSize/2) + j << std::endl;
					#endif

					if (DIRECT_A2D_ELEM(PmaskProtein(), y + i, x + j) == 0 || DIRECT_A2D_ELEM(PmaskRoi(), y + i, x + j) > 0)
					{
						invalidRegion = true;

						#ifdef DEBUG_NOISE_CALCULATION
					 	std::cout << "Invalid region. Trying again..." << std::endl;
						#endif
					
						break;
					}

					DIRECT_A2D_ELEM(noiseCrop,  (Ydim/2) - (cropSize/2) + i, (Xdim/2) - (cropSize/2) + j) = scallignFactor * DIRECT_A2D_ELEM(I(), y + i, x + j);
				}

				if (invalidRegion) {
					break;
				}
		}
		} while (invalidRegion);

		#ifdef DEBUG_NOISE_CALCULATION
		size_t lastindex = fn_out.find_last_of(".");
		std::string rawname = fn_out.substr(0, lastindex);

		Image<double> saveImage;
		std::string debugFileFn = rawname + "_noiseCrop.mrc";

		saveImage() = noiseCrop;
		saveImage.write(debugFileFn);
		#endif

	    FourierTransformer transformerNoise;
		transformerNoise.FourierTransform(noiseCrop, noiseSpectrum, false);

		#ifdef DEBUG_NOISE_CALCULATION
		MultidimArray< double > noiseSpectrumReal;
		noiseSpectrumReal.initZeros(YSIZE(noiseSpectrum), XSIZE(noiseSpectrum)); 

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(noiseSpectrum)
			DIRECT_MULTIDIM_ELEM(noiseSpectrumReal,n) = DIRECT_MULTIDIM_ELEM(noiseSpectrum,n).real();

		Image<double> saveImageHalf;
		debugFileFn = rawname + "_noiseSpectrumReal.mrc";

		saveImageHalf() = noiseSpectrumReal;
		saveImageHalf.write(debugFileFn);
		#endif

 		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(noiseSpectrum)
			DIRECT_MULTIDIM_ELEM(powerNoise(),n) += (DIRECT_MULTIDIM_ELEM(noiseSpectrum,n) * std::conj(DIRECT_MULTIDIM_ELEM(noiseSpectrum,n))).real();
			
		#ifdef DEBUG_NOISE_CALCULATION
		std::cout << "Noise estimated from particle " << processedParticles + 1 << " sucessfully." << std::endl;
		#endif

		processedParticles++;

		if (processedParticles == numParticlesNoiseEst)
		{
			break;
		}
	}

	powerNoise() /= processedParticles;


	#ifdef DEBUG_OUTPUT_FILES
	size_t lastindex = fn_out.find_last_of(".");
	std::string debugFileFn = fn_out.substr(0, lastindex) + "_noisePower.mrc";
	powerNoise.write(debugFileFn);
	#endif

	#ifdef VERBOSE_OUTPUT
	auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1);	// Getting number of seconds as an integer
	std::cout << "Execution time for noise estimation: " << ms_int.count() << " seconds." << std::endl;

	std::cout << "Number of particles processed for noise estimation: " << processedParticles << std::endl;
	#endif
}

void ProgClassifyPartialOccupancy::compareRegions(double &ll_I, double &ll_IsubP, const FileName &fnImgOut)
{	
	// -- Detect ligand regions
	binarizeMask(PmaskRoi);
	MultidimArray<double> PmaskRoiLabel;
	PmaskRoiLabel.resizeNoCopy(PmaskRoi());
	int numLig = labelImage2D(PmaskRoi(), PmaskRoiLabel, 8);

	#ifdef DEBUG_REGIONS_COMPARISON
	size_t dotPos = fnImgOut.find_last_of('.');
	Image<double> saveImage;

	saveImage = PmaskRoi;
	saveImage.write(fnImgOut.substr(0, dotPos) + "_PmaskRoiBinarize.mrcs");

	saveImage() = PmaskRoiLabel;
	saveImage.write(fnImgOut.substr(0, dotPos) + "_PmaskRoiLabel.mrcs");
	#endif

	// Calculate bounding box for each ligand region
	std::vector<int> minX(numLig, std::numeric_limits<int>::max());
	std::vector<int> minY(numLig, std::numeric_limits<int>::max());
	std::vector<int> maxX(numLig, 0);
	std::vector<int> maxY(numLig, 0);

	calculateBoundingBox(PmaskRoiLabel, minX, minY, maxX, maxY, numLig);

	// Calculate metric for each region
	MultidimArray<double> centeredLigand;
	MultidimArray<double> centeredLigandSubP;
	MultidimArray< std::complex<double> > fftI;
	MultidimArray< std::complex<double> > fftIsubP;

	// Subtract proyected weight volume to particle

	// Por ahora solo consideramos ajuste de orden 0. 
	// Si queremos considerar el del orden 1 hay que que comprobar que b1 > 0
	// y ajustar por frecuencia
	IsubP() = (I() - adjustParams.b) - (P() * adjustParams.b0);

	#ifdef DEBUG_REGIONS_COMPARISON
	saveImage = IsubP;
	saveImage.write(fnImgOut.substr(0, dotPos) + "_IsubP.mrcs");
	#endif

	// // Calcualte metrics for both images in Real space
	// double avg_I = 0;
	// double std_I = 0;
	// double zScore_I = 0;
	// double energy_I = 0;
	// double avg_IsubP = 0;
	// double std_IsubP = 0;
	// double zScore_IsubP = 0;
	// double energy_IsubP = 0;

	// computeParticleStats(I(), avg_I, std_I, zScore_I, energy_I);
	// computeParticleStats(IsubP(), avg_IsubP, std_IsubP, zScore_IsubP, energy_IsubP);
	// ll_I += energy_I;
	// ll_IsubP += energy_IsubP;

	// #ifdef DEBUG_METRICS
	// std::cout << "Final ll_I: " << ll_I << std::endl;
	// std::cout << "Final ll_IsubP: " << ll_IsubP << std::endl;
	// std::cout << "Final diff ll_I - ll_IsubP: " << ll_I - ll_IsubP << std::endl;
	// #endif	

	// return;

	std::cout << "--------------------------------------------------------------------- " 	<< std::endl;
	std::cout << fnImgI << std::endl;

	// Analyze each ligand region independently
	for (size_t value = 0; value < numLig; ++value) 
	{
		#ifdef DEBUG_REGIONS_COMPARISON
		std::cout << "Analyzign ligand region " << int(value +1) << std::endl;
		#endif

		// Cropping regions
		int width = maxX[value] - minX[value] + 1;
		int height = maxY[value] - minY[value] + 1;
		int centerX = Xdim / 2;
		int centerY = Ydim / 2;
		int numberOfPx = 0;

		#ifdef DEBUG_REGIONS_COMPARISON
		std::cout << "minX[value] " << minX[value] << std::endl;
		std::cout << "maxX[value] " << maxX[value] << std::endl;
		std::cout << "minY[value] " << minY[value] << std::endl;
		std::cout << "maxY[value] " << maxY[value] << std::endl;
		std::cout << "width " 		<< width << std::endl;
		std::cout << "height " 		<< height << std::endl;
		#endif

		// Initialize new images for cropping
		centeredLigand.initZeros(Ydim, Xdim);
		centeredLigandSubP.initZeros(Ydim, Xdim);

		// Copy the region to the center of the new image
		for (int i = minY[value]; i <= maxY[value]; ++i) 
		{
			for (int j = minX[value]; j <= maxX[value]; ++j) 
			{
				int newI = centerY - height / 2 + (i - minY[value]);
				int newJ = centerX - width / 2 + (j - minX[value]);
				
				if (DIRECT_A2D_ELEM(PmaskRoiLabel, i, j) > 0)
				{
					DIRECT_A2D_ELEM(centeredLigand, newI, newJ) = DIRECT_A2D_ELEM(I(), i, j);
					DIRECT_A2D_ELEM(centeredLigandSubP, newI, newJ) = DIRECT_A2D_ELEM(IsubP(), i, j);
					
					numberOfPx++;
				}
			}
		}

		#ifdef DEBUG_REGIONS_COMPARISON
		std::cout << "numberOfPx " 	<< numberOfPx << std::endl;
		#endif

		// // Fix scale previous to FT
		// double scallignFactor = (Xdim * Ydim) / (numberOfPx);

		// for (int i = minY[value]; i <= maxY[value]; ++i) 
		// {
		// 	for (int j = minX[value]; j <= maxX[value]; ++j) 
		// 	{
		// 		int newI = centerY - height / 2 + (i - minY[value]);
		// 		int newJ = centerX - width / 2 + (j - minX[value]);
				
		// 		if (DIRECT_A2D_ELEM(PmaskRoiLabel, i, j) > 0)
		// 		{
		// 			DIRECT_A2D_ELEM(centeredLigand, newI, newJ) *= scallignFactor;
		// 			DIRECT_A2D_ELEM(centeredLigandSubP, newI, newJ) *= scallignFactor;
		// 		}
		// 	}
		// }

		#ifdef DEBUG_REGIONS_COMPARISON
		size_t lastindex = fnImgOut.find_last_of(".");

		std::string debugFileFn = fnImgOut.substr(0, lastindex) + "_centeredLigand_" + std::to_string(value) + ".mrcs";
		saveImage() = centeredLigand;
		saveImage.write(debugFileFn);

		debugFileFn = fnImgOut.substr(0, lastindex) + "_centeredLigandSubP_" + std::to_string(value) + ".mrcs";
		saveImage() = centeredLigandSubP;
		saveImage.write(debugFileFn);
		#endif

		// Calcualte metrics for both regions in Fourier space
		transformerI.FourierTransform(centeredLigand, fftI, false);
		transformerIsubP.FourierTransform(centeredLigandSubP, fftIsubP, false);
		
		double ll_I_it = 0;
		double ll_IsubP_it = 0;

		// Calcualte metrics for both regions
		logLikelihood(ll_I_it, ll_IsubP_it, fftI, fftIsubP);
		// radialLogLikelihood(ll_I_it, ll_IsubP_it, fftI, fftIsubP);
		// entropy(ll_I_it, ll_IsubP_it, fftI, fftIsubP);
		// kullbackLeibler(ll_I_it, ll_IsubP_it, fftI, fftIsubP);
		// crossEntropy(ll_I_it, ll_IsubP_it, fftI, fftIsubP);

		// Accumulate for all regions
		ll_I	 += ll_I_it;
		ll_IsubP += ll_IsubP_it;
	}

	#ifdef DEBUG_METRICS
	std::cout << "Final ll_I: " << ll_I << std::endl;
	std::cout << "Final ll_IsubP: " << ll_IsubP << std::endl;
	std::cout << "Final diff ll_I - ll_IsubP: " << ll_I - ll_IsubP << std::endl;
	#endif
}

void ProgClassifyPartialOccupancy::logLikelihood(double &ll_I_it, 
												 double &ll_IsubP_it, 
												 MultidimArray<std::complex<double>> fftI,
												 MultidimArray<std::complex<double>> fftIsubP)
{
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fftI)
	{
		// Consider only those frequencies (under Nyquist) whose radial module is over threshold
		// if (radialAvgVolFT[DIRECT_MULTIDIM_ELEM(particleFreqMap,n)] > thrModuleFT && DIRECT_MULTIDIM_ELEM(particleFreqMap,n) / Xdim <= 0.5)
		// {
		
		// Consider all frequencies
		if (DIRECT_MULTIDIM_ELEM(particleFreqMap,n) / Xdim <= 0.5)
		{

		// Consider only "mount Fuji" frequencies (in Halo but not in APO)
		// if (DIRECT_MULTIDIM_ELEM(particleFreqMap,n) > 50 && DIRECT_MULTIDIM_ELEM(particleFreqMap,n) < 150)
		// {
			
			ll_I_it     += (DIRECT_MULTIDIM_ELEM(fftI,n)     * std::conj(DIRECT_MULTIDIM_ELEM(fftI,n))).real()     / (DIRECT_MULTIDIM_ELEM(powerNoise(), n));
			ll_IsubP_it += (DIRECT_MULTIDIM_ELEM(fftIsubP,n) * std::conj(DIRECT_MULTIDIM_ELEM(fftIsubP,n))).real() / (DIRECT_MULTIDIM_ELEM(powerNoise(), n));

			// Weight by frquency magnitude (normalized with the maximum)
			// double freqNormFactor = radialAvgVolFT[DIRECT_MULTIDIM_ELEM(particleFreqMap,n)] / maxModuleFT;
			// ll_I_it		*= freqNormFactor;
			// ll_IsubP_it	*= freqNormFactor;
		}
	}
}

void ProgClassifyPartialOccupancy::radialLogLikelihood(double &ll_I_it, 
												       double &ll_IsubP_it, 
												 	   MultidimArray<std::complex<double>> fftI,
												 	   MultidimArray<std::complex<double>> fftIsubP)
{
	// Take radial average for each FT
	std::vector<double> fftI_RA;
	std::vector<double> fftIsubP_RA;
	std::vector<double> powerNoise_RA;

	fftI_RA.resize((Xdim/2) + 1, 0);	// assume squared paticles 
	fftIsubP_RA.resize((Xdim/2) + 1, 0);
	powerNoise_RA.resize((Xdim/2) + 1, 0);

	calculateRadialAverage(fftI, fftI_RA, false);
	calculateRadialAverage(fftIsubP, fftIsubP_RA, false);
	calculatePowerNoiseRadialAverage(powerNoise_RA, false);

	for(size_t n = 0; n < fftI_RA.size(); n++)
	{		
		// Consider all frequencies
		if (true)
		{

		// Consider only "mount Fuji" frequencies (in Halo but not in APO)
		// if (n > 50 && n < 150)
		// {
			// std::cout << "fftI_RA[" << n << "]" << fftI_RA[n] << "        std::log2(fftI_RA[" << n << "])" << std::log2(fftI_RA[n]) << std::endl;
			ll_I_it     += fftI_RA[n]     / powerNoise_RA[n];
			ll_IsubP_it += fftIsubP_RA[n] / powerNoise_RA[n];
		}
	}
}

void ProgClassifyPartialOccupancy::entropy(double &entropy_I_it, 
										   double &entropy_IsubP_it, 
										   MultidimArray<std::complex<double>> fftI,
										   MultidimArray<std::complex<double>> fftIsubP)
{	
	// Take radial average for each FT
	std::vector<double> fftI_RA;
	std::vector<double> fftIsubP_RA;

	fftI_RA.resize((Xdim/2) + 1, 0);	// assume squared paticles 
	fftIsubP_RA.resize((Xdim/2) + 1, 0);

	calculateRadialAverage(fftI, fftI_RA, false);
	calculateRadialAverage(fftIsubP, fftIsubP_RA, false);

	for(size_t n = 0; n < fftI_RA.size(); n++)
	{		
		// Consider all frequencies
		if (true)
		{

		// Consider only "mount Fuji" frequencies (in Halo but not in APO)
		// if (n > 50 && n < 150)
		// {
			// std::cout << "fftI_RA[" << n << "]" << fftI_RA[n] << "        std::log2(fftI_RA[" << n << "])" << std::log2(fftI_RA[n]) << std::endl;
			entropy_I_it     -= fftI_RA[n]     * std::log2(fftI_RA[n]);
			entropy_IsubP_it -= fftIsubP_RA[n] * std::log2(fftIsubP_RA[n]);
		}
	}
}

void ProgClassifyPartialOccupancy::kullbackLeibler(double &entropy_I_it, 
										           double &entropy_IsubP_it, 
										           MultidimArray<std::complex<double>> fftI,
										           MultidimArray<std::complex<double>> fftIsubP)
{	
	// Take radial average for each FT
	std::vector<double> fftI_RA;
	std::vector<double> fftIsubP_RA;
	std::vector<double> powerNoise_RA;

	fftI_RA.resize((Xdim/2) + 1, 0);	// assume squared paticles 
	fftIsubP_RA.resize((Xdim/2) + 1, 0);
	powerNoise_RA.resize((Xdim/2) + 1, 0);

	calculateRadialAverage(fftI, fftI_RA, false);
	calculateRadialAverage(fftIsubP, fftIsubP_RA, false);
	calculatePowerNoiseRadialAverage(powerNoise_RA, false);

	for(size_t n = 0; n < fftI_RA.size(); n++)
	{		
		// Consider all frequencies
		if (true)
		{

		// Consider only "mount Fuji" frequencies (in Halo but not in APO)
		// if (n > 50 && n < 150)
		// {
			entropy_I_it     += fftI_RA[n]     * std::log2(fftI_RA[n]    /powerNoise_RA[n]);
			entropy_IsubP_it += fftIsubP_RA[n] * std::log2(fftIsubP_RA[n]/powerNoise_RA[n]); 
		}
	}

	entropy_I_it     = -entropy_I_it;
	entropy_IsubP_it = -entropy_IsubP_it;
}

void ProgClassifyPartialOccupancy::crossEntropy(double &entropy_I_it, 
										   double &entropy_IsubP_it, 
										   MultidimArray<std::complex<double>> fftI,
										   MultidimArray<std::complex<double>> fftIsubP)
{	
	// Take radial average for each FT
	std::vector<double> fftI_RA;
	std::vector<double> fftIsubP_RA;
	std::vector<double> powerNoise_RA;

	fftI_RA.resize((Xdim/2) + 1, 0);	// assume squared paticles 
	fftIsubP_RA.resize((Xdim/2) + 1, 0);
	powerNoise_RA.resize((Xdim/2) + 1, 0);

	calculateRadialAverage(fftI, fftI_RA, false);
	calculateRadialAverage(fftIsubP, fftIsubP_RA, false);
	calculatePowerNoiseRadialAverage(powerNoise_RA, false);

	for(size_t n = 0; n < fftI_RA.size(); n++)
	{		
		// Consider all frequencies
		if (true)
		{

		// Consider only "mount Fuji" frequencies (in Halo but not in APO)
		// if (n > 50 && n < 150)
		// {
			entropy_I_it     += fftI_RA[n]     * std::log2(powerNoise_RA[n]);
			entropy_IsubP_it += fftIsubP_RA[n] * std::log2(powerNoise_RA[n]); 
		}
	}

	entropy_I_it     = -entropy_I_it;
	entropy_IsubP_it = -entropy_IsubP_it;
}

// Utils methods ===================================================================
Image<double> ProgClassifyPartialOccupancy::binarizeMask(Projection &m) const 
{
	MultidimArray<double> &mm=m();

 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mm)
		DIRECT_MULTIDIM_ELEM(mm,n) = (DIRECT_MULTIDIM_ELEM(mm,n)>0) ? 1:0; 
 	return m;
}

Image<double> ProgClassifyPartialOccupancy::invertMask(const Image<double> &m) 
{
	PmaskI = m;
	MultidimArray<double> &mPmaskI=PmaskI();
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mPmaskI)
		DIRECT_MULTIDIM_ELEM(mPmaskI,n) = (DIRECT_MULTIDIM_ELEM(mPmaskI,n)*(-1))+1;
	return PmaskI;
}

void ProgClassifyPartialOccupancy::calculateBoundingBox(MultidimArray<double> PmaskRoiLabel, 
														std::vector<int> &minX, 
														std::vector<int> &minY, 
														std::vector<int> &maxX, 
														std::vector<int> &maxY, 
														int numLig)
{	
	for (size_t i = 0; i < Ydim; ++i) 
	{
		for (size_t j = 0; j < Xdim; ++j) 
		{
			int value = int(DIRECT_A2D_ELEM(PmaskRoiLabel, i, j));
			if (value != 0) 
			{
				if (j < minX[value - 1]) minX[value - 1] = j;
				if (j > maxX[value - 1]) maxX[value - 1] = j;
				if (i < minY[value - 1]) minY[value - 1] = i;
				if (i > maxY[value - 1]) maxY[value - 1] = i;
			}
		}
	}

	// Adjust bounding boxes to be squares
    for (int k = 0; k < numLig; ++k) 
	{
        int width = maxX[k] - minX[k] + 1;
        int height = maxY[k] - minY[k] + 1;
        int maxDim = std::max(width, height);

        int centerX = (minX[k] + maxX[k]) / 2;
        int centerY = (minY[k] + maxY[k]) / 2;

        minX[k] = centerX - maxDim / 2;
        maxX[k] = centerX + maxDim / 2;
        minY[k] = centerY - maxDim / 2;
        maxY[k] = centerY + maxDim / 2;

        // Ensure the bounding box stays within the image boundaries
        minX[k] = std::max(0, minX[k]);
        maxX[k] = std::min(static_cast<int>(Xdim - 1), maxX[k]);
        minY[k] = std::max(0, minY[k]);
        maxY[k] = std::min(static_cast<int>(Ydim - 1), maxY[k]);
    }
}


void ProgClassifyPartialOccupancy::calculateRadialAverage(const MultidimArray<std::complex<double>> &particleFT, 
														  std::vector<double> &radialAvgFT,
														  bool normalize) 
{
	std::vector<double> radialCounter_FT;
	radialCounter_FT.resize(radialAvgFT.size(), 0);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(particleFreqMap)
	{
		// Limit analysis to Nyquist
		if(DIRECT_MULTIDIM_ELEM(particleFreqMap,n) < (Xdim/2) + 1)
		{
			double value_mod  = (DIRECT_MULTIDIM_ELEM(particleFT,n) * std::conj(DIRECT_MULTIDIM_ELEM(particleFT,n))).real();		

			radialAvgFT[(int)(DIRECT_MULTIDIM_ELEM(particleFreqMap,n))]  += value_mod;
			radialCounter_FT[(int)(DIRECT_MULTIDIM_ELEM(particleFreqMap,n))] += 1;
		}
	}

	for(size_t i = 0; i <  radialCounter_FT.size(); i++)
	{
		radialAvgFT[i] /= radialCounter_FT[i];
	}

	// This ensure the elements of the radial average sum 1
	// usefull to entropy and Kullback-Leibler 
	if (normalize)
	{
		double sumPower = 0;

		for(size_t i = 0; i < radialAvgFT.size(); i++)
		{
			sumPower += radialAvgFT[i];
		}	

		for(size_t i = 0; i < radialAvgFT.size(); i++)
		{
			radialAvgFT[i] /= sumPower;
		}
	}

	#ifdef DEBUG_RADIAL_AVERAGE	
	for(size_t i = 0; i < radialAvgFT.size(); i++)
	{
		std::cout <<"radialAvgFT["<<i<< "]=" << radialAvgFT[i]<< std::endl;
	}
	#endif
}

void ProgClassifyPartialOccupancy::calculatePowerNoiseRadialAverage(std::vector<double> &radialAvgFT,
																	bool normalize) 
{
	std::vector<double> radialCounter_FT;
	radialCounter_FT.resize(radialAvgFT.size(), 0);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(powerNoise())
	{
		// Limit analysis to Nyquist
		if(DIRECT_MULTIDIM_ELEM(particleFreqMap,n) < (Xdim/2) + 1)
		{
			radialAvgFT[(int)(DIRECT_MULTIDIM_ELEM(particleFreqMap,n))]      += DIRECT_MULTIDIM_ELEM(powerNoise(),n);
			radialCounter_FT[(int)(DIRECT_MULTIDIM_ELEM(particleFreqMap,n))] += 1;
		}
	}

	for(size_t i = 0; i <  radialCounter_FT.size(); i++)
	{
		radialAvgFT[i] /= radialCounter_FT[i];
	}

	// This ensure the elements of the radial average sum 1
	// usefull to entropy and Kullback-Leibler 
	if (normalize)
	{
		double sumPower = 0;

		for(size_t i = 0; i < radialAvgFT.size(); i++)
		{
			sumPower += radialAvgFT[i];
		}	

		for(size_t i = 0; i < radialAvgFT.size(); i++)
		{
			radialAvgFT[i] /= sumPower;
		}
	}

	#ifdef DEBUG_RADIAL_AVERAGE	
	for(size_t i = 0; i < radialAvgFT.size(); i++)
	{
		std::cout <<"radialAvgFT["<<i<< "]=" << radialAvgFT[i]<< std::endl;
	}
	#endif
}

// Class methods =======================================================
ProgClassifyPartialOccupancy::ProgClassifyPartialOccupancy()
{
	produces_a_metadata = true;
	produces_an_output = true;
	each_image_produces_an_output = true;
	projector = nullptr;
	rank = 0;
}

ProgClassifyPartialOccupancy::~ProgClassifyPartialOccupancy()
{
	delete projector;
}


// Unused methods ===================================================================
void ProgClassifyPartialOccupancy::computeParticleStats(MultidimArray<double> &mI,
														double &avg, 
														double &std, 
														double &zScore, 
														double &energy)
{	
	double sum = 0;
	double sum2 = 0;
	int Nelems = 0;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PmaskRoi())
	{
		if (DIRECT_MULTIDIM_ELEM(PmaskRoi(),n) > 0)
		{
			double value = DIRECT_MULTIDIM_ELEM(mI, n);

			sum += value;
			sum2 += value*value;
			++Nelems;
		}
	}

	energy =sum2;
	avg = sum / Nelems;
	std = sqrt(sum2/Nelems - avg*avg);
	int zScoreThr = 3;
	zScore = 0;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PmaskRoi())
	{
		if (DIRECT_MULTIDIM_ELEM(PmaskRoi(),n) > 0)
		{
			double value = DIRECT_MULTIDIM_ELEM(mI, n);

			zScore += (value - avg) / std;
			// if(value > (avg + std * zScoreThr))
			// {
			// 	zScore++;
			// }
		}
	}

	zScore /= Nelems;
	
	#ifdef DEBUG_COMPUTE_STATISTICS
	std::cout << "--------------------------------------------------" << std::endl;
	std::cout << "sum " << sum << std::endl;
	std::cout << "sum2 " << sum2 << std::endl;
	std::cout << "Nelems " << Nelems << std::endl;
	std::cout << "avg " << avg << std::endl;
	std::cout << "std " << std << std::endl;
	std::cout << "zScore " << zScore << std::endl;
	std::cout << "energy " << energy << std::endl;
	#endif
}
