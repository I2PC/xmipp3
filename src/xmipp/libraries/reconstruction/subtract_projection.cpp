/***************************************************************************
 *
 * Authors:     Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
 * 				Federico P. de Isidro-Gomez (federico.pdeisidro@astx.com)
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

 #include "subtract_projection.h"
 #include "core/transformations.h"
 #include "core/metadata_extension.h"
 #include "core/multidim_array.h"
 #include "core/xmipp_image_extension.h"
 #include "core/xmipp_image_generic.h"
 #include "core/xmipp_image_base.h"
 #include "core/xmipp_fft.h"
 #include "core/xmipp_fftw.h"
 #include "core/linear_system_helper.h"
 #include "data/projection.h"
 #include "data/mask.h"
 #include "data/filters.h"
 #include "data/morphology.h"
 #include <iostream>
 #include <string>
 #include <sstream>
 #include "data/image_operate.h"
 #include <iostream>
 #include <cstdlib>
 #include <vector>
 #include <utility>


 // Empty constructor =======================================================
ProgSubtractProjection::ProgSubtractProjection()
{
	projector = nullptr;
	rank = 0;
}

ProgSubtractProjection::~ProgSubtractProjection()
{
	delete projector;
}

 // Read arguments ==========================================================
 void ProgSubtractProjection::readParams()
 {
	XmippMetadataProgram::readParams();
 	fnVolR = getParam("--ref");
	fnMaskRoi=getParam("--mask_roi");
	sigma=getIntParam("--sigma");
	sampling = getDoubleParam("--sampling");
	padFourier = getDoubleParam("--padding");
    maxResol = getDoubleParam("--max_resolution");

	if (checkParam("--cirmaskrad"))
	{
		cirmaskrad = getDoubleParam("--cirmaskrad");
		maskVolProvided = false;
	}
	else if (checkParam("--mask"))
	{
		fnMaskVol = getParam("--mask");
		maskVolProvided = true;
	}
	
	fnProj = getParam("--save"); 
	nonNegative = checkParam("--nonNegative");
	boost = checkParam("--boost");
	subtract = checkParam("--subtract");
	realSpaceProjector = checkParam("--realSpaceProjection");
	ignoreCTF = checkParam("--ignoreCTF");

	noiseEstimationBool = checkParam("--noise_est");
 }

 // Show ====================================================================
 void ProgSubtractProjection::show() const
 {
    if (!verbose)
        return;
	std::cout
	<< "Input particles:\t" << fnParticles << std::endl
	<< "Reference volume:\t" << fnVolR << std::endl
	<< "Mask of the region of interest to keep or subtract:\t" << fnMaskRoi << std::endl
	<< "Sigma of low pass filter:\t" << sigma << std::endl
	<< "Sampling rate:\t" << sampling << std::endl
	<< "Padding factor:\t" << padFourier << std::endl
    << "Max. Resolution:\t" << maxResol << std::endl;

	if (noiseEstimationBool)
	{
		std::cout << "Computing noise estimation " << std::endl;
	}
 }

 // Usage ===================================================================
 void ProgSubtractProjection::defineParams()
 {
	// Labels
	each_image_produces_an_output = true;

	//Usage
    addUsageLine("This program computes the subtraction between particles and a reference"); 
	addUsageLine(" volume, by computing its projections with the same angles that input particles have."); 
	addUsageLine(" Then, each particle and the correspondent projection of the reference volume are numerically");
	addUsageLine(" adjusted and subtracted using a mask which denotes the region of interest to keep or subtract.");

    //Parameters
	XmippMetadataProgram::defineParams();
    addParamsLine("--ref <volume>\t: Reference volume to subtract");
    addParamsLine("[--mask_roi <mask_roi=\"\">]		: 3D mask for region of interest to keep or subtract, no mask implies subtraction of whole images");
 	addParamsLine("--cirmaskrad <c=-1.0>			: Apply circular mask to proyected particles. Radius = -1 fits a sphere in the reference volume.");
	addParamsLine("or --mask <mask=\"\">            : Provide a mask to be applied. Any desity out of the mask is removed from further analysis.");
	addParamsLine("[--sampling <sampling=1>]		: Sampling rate (A/pixel)");
	addParamsLine("[--max_resolution <f=-1>]		: Maximum resolution in Angtroms up to which the substraction is calculated. \
													  By default (-1) it is set to sampling/sqrt(2).");
	addParamsLine("[--padding <p=2>]				: Padding factor for Fourier projector");
	addParamsLine("[--sigma <s=1>]					: Decay of the filter (sigma) to smooth the mask transition");
	addParamsLine("[--nonNegative]					: Ignore particles with negative beta0 or R2");
	addParamsLine("[--boost]						: Perform a boosting of original particles");
	addParamsLine("[--save <structure=\"\">]		: Path for saving intermediate files");
	addParamsLine("[--subtract]						: The mask contains the region to SUBTRACT");
	addParamsLine("[--realSpaceProjection]			: Project volume in real space to avoid Fourier artifacts");
	addParamsLine("[--ignoreCTF]					: Do not consider CTF in the subtraction. Use if particles have been CTF corrected.");
	addParamsLine("[--noise_est]					: Compute noise estimation from the subtracted regin of the particles. \
													  This caluclation do not modifies the subtration, just produces a noise estimation.");

	// Example
    addExampleLine("A typical use is:",false);
    addExampleLine("xmipp_subtract_projection -i input_particles.xmd --ref input_map.mrc --mask_roi mask_vol.mrc "
				   "-o output_particles --sampling 1 --max_resolution 4");
 }

 // I/O methods ===================================================================
 void ProgSubtractProjection::readParticle(const MDRow &r) 
 {
	r.getValueOrDefault(MDL_IMAGE, fnImgI, "no_filename");
	I.read(fnImgI);
	I().setXmippOrigin();
 }

 void ProgSubtractProjection::writeParticle(MDRow &rowOut, FileName fnImgOut, Image<double> &img, double R2a, double b0save, double b1save, double b) 
 {
	img.write(fnImgOut);

	rowOut.setValue(MDL_IMAGE, fnImgOut);
	rowOut.setValue(MDL_SUBTRACTION_R2, R2a); 
	rowOut.setValue(MDL_SUBTRACTION_BETA0, b0save); 
	rowOut.setValue(MDL_SUBTRACTION_BETA1, b1save); 
	rowOut.setValue(MDL_SUBTRACTION_B, b); 
	if (nonNegative && (disable || R2a < 0))
	{
		rowOut.setValue(MDL_ENABLED, -1);
	}
 }

 // Utils methods ===================================================================
 void ProgSubtractProjection::createMask(const FileName &fnM, Image<double> &m, Image<double> &im) 
 {
	if (fnM.isEmpty()) 
	{
		m().initZeros((int)XSIZE(V()),(int)YSIZE(V()),(int)ZSIZE(V()));
		im = m;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(im())
			DIRECT_MULTIDIM_ELEM(im(),n) += 1; 
	}
	else 
	{
		m.read(fnM);
		m().setXmippOrigin();
		im = m;
		if (!subtract)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(im())
				DIRECT_MULTIDIM_ELEM(im(),n) = (DIRECT_MULTIDIM_ELEM(m(),n)*(-1))+1;
		}
	}
 }

 Image<double> ProgSubtractProjection::binarizeMask(Projection &m) const 
 {
	MultidimArray<double> &mm=m();

 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mm)
		DIRECT_MULTIDIM_ELEM(mm,n) = (DIRECT_MULTIDIM_ELEM(mm,n)>0) ? 1:0; 
 	return m;
 }

 Image<double> ProgSubtractProjection::invertMask(const Image<double> &m) 
 {
	PmaskI = m;
	MultidimArray<double> &mPmaskI=PmaskI();
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mPmaskI)
		DIRECT_MULTIDIM_ELEM(mPmaskI,n) = (DIRECT_MULTIDIM_ELEM(mPmaskI,n)*(-1))+1;
	return PmaskI;
 }

 Image<double> ProgSubtractProjection::applyCTF(const MDRow &r, Projection &proj) 
 {
	if (r.containsLabel(MDL_CTF_DEFOCUSU) || r.containsLabel(MDL_CTF_MODEL)){
		ctf.readFromMdRow(r);
		ctf.Tm = sampling;
		ctf.produceSideInfo();
	 	FilterCTF.FilterBand = CTF;
	 	FilterCTF.ctf.enable_CTFnoise = false;
		FilterCTF.ctf = ctf;

		// Padding before apply CTF
		MultidimArray <double> &mpad = padp();
		mpad.setXmippOrigin();
		MultidimArray<double> &mproj = proj();
		mproj.setXmippOrigin();
		mproj.window(mpad,STARTINGY(mproj)*(int)padFourier, STARTINGX(mproj)*(int)padFourier, FINISHINGY(mproj)*(int)padFourier, FINISHINGX(mproj)*(int)padFourier);
		FilterCTF.generateMask(mpad);
		FilterCTF.applyMaskSpace(mpad);

		//Crop to restore original size
		mpad.window(mproj,STARTINGY(mproj), STARTINGX(mproj), FINISHINGY(mproj), FINISHINGX(mproj));
	}
	return proj;
 }

void ProgSubtractProjection::processParticle(const MDRow &rowprocess, int sizeImg) 
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
	
	// Project volume + apply translation, CTF and mask
	// If provided, mask already have been applied to volume
	if (realSpaceProjector)
	{
		projectVolume(V(), P, sizeImg, sizeImg, part_angles.rot, part_angles.tilt, part_angles.psi, &roffset);
	}
	else
	{
		projectVolume(*projector, P, sizeImg, sizeImg, part_angles.rot, part_angles.tilt, part_angles.psi, ctfImage);
		selfTranslate(xmipp_transformation::LINEAR, P(), roffset, xmipp_transformation::WRAP);
	}
	
	if (ignoreCTF)
	{
		Pctf = P;
	}
	else
	{
		Pctf = applyCTF(rowprocess, P);
	}	

	MultidimArray<double> &mPctf = Pctf();
	MultidimArray<double> &mI = I();

	if(maskVolProvided)
	{
		projectVolume(maskVol(), Pmask, sizeImg, sizeImg, part_angles.rot, part_angles.tilt, part_angles.psi, &roffset);
		PmaskImg() = Pmask();
	}
	else
	{
		Pmask() = maskVol();
		PmaskImg() = maskVol();
	}

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PmaskImg())
	{
		if (!(DIRECT_MULTIDIM_ELEM(PmaskImg(),n) > 0))
		{
			DIRECT_MULTIDIM_ELEM(mPctf,n) = 0;
			DIRECT_MULTIDIM_ELEM(mI,n) = 0;
		}
	}

	// FT of projection and particle
	transformerP.FourierTransform(Pctf(), PFourier, false);
	transformerI.FourierTransform(I(), IFourier, false);
}

MultidimArray< std::complex<double> > ProgSubtractProjection::computeEstimationImage(const MultidimArray<double> &Img, 
const MultidimArray<double> *InvM, FourierTransformer &transformerImgiM) 
{
	ImgiM().initZeros(Img);
	ImgiM().setXmippOrigin();

	if(InvM)
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Img)
			DIRECT_MULTIDIM_ELEM(ImgiM(),n) = DIRECT_MULTIDIM_ELEM(Img,n) * DIRECT_MULTIDIM_ELEM(*(InvM),n);
	}
	else
	{
		ImgiM = Img;
	}

	transformerImgiM.FourierTransform(ImgiM(),ImgiMFourier,false);
	return ImgiMFourier;
}

 double ProgSubtractProjection::evaluateFitting(const MultidimArray< std::complex<double> > &y, const MultidimArray< std::complex<double> > &yp) const
{
	double sumY = 0;
	double sumY2 = 0;
	double sumE2 = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(y) {
		double realyn = real(DIRECT_MULTIDIM_ELEM(y, n)); 
		double imagyn = imag(DIRECT_MULTIDIM_ELEM(y, n)); 
		double ereal = realyn - real(DIRECT_MULTIDIM_ELEM(yp, n)); 
		double eimag = imagyn - imag(DIRECT_MULTIDIM_ELEM(yp, n));  
		sumE2 += ereal*ereal + eimag*eimag; 
		sumY += realyn + imagyn; 
		sumY2 += realyn*realyn + imagyn*imagyn; 
	}
	auto meanY = sumY / (2.0*(double)MULTIDIM_SIZE(y));
	auto varY = sumY2 / (2.0*(double)MULTIDIM_SIZE(y)) - meanY*meanY; 
	auto R2 = 1.0 - (sumE2/(2.0*(double)MULTIDIM_SIZE(y))) / varY;
	return R2;
 }

Matrix1D<double> ProgSubtractProjection::checkBestModel(MultidimArray< std::complex<double> > &PFourierf, const MultidimArray< std::complex<double> > &PFourierf0,
 const MultidimArray< std::complex<double> > &PFourierf1, const MultidimArray< std::complex<double> > &IFourierf) const 
 { 
	// Compute R2 coefficient for order 0 model (R20) and order 1 model (R21)
	auto N = 2.0*(double)MULTIDIM_SIZE(PFourierf);
	double R20adj = evaluateFitting(IFourierf, PFourierf0); // adjusted R2 for an order 0 model = R2
	double R21 = evaluateFitting(IFourierf, PFourierf1); 
	double R21adj = 1.0 - (1.0 - R21) * (N - 1.0) / (N - 2.0); // adjusted R2 for an order 1 model -> p = 2
	//Decide best fitting
	Matrix1D<double> R2(2);
	if (R21adj > R20adj) { // Order 1: T(w) = b01 + b1*wi 
		PFourierf = PFourierf1;
		R2(0) = R21adj;
		R2(1) = 1;
	} 
	else { // Order 0: T(w) = b00 
		PFourierf = PFourierf0;
		R2(0) = R20adj;
		R2(1) = 0;		
	}
	return R2;
}

void ProgSubtractProjection::computeParticleStats(Image<double> &Idiff, Image<double> &iM, FileName fnImgOut, double &avg, double &std, double &zScore)
{
	const auto sizeI = (int)XSIZE(I());
	MultidimArray<double> &mIdiff=Idiff();

	#ifdef DEBUG_OUTPUT_FILES
	MultidimArray<double> maskedIdiff;
	maskedIdiff.initZeros(mIdiff);
	#endif

	double sum = 0;
	double sum2 = 0;
	int Nelems = 0;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(M())
	{
		if (DIRECT_MULTIDIM_ELEM(M(),n) > 0)
		{
			double value = DIRECT_MULTIDIM_ELEM(mIdiff, n);

			sum += value;
			sum2 += value*value;
			++Nelems;
		}
	}

	avg = sum / Nelems;
	std = sqrt(sum2/Nelems - avg*avg);
	int zScoreThr = 3;
	zScore = 0;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(M())
	{
		if (DIRECT_MULTIDIM_ELEM(M(),n) > 0)
		{
			double value = DIRECT_MULTIDIM_ELEM(mIdiff, n);

			zScore += (value - avg) / std;
			// if(value > (avg + std * zScoreThr))
			// {
			// 	zScore++;
			// }
		}
	}

	zScore /= Nelems;
	
	#ifdef DEBUG
	std::cout << "sum " << sum << std::endl;
	std::cout << "sum2 " << sum2 << std::endl;
	std::cout << "Nelems " << Nelems << std::endl;
	std::cout << "avg " << avg << std::endl;
	std::cout << "std " << std << std::endl;
	std::cout << "zScore " << zScore << std::endl;
	#endif

	#ifdef DEBUG_OUTPUT_FILES
	// Save output masked particle for debugging
	FileName fnMaskedImgOut;
	size_t dotPos = fnImgOut.find_last_of('.');

    if (dotPos == std::string::npos) {
        // No extension found
        fnMaskedImgOut = fnImgOut + "_masked";
    }
    fnMaskedImgOut = fnImgOut.substr(0, dotPos) + "_masked" + fnImgOut.substr(dotPos);

	Image<double> saveImage;
	saveImage() = maskedIdiff;
	saveImage.write(fnMaskedImgOut);

	M.write(fnImgOut.substr(0, dotPos) + "_mask" + fnImgOut.substr(dotPos));
	#endif
}
	
void ProgSubtractProjection::generateNoiseEstimationSideInfo()
{
	// Initialize powerNoise
	powerNoise.initZeros((int)Ydim, (int)Xdim/2 +1);

	// Calculate boundaries for noise estimation 
	// (make a more eficinet sampling of the subtracted region)
	if(maskVolProvided)
	{
		int minX = Xdim;
		int minY = Ydim;
		int minZ = Zdim;
		int maxX = 0;
		int maxY = 0;
		int maxZ = 0;

		long n = 0;

		for(size_t k=0; k<Zdim; ++k)
		{
			for(size_t i=0; i<Ydim; ++i)
			{
				for(size_t j=0; j<Xdim; ++j)
				{
					if (DIRECT_MULTIDIM_ELEM(maskVol(), n) > 0) {
						minX = std::min(minX, (int)i);
						minY = std::min(minY, (int)j);
						minZ = std::min(minZ, (int)k);
						maxX = std::max(maxX, (int)i);
						maxY = std::max(maxY, (int)j);
						maxZ = std::max(maxZ, (int)k);
					}

					++n;
				}
			}
		}

		min_noiseEst = std::min(minX, std::min(minY, minZ));
		max_noiseEst = std::max(maxX, std::max(maxY, maxZ));
	}
	else
	{
		// Assuming square particles
		max_noiseEst = int((double)Xdim/2 + cirmaskrad/2);
		min_noiseEst = int((double)Xdim/2 - cirmaskrad/2);
	}

	#ifdef DEBUG_NOISE_ESTIMATION
	std::cout << "max_noiseEst  " << max_noiseEst << " min_noiseEst " << min_noiseEst  << std::endl;
	#endif
}

void ProgSubtractProjection::noiseEstimation()
{
	#ifdef DEBUG_NOISE_ESTIMATION
	std::cout << "Estimating noise from particle..." << std::endl;
	#endif

    srand(time(0)); // Seed for random number generation
	double scallignFactor = (Xdim * Ydim) / (PI * cropRadius * cropRadius);
    bool invalidRegion;
    MultidimArray< double > noiseCrop;

	#ifdef DEBUG_NOISE_ESTIMATION
	std::cout << "max_noiseEst  " << max_noiseEst << " min_noiseEst " << min_noiseEst  << std::endl;
	std::cout << "(Ydim/2) " << (Ydim/2) << " (Xdim/2) " << (Xdim/2) << std::endl;
	std::cout << "scallignFactor " << scallignFactor << std::endl;
	#endif

	do {
		invalidRegion = false;
		noiseCrop.initZeros((int)Ydim, (int)Xdim);

		int x = min_noiseEst + cropRadius + rand() % (max_noiseEst - min_noiseEst + 1 - 2*cropRadius);
		int y = min_noiseEst + cropRadius + rand() % (max_noiseEst - min_noiseEst + 1 - 2*cropRadius);

		#ifdef DEBUG_NOISE_ESTIMATION
		std::cout << "x  " << x << " y " << y  << std::endl;
		#endif

		for (int i = -cropRadius; i <= cropRadius; i++)
		{
			for (int j = -cropRadius; j <= cropRadius; j++)
			{
				// Impose circular shape
				if (i*i + j*j > cropRadius*cropRadius) 
				{
					continue;
				}

				if (DIRECT_A2D_ELEM(Pmask(), y + i, x + j) == 0 || DIRECT_A2D_ELEM(PmaskRoi(), y + i, x + j) > 0)	// FIX: si no se ha dado una ROI mask falla
				{
					invalidRegion = true;

					#ifdef DEBUG_NOISE_ESTIMATION
					std::cout << "Invalid region. Trying again..." << std::endl;
					#endif
				
					break;
				}

				DIRECT_A2D_ELEM(noiseCrop,  (Ydim/2) + i, (Xdim/2) + j) = scallignFactor * DIRECT_A2D_ELEM(Idiff(), y + i, x + j);
			}

			if (invalidRegion) {
				break;
			}
		}
	} while (invalidRegion);

	#ifdef DEBUG_NOISE_ESTIMATION
	size_t lastindex = fn_out.find_last_of(".");
	std::string rawname = fn_out.substr(0, lastindex);

	Image<double> saveImage;
	std::string debugFileFn = rawname + "_noiseCrop.mrc";

	saveImage() = noiseCrop;
	saveImage.write(debugFileFn);
	#endif

	FourierTransformer transformerNoise;
	MultidimArray<std::complex<double>> noiseSpectrum;
	transformerNoise.FourierTransform(noiseCrop, noiseSpectrum, false);

	#ifdef DEBUG_NOISE_ESTIMATION
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
		DIRECT_MULTIDIM_ELEM(powerNoise,n) += (DIRECT_MULTIDIM_ELEM(noiseSpectrum,n) * std::conj(DIRECT_MULTIDIM_ELEM(noiseSpectrum,n))).real();
		
	#ifdef DEBUG_NOISE_ESTIMATION
	std::cout << "Noise sucessfully estimated from particle." << std::endl;
	#endif
}

 // Main methods ===================================================================
void ProgSubtractProjection::preProcess() 
{
	// Read input volume, mask and particles metadata
	show();
	V.read(fnVolR);
	V().setXmippOrigin();

	// Read input vol dimensions
	V.getDimensions(Xdim, Ydim, Zdim, Ndim);
	
	// Read input mask or create 2D circular if not provided
	if (maskVolProvided)
	{
		maskVol.read(fnMaskVol);
		maskVol().setXmippOrigin();

		#ifdef DEBUG_OUTPUT_FILES
		maskVol.write(formatString("%s/maskVol.mrc", fnProj.c_str()));
		#endif
	}
	else
	{
		maskVol().initZeros((int)Ydim, (int)Xdim);
		maskVol().setXmippOrigin();

		if (cirmaskrad == -1.0)
			cirmaskrad = (double)XSIZE(V())/2;

		RaisedCosineMask(maskVol(), cirmaskrad, cirmaskrad);

		#ifdef DEBUG_OUTPUT_FILES
		maskVol.write(formatString("%s/cirmask.mrc", fnProj.c_str()));
		#endif
	}

	// Create mock image of same size as particles (and reference volume) to construct frequencies map
	I().initZeros((int)Ydim, (int)Xdim);
	I().initConstant(1);
	transformerI.FourierTransform(I(), IFourier, false);

	wi.initZeros(IFourier);
	Matrix1D<double> w(2);

	for (int i=0; i<YSIZE(wi); i++) {
		FFT_IDX2DIGFREQ(i,YSIZE(IFourier),YY(w)) 
		for (int j=0; j<XSIZE(wi); j++)  {
			FFT_IDX2DIGFREQ(j,YSIZE(IFourier),XX(w))
			DIRECT_A2D_ELEM(wi,i,j) = (int)round((sqrt(YY(w)*YY(w) + XX(w)*XX(w))) * (int)YSIZE(IFourier));
		}
	}

	#ifdef DEBUG_OUTPUT_FILES
	Image<int> saveImage;
	saveImage() = wi; 
	saveImage.write(formatString("%s/wi_freqMap.mrc", fnProj.c_str()));
	#endif

	// Calculate index corresponding to cut-off freq
	if (maxResol == -1.0)
		maxResol = sampling;
	
	// Normalize Nyquist=0.5
	double cutFreq = 0.5 * (sampling/maxResol);
	
	// Analyze up to proyection Freq / sqrt(2) to consider corners
	double substractionCutFreq = (sampling/maxResol) / sqrt(2);
	DIGFREQ2FFT_IDX(substractionCutFreq, (int)YSIZE(IFourier), maxwiIdx);

	#ifdef DEBUG
	std::cout << "------------------- sampling " << sampling << std::endl;
	std::cout << "------------------- maxResol " << maxResol << std::endl;
	std::cout << "------------------- cutFreq " << cutFreq << std::endl;
	std::cout << "------------------- substractionCutFreq " << substractionCutFreq << std::endl;
	std::cout << "------------------- maxwiIdx " << maxwiIdx << std::endl;
	#endif

	// Read or create mask keep and compute inverse of mask keep (mask subtract)
	createMask(fnMaskRoi, vM, ivM);

	// If real space projector every execution must mask-multiply and project the input volume
	if (realSpaceProjector)
	{
		// Apply mask to volume
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
			DIRECT_MULTIDIM_ELEM(V(),n) = DIRECT_MULTIDIM_ELEM(V(),n)*DIRECT_MULTIDIM_ELEM(ivM(),n);
	}

	if (rank==0)
	{
		// Initialize noise power variables
		if (noiseEstimationBool)
		{
			generateNoiseEstimationSideInfo();
		}

		if (!realSpaceProjector)
		{
			// If  Fourier projector one volume is shared by all execution and this operation is done only once
			// Apply mask to volume
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
				DIRECT_MULTIDIM_ELEM(V(),n) = DIRECT_MULTIDIM_ELEM(V(),n)*DIRECT_MULTIDIM_ELEM(ivM(),n);

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

void ProgSubtractProjection::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
 { 
	// Initialize aux variable
	disable = false;

	// Project volume and process projections 
	const auto sizeI = (int)XSIZE(I());
	processParticle(rowIn, sizeI);

	// Build projected and final masks
	if (fnMaskRoi.isEmpty())  // If there is no provided mask
	{
		M().initZeros(P());
		// inverse mask (iM) is all 1s
		iM = invertMask(M);
	}
	else  // If a mask has been provided
	{
		// Mask projection is always calculated in real space
		projectVolume(vM(), PmaskRoi, sizeI, sizeI, part_angles.rot, part_angles.tilt, part_angles.psi, &roffset);

		#ifdef DEBUG_OUTPUT_FILES
		size_t dotPos = fnImgOut.find_last_of('.');
		PmaskRoi.write(fnImgOut.substr(0, dotPos) + "_PmaskRoi" + fnImgOut.substr(dotPos));
		Pmask.write(fnImgOut.substr(0, dotPos) + "_Pmask" + fnImgOut.substr(dotPos));
		#endif

		// Apply binarization, shift and gaussian filter to the projected mask
		M = binarizeMask(PmaskRoi);

		if (subtract) // If the mask contains the part to SUBTRACT: iM = input mask
			iM = M;
		else // If the mask contains the part to KEEP: iM = INVERSE of original maskImgIm
			iM = invertMask(M);
	}

	#ifdef DEBUG_OUTPUT_FILES
	size_t dotPos = fnImgOut.find_last_of('.');
	M.write(fnImgOut.substr(0, dotPos) + "_M" + fnImgOut.substr(dotPos));
	iM.write(fnImgOut.substr(0, dotPos) + "_iM" + fnImgOut.substr(dotPos));
	#endif

	// Estimate background adjustment (b) as the mean difference of the pixels of the regions to adjust
	// (outside iM mask)
	double meanP = 0;
	double meanI = 0;
	double Nelems = 0;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(P())
		if(DIRECT_MULTIDIM_ELEM(iM(), n) > 0 && DIRECT_MULTIDIM_ELEM(PmaskImg(), n) > 0)
		{
			meanP += DIRECT_MULTIDIM_ELEM(P(), n); 
			meanI += DIRECT_MULTIDIM_ELEM(I(), n);
			Nelems++;
		}

	double b = (meanI - meanP) / Nelems;

	I() -= b;

	// Compute estimation images: IiM = I*iM and PiM = P*iM	
	IiMFourier = computeEstimationImage(I(), &(iM()), transformerIiM);
	PiMFourier = computeEstimationImage(Pctf(), &(iM()), transformerPiM);

	// Estimate transformation with model of order 0: T(w) = beta00 and model of order 1: T(w) = beta01 + beta1*w
	MultidimArray<double> num0;
	num0.initZeros(maxwiIdx+1); 
	MultidimArray<double> den0;
	den0.initZeros(maxwiIdx+1);
	Matrix2D<double> A1;
	A1.initZeros(2,2);
	Matrix1D<double> b1;
	b1.initZeros(2);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PiMFourier) {
		int win = DIRECT_MULTIDIM_ELEM(wi, n);
		if (win > 0 && win < maxwiIdx) 
		{
			double realPiMFourier = real(DIRECT_MULTIDIM_ELEM(PiMFourier,n));
			double imagPiMFourier = imag(DIRECT_MULTIDIM_ELEM(PiMFourier,n));
			DIRECT_MULTIDIM_ELEM(num0,win) += real(DIRECT_MULTIDIM_ELEM(IiMFourier,n)) * realPiMFourier
											+ imag(DIRECT_MULTIDIM_ELEM(IiMFourier,n)) * imagPiMFourier;
			DIRECT_MULTIDIM_ELEM(den0,win) += realPiMFourier*realPiMFourier + imagPiMFourier*imagPiMFourier;
			A1(0,0) += realPiMFourier*realPiMFourier + imagPiMFourier*imagPiMFourier;
			A1(0,1) += win*(realPiMFourier + imagPiMFourier);
			A1(1,1) += 2*win;
			b1(0) += real(DIRECT_MULTIDIM_ELEM(IiMFourier,n)) * realPiMFourier + imag(DIRECT_MULTIDIM_ELEM(IiMFourier,n)) * imagPiMFourier;
			b1(1) += win*(real(DIRECT_MULTIDIM_ELEM(IiMFourier,n))+imag(DIRECT_MULTIDIM_ELEM(IiMFourier,n)));
		}
	}
	A1(1,0) = A1(0,1);

	// Compute beta00 from order 0 model
	double beta00 = num0.sum()/den0.sum();	

	// If selected, filter particles with negative beta00
	if (nonNegative && beta00 < 0) 
	{
		disable = true;
	}

	// Apply adjustment order 0: PFourier0 = T(w) * PFourier = beta00 * PFourier
	PFourier0 = PFourier;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PFourier0)
	{
		int win = DIRECT_MULTIDIM_ELEM(wi, n);
		if (win < maxwiIdx) 
		{
			DIRECT_MULTIDIM_ELEM(PFourier0,n) *= beta00;
		}
	}

	// Compute beta01 and beta1 from order 1 model
	PseudoInverseHelper h;
	h.A = A1;
	h.b = b1;
	Matrix1D<double> betas1;
	solveLinearSystem(h,betas1); 
	double beta01 = betas1(0);
	double beta1 = betas1(1);

	// Apply adjustment order 1: PFourier1 = T(w) * PFourier = (beta01 + beta1*w) * PFourier
	PFourier1 = PFourier;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(PFourier1)
		DIRECT_MULTIDIM_ELEM(PFourier1,n) *= (beta01+beta1*DIRECT_MULTIDIM_ELEM(wi,n)); 
	PFourier1(0,0) = IiMFourier(0,0); 

	// Check best model, this function also saves best fitted model in PFourier
	Matrix1D<double> R2adj = checkBestModel(PFourier, PFourier0, PFourier1, IFourier);

	double beta0save;
	double beta1save;

	if (R2adj(1) == 0)
	{
		beta0save = beta00;
		beta1save = 0;
	}
	else
	{
		beta0save = beta01;
		beta1save = beta1;
	}

	// Create empty new image for output particle
	MultidimArray<double> &mIdiff=Idiff();
	mIdiff.initZeros(I());
	mIdiff.setXmippOrigin();

	if (boost) // Boosting of original particles
	{
		if (R2adj(1) == 0)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(IFourier) 
				DIRECT_MULTIDIM_ELEM(IFourier,n) /= beta00; 
		} 
		else if (R2adj(1) == 1)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(IFourier)
				DIRECT_MULTIDIM_ELEM(IFourier,n) /= (beta01+beta1*DIRECT_MULTIDIM_ELEM(wi,n));
		}
		transformerI.inverseFourierTransform(IFourier, Idiff());
	} 
	else  // Subtraction
	{
		// Recover adjusted projection (P) in real space
		transformerP.inverseFourierTransform(PFourier, P());

		// Subtract projection
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mIdiff)
			DIRECT_MULTIDIM_ELEM(mIdiff,n) = (DIRECT_MULTIDIM_ELEM(I(),n) - DIRECT_MULTIDIM_ELEM(P(),n));
	}

	#ifdef DEBUG_OUTPUT_FILES
	dotPos = fnImgOut.find_last_of('.');
	P.write(fnImgOut.substr(0, dotPos) + "_P" + fnImgOut.substr(dotPos));
	I.write(fnImgOut.substr(0, dotPos) + "_I" + fnImgOut.substr(dotPos));
	#endif

	// Estimate noise after sutraction
	if(noiseEstimationBool)
	{
		noiseEstimation();
	}

	writeParticle(rowOut, fnImgOut, Idiff, R2adj(0), beta0save, beta1save, b); 
}

void ProgSubtractProjection::finishProcessing()
{
	if (allow_time_bar && verbose && !single_image)
        progress_bar(time_bar_size);
    writeOutput();

	if(noiseEstimationBool)
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(powerNoise)
		{
			DIRECT_MULTIDIM_ELEM(powerNoise,n) /= (int)mdInSize;
		}

		Image<double> saveImage;
		size_t lastindex = fn_out.find_last_of("/\\");
		std::string noiseEstOuputFile = fn_out.substr(0, lastindex) + "/noisePower.mrc";

		std::cout << "Saving noise power at: " << noiseEstOuputFile << std::endl;

		saveImage() = powerNoise;
		saveImage.write(noiseEstOuputFile);
	}
}
