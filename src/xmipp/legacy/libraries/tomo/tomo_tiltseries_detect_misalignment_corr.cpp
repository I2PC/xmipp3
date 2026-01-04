/***************************************************************************
 *
 * Authors:    Federico P. de Isidro Gomez			  fp.deisidro@cnb.csic.es
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

#include "tomo_tiltseries_detect_misalignment_corr.h"
#include <chrono>
#include <core/xmipp_fft.h>
#include <core/xmipp_fftw.h>
#include <data/fourier_filter.h>
#include <core/transformations.h>



// --------------------------- INFO functions ----------------------------
void ProgTomoTSDetectMisalignmentCorr::readParams()
{
	fnIn = getParam("-i");
	fnTiltAngles = getParam("--tlt");
	fnOut = getParam("-o");
	shiftTol = getDoubleParam("--shiftTol");
	samplingRate = getDoubleParam("--samplingRate");
}


void ProgTomoTSDetectMisalignmentCorr::defineParams()
{
	addUsageLine("This program detect the misaligned images in a tilt-series calcualting the relatve shift between contiguous images.");
	addParamsLine("  -i <mrcs_file=\"\">                   	: Input tilt-series.");
	addParamsLine("  --tlt <xmd_file=\"\">      			: Input file containning the tilt angles of the tilt-series in .xmd format.");
	addParamsLine("  -o <o=\"./alignemntReport.xmd\">      	: Output file containing the alignemnt report.");
	addParamsLine("  [--shiftTol <shiftTol=1>]				: Sampling rate of the input tomogram (A/px).");
	addParamsLine("  [--samplingRate <samplingRate=1>]		: Sampling rate of the input tomogram (A/px).");
}


void ProgTomoTSDetectMisalignmentCorr::generateSideInfo()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Generating side info..." << std::endl;
	#endif

	// Read tilt angles file
	MetaDataVec inputTiltAnglesMd;
	inputTiltAnglesMd.read(fnTiltAngles);

	size_t objIdTlt;
	double tiltAngle;

	for(size_t objIdTlt : inputTiltAnglesMd.ids())
	{
		inputTiltAnglesMd.getValue(MDL_ANGLE_TILT, tiltAngle, objIdTlt);
		tiltAngles.push_back(tiltAngle);
	}


	#ifdef VERBOSE_OUTPUT
	std::cout << "Input tilt angles read from: " << fnTiltAngles << std::endl;
	#endif

	// Initialize local alignment vector (depends on the number of acquisition angles)
	localAlignment.resize(nSize, true);

	// Normalized dimension
	normDim = (xSize<ySize) ? xSize : ySize;

	#ifdef VERBOSE_OUTPUT
	std::cout << "Side info generated succesfully!" << std::endl;
	#endif
}


// --------------------------- HEAD functions ----------------------------
void ProgTomoTSDetectMisalignmentCorr::lowpassFilter(MultidimArray<double> &tiltImage)
{
	MultidimArray< std::complex<double> > fftTI;

    FourierTransformer transformer;
	transformer.FourierTransform(tiltImage, fftTI, false);

	int n=0;
	
	double freq = samplingRate / (normDim);
	
	double w= 0.05;
	double cutoffFreq = freq + w;
	double delta = PI / w;

	#ifdef DEBUG_PREPROCESS
	std::cout << "samplingRate " << samplingRate << std::endl;
	std::cout << "freq " << freqLow << std::endl;
	std::cout << "cutoffFreq " << cutoffFreqLow << std::endl;
	#endif

	for(size_t i=0; i<YSIZE(fftTI); ++i)
	{
		double uy;
		double ux;
		double uy2;

		FFT_IDX2DIGFREQ(i,ySize,uy);
		uy2=uy*uy;

		for(size_t j=0; j<XSIZE(fftTI); ++j)
		{
			FFT_IDX2DIGFREQ(j,xSize,ux);
			double u=sqrt(uy2+ux*ux);

			if(u > cutoffFreq)
			{
				DIRECT_MULTIDIM_ELEM(fftTI, n) = 0;
			} 
			else if(u >= freq && u < cutoffFreq)
			{
				DIRECT_MULTIDIM_ELEM(fftTI, n) *= 0.5*(1+cos((u-freq)*delta));
			}
			
			++n;
		}
	}

	transformer.inverseFourierTransform();
}


void ProgTomoTSDetectMisalignmentCorr::detectSubtleMisalingment(MultidimArray<double> &ts)
{
	std::cout << "Detecting misalignment..." << std::endl;
	MultidimArray<double> ti_bw;
	MultidimArray<double> ti_fw;

	ti_bw.initZeros(ySize, xSize);
	ti_fw.initZeros(ySize, xSize);

	double ta_bw;
	double ta_fw;

	for (size_t n = 0; n < nSize-1; n++)
	{
		// Construct forward and backward images
		for (size_t i = 0; i < ySize; i++)
		{
			for (size_t j = 0; j < xSize; j++)
			{
				DIRECT_A2D_ELEM(ti_bw, i, j) = DIRECT_NZYX_ELEM(ts, n, 0, i, j);
				DIRECT_A2D_ELEM(ti_fw, i, j) = DIRECT_NZYX_ELEM(ts, n+1, 0, i, j);
			}
		}

		#ifdef DEBUG_OUTPUT_FILES
		Image<double> saveImage;

		saveImage() = ti_bw;
		saveImage.write("./ti_bw.mrc");

		saveImage() = ti_fw;
		saveImage.write("./ti_fw.mrc");
		#endif

		// Apply cosine stretching
		MultidimArray<double> ti_bw_cs;
		MultidimArray<double> ti_fw_cs;
		ti_bw_cs = ti_bw;
		ti_fw_cs = ti_fw;

		ta_bw = tiltAngles[n];
		ta_fw = tiltAngles[n+1];

		if (abs(ta_bw)>abs(ta_fw))
		{
			cosineStretching(ti_bw, ta_bw, ta_fw);
		}
		else
		{
			cosineStretching(ti_fw, ta_fw, ta_bw);
		}


		#ifdef DEBUG_OUTPUT_FILES
		saveImage() = ti_bw;
		saveImage.write("./ti_bw_cs.mrc");

		saveImage() = ti_fw;
		saveImage.write("./ti_fw_cs.mrc");
		#endif

		// Calculate shift for maximum correlation
		Matrix2D<double> relShift;
		relShift = maxCorrelationShift(ti_bw, ti_fw);

		relativeShifts.push_back(relShift);

		if (abs(MAT_ELEM(relShift, 0, 1)) > shiftTol)
		{
			localAlignment[n] = false;
		}	
	}

	for (size_t i = 0; i < relativeShifts.size(); i++)
	{
		std::cout << "image " << i << " "
					 "[" << MAT_ELEM(relativeShifts[i], 0, 0) << " ," 
						 << MAT_ELEM(relativeShifts[i], 0, 1) << " ]"
														<< std::endl;
	}
}


void ProgTomoTSDetectMisalignmentCorr::refineAlignment(MultidimArray<double> &ts)
{
	std::cout << "Refine alignment..." << std::endl;
	MultidimArray<double> ti_bw;
	MultidimArray<double> ti_fw;

	ti_bw.initZeros(ySize, xSize);
	ti_fw.initZeros(ySize, xSize);

	double ta_bw;
	double ta_fw;

	MultidimArray<double> ti;
	MultidimArray<double> ti_tmp;
	ti_tmp.initZeros(ySize, xSize);

	double xShiftPrev = 0;
	double yShiftPrev = 0;

	for (size_t n = 0; n < nSize-2; n++)
	{
		for (size_t i = 0; i < ySize; i++)
		{
			for (size_t j = 0; j < xSize; j++)
			{
				DIRECT_A2D_ELEM(ti_tmp, i, j) = DIRECT_NZYX_ELEM(ts, n, 0, i, j);
			}
		}
		
		Matrix2D<double> m;
		m.initIdentity(3);

		MAT_ELEM(m, 0, 2) = MAT_ELEM(relativeShifts[n], 0, 0) + xShiftPrev;
		MAT_ELEM(m, 1, 2) = MAT_ELEM(relativeShifts[n], 0, 1) + yShiftPrev;

		xShiftPrev += MAT_ELEM(relativeShifts[n], 0, 0);
		yShiftPrev += MAT_ELEM(relativeShifts[n], 0, 1);		
	
    	applyGeometry(1, ti, ti_tmp, m, xmipp_transformation::IS_NOT_INV, xmipp_transformation::DONT_WRAP);

		for (size_t i = 0; i < ySize; i++)
		{
			for (size_t j = 0; j < xSize; j++)
			{
				DIRECT_NZYX_ELEM(ts, n, 0, i, j) = DIRECT_A2D_ELEM(ti, i, j);
			}
		}
	}

	#ifdef DEBUG_OUTPUT_FILES
	Image<double> saveImage;

	saveImage() = ts;
	saveImage.write("./ts_ali.mrc");
	#endif
}


// --------------------------- I/O functions ----------------------------
void ProgTomoTSDetectMisalignmentCorr::writeOutputShifts()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Saving calculated shifts... " << std::endl;
	#endif

	MetaDataVec md;
	size_t id;

	for(size_t i = 0; i < relativeShifts.size(); i++)
	{
		id = md.addObject();
		md.setValue(MDL_SHIFT_X, MAT_ELEM(relativeShifts[i], 0, 0), id);
		md.setValue(MDL_SHIFT_Y, MAT_ELEM(relativeShifts[i], 0, 1), id);
	}

	size_t li = fnOut.find_last_of("\\/");
	std::string rn = fnOut.substr(0, li);
	std::string outShiftsFn;
    outShiftsFn = rn + "/outputShifts.xmd";

	md.write(outShiftsFn);

	#ifdef VERBOSE_OUTPUT
	std::cout << "Output shifts saved at: " << outShiftsFn << std::endl;
	#endif
}


void ProgTomoTSDetectMisalignmentCorr::writeOutputAlignmentReport()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Saving alginemnt report... " << std::endl;
	#endif
	
	MetaDataVec md;
	FileName fn;
	size_t id;

	size_t li = fnIn.find_last_of(":");
	std::string rawnameTS = fnIn.substr(0, li);

	for(size_t i = 0; i < localAlignment.size(); i++)
	{
		fn.compose(i + FIRST_IMAGE, rawnameTS);
		id = md.addObject();

		// Tilt-image			
		md.setValue(MDL_IMAGE, fn, id);

		// Alignment
		if(localAlignment[i])
		{
			md.setValue(MDL_ENABLED, 1, id);
		}
		else
		{
			md.setValue(MDL_ENABLED, -1, id);
		}
	}

	md.write(fnOut);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Alignment report saved at: " << fnOut << std::endl;
	#endif
}


// --------------------------- MAIN ----------------------------------
void ProgTomoTSDetectMisalignmentCorr::run()
{
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;	

	auto t1 = high_resolution_clock::now();

	size_t Xdim, Ydim;

	MetaDataVec tiltseriesmd;
    Image<double> tiltSeriesImages;

    // if (fnIn.isMetaData())
    // {
    //     tiltseriesmd.read(fnIn);
    // }
    // else
    // {
    //     tiltSeriesImages.read(fnIn, HEADER);

    //     size_t Zdim, Ndim;
    //     tiltSeriesImages.getDimensions(Xdim, Ydim, Zdim, Ndim);

    //     if (fnIn.getExtension() == "mrc" and Ndim == 1)
    //         Ndim = Zdim;

    //     size_t id;
    //     FileName fn;
    //     for (size_t i = 0; i < Ndim; i++) 
    //     {
    //         id = tiltseriesmd.addObject();
    //         fn.compose(i + FIRST_IMAGE, fnIn);
    //         tiltseriesmd.setValue(MDL_IMAGE, fn, id);
    //     }
    // }

	tiltSeriesImages.read(fnIn);
	tiltSeriesImages.getDimensions(xSize, ySize, zSize, nSize);

	if (fnIn.getExtension() == "mrc" and nSize == 1)
		nSize = zSize;


	#ifdef DEBUG_DIM
	std::cout << "Input tilt-series dimensions:" << std::endl;
	std::cout << "x " << xSize << std::endl;
	std::cout << "y " << ySize << std::endl;
	std::cout << "z " << zSize << std::endl;
	std::cout << "n " << nSize << std::endl;
	#endif

	generateSideInfo();

	auto &ptrts = tiltSeriesImages();
	detectSubtleMisalingment(ptrts);
	// refineAlignment(ptrts);

	writeOutputShifts();
	writeOutputAlignmentReport();

	auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1); 	// Getting number of milliseconds as an integer
 	std::cout << "Execution time: " << ms_int.count() << "ms\n";
}


// --------------------------- UTILS functions ----------------------------
void ProgTomoTSDetectMisalignmentCorr::cosineStretching(MultidimArray<double> &ti, double ti_angle_high, double ti_angle_low)
{
	Matrix2D<double> sm = getCosineStretchingMatrix(ti_angle_high, ti_angle_low);
	MultidimArray<double> ti_tmp;
	ti_tmp = ti;

    applyGeometry(1, ti, ti_tmp, sm, xmipp_transformation::IS_NOT_INV, xmipp_transformation::DONT_WRAP);
}


Matrix2D<double> ProgTomoTSDetectMisalignmentCorr::getCosineStretchingMatrix(double ti_angle_high, double ti_angle_low)
{
	double cosFactor = cos(ti_angle_low * PI/180.0) / cos(ti_angle_high * PI/180.0);
	Matrix2D<double> m(3,3);

	MAT_ELEM(m, 0, 0) = cosFactor;
	// MAT_ELEM(m, 0, 1) = 0;
	// MAT_ELEM(m, 0, 2) = 0;
	// MAT_ELEM(m, 1, 0) = 0;
	MAT_ELEM(m, 1, 1) = 1;
	// MAT_ELEM(m, 1, 2) = 0;
	// MAT_ELEM(m, 2, 0) = 0;
	// MAT_ELEM(m, 2, 1) = 0;
	MAT_ELEM(m, 2, 2) = 1;

	return m;
}


Matrix2D<double> ProgTomoTSDetectMisalignmentCorr::maxCorrelationShift(MultidimArray<double> &ti1, MultidimArray<double> &ti2)
{
	double shiftX;
	double shiftY;
	CorrelationAux aux;

	// bestShift(ti1, ti2, shiftX, shiftY, aux, nullptr, 100);

	// Matrix2D<double> relShift(2, 1);

	// MAT_ELEM(relShift, 0, 0) = shiftX;
	// MAT_ELEM(relShift, 0, 1) = shiftY;

	// 		// double bestShift(const MultidimArray<double> &I1, const MultidimArray<double> &I2,
	// 		//            double &shiftX, double &shiftY, CorrelationAux &aux,
	// 		//            const MultidimArray<int> *mask, int maxShift)

	Matrix2D<double> relShift(2, 1);
	MultidimArray<double> tsCorr;

	// Shift the particle respect to its symmetric to look for the maximum correlation displacement
	correlation_matrix(ti1, ti2, tsCorr, aux, true);

	#ifdef DEBUG_OUTPUT_FILES
	Image<double> saveImage;

	saveImage() = tsCorr;
	saveImage.write("./tsCorr.mrc");
	#endif

	auto maximumCorrelation = MINDOUBLE;
	int xDisplacement = 0;
	int yDisplacement = 0;

	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(tsCorr)
	{
		double value = DIRECT_A2D_ELEM(tsCorr, i, j);

		if (value > maximumCorrelation)
		{
			#ifdef DEBUG_TI_CORR
			std::cout << "new maximumCorrelation " << value << " at (" << i << ", " << j << ")" << std::endl;
			#endif

			maximumCorrelation = value;
			xDisplacement = j;
			yDisplacement = i;
		}
	}
	
	MAT_ELEM(relShift, 0, 0) = (double)(xDisplacement) - (double)xSize / 2;
	MAT_ELEM(relShift, 0, 1) = (double)(yDisplacement) - (double)ySize / 2;


	#ifdef DEBUG_TI_CORR
	std::cout << "maximumCorrelation " << maximumCorrelation << std::endl;
	std::cout << "xDisplacement " << -(double)(xDisplacement) + (double)xSize / 2 << std::endl;
	std::cout << "yDisplacement " << -(double)(yDisplacement) + (double)ySize / 2 << std::endl;

	std::cout << "Correlation volume dimensions (" << XSIZE(tsCorr) << ", " << YSIZE(tsCorr) << ")" << std::endl;
	#endif

	return relShift;
}


// void ProgTomoTSDetectMisalignmentCorr::removeOutliers(MultidimArray<double> &ti)
// {
// 	double avg;
// 	double std;

// 	ti.computeAvgStdev(avg, std);

// 	double thr = avg - 2*std;

// 	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(ti)
// 	{
// 		if (DIRECT_A1D_ELEM(ti, n) < thr)
// 		{
// 			DIRECT_A1D_ELEM(ti, n) = avg;
// 		}
// 	}
// }


// void ProgTomoTSDetectMisalignmentCorr::removeOutliers(MultidimArray<double> &ti) 
// {
// 	size_t windowSize = 64;
// 	MultidimArray<double> window;

// 	double avg;
// 	double std;

// 	std::cout << "lets go con esta movida " <<std::endl;
//     // Iterate through each pixel in the ti
//     for (size_t i = 0; i < ySize; ++i) 
// 	{
//         for (size_t j = 0; j < xSize; ++j) 
// 		{
//             // Define the local region based on the windowSize
//             int y0 = std::max(0, int(i - windowSize / 2));
//             int x0 = std::max(0, int(j - windowSize / 2));
//             int yF = std::min(ySize - 1, i + windowSize / 2);
//             int xF = std::min(xSize - 1, j + windowSize / 2);

//             // Construct window
// 			window2D(ti, window, y0, x0, yF, xF);

// 			window.computeAvgStdev(avg, std);

// 			double thr = avg - std;

// 			if (DIRECT_A2D_ELEM(ti, i, j) < thr)
// 			{
// 				DIRECT_A2D_ELEM(ti, i, j) = avg;
// 			}
//         }
//     }
// }
