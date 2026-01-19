/***************************************************************************
 *
 * Authors:     Jose Luis Vilas 			(jlvilas@cnb.csic.es)
 *
 * Yale University, New Haven, Connecticut, United States of America
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

#include "angular_resolution_alignment.h"
#include <core/metadata_extension.h>
#include <data/monogenic_signal.h>
#include <data/fourier_filter.h>
#include <random>
#include <limits>
#include <ctpl_stl.h>

void ProgAngResAlign::defineParams()
{
	addUsageLine("This algorithm estimates the existence of angular alignment errors by measuring a set of directional FSC.");
	addUsageLine("To do that, the method only makes use of two half maps, but it does not use the set of particles.");
	addUsageLine("The result is a plot resolution-radius. When this plot presents a slope the map has angular assignment error.");
	addUsageLine("In contrast, if the curve is flat, the map is error free or there is shift errors in the particle alignment.");
	addUsageLine("Reference: J.L. Vilas, et. al, XXXXX (2023)");
	addUsageLine("+ The optimal con angle is 17 degree.", true);
	addUsageLine(" This fact was proved in J.L Vilas & H.D. Tagare Nat. Meth 2023. Other values can be used,");
	addUsageLine(" and this parameter does not seems to affect in a significative manner");
	addUsageLine("+* On the helix", true);
	addUsageLine(" If the map is a helix, the helix option should be activated. This option ensures a better estimation of the angular");
	addUsageLine(" assignment errors. The helix is assumen that is oriented along the z-axis. This flag modifies the shape of the ");
	addUsageLine("gaussian mask usign a cylinder.");
	addUsageLine(" ");
	addUsageLine(" ");
	addSeeAlsoLine("resolution_fsc");

	addParamsLine("   --half1 <input_file>               : Input Half map 1");
	addParamsLine("   --half2 <input_file>               : Input Half map 2");
	addParamsLine("   [--directional_resolution]         : (Optional) Uses direcitonal FSC instead of global FSC shell ");
	addParamsLine("   [--limit_radius]                   : (Optional) Limits the maximum radius ");

	addParamsLine("   [-o <output_folder=\"\">]          : Folder where the results will be stored.");

	addParamsLine("   [--sampling <Ts=1>]                : (Optical) Pixel size (Angstrom). If it is not provided by default will be 1 A/px.");
	addParamsLine("   [--mask <input_file=\"\">]         : (Optional) Smooth mask to remove noise. If it is not provided, the computation will be carried out without mask.");

	addParamsLine("   [--anglecone <ang_con=17>]         : (Optional) Angle Cone (angle between the axis and the  generatrix) for estimating the directional FSC");
	addParamsLine("   [--threshold <thrs=0.143>]         : (Optional) Threshold for the FSC/directionalFSC estimation ");
	addParamsLine("   [--helix]                          : (Optional) If the reconstruction is a helix put this flag. The axis of the helix must be along the z-axis");
	addParamsLine("   [--threads <Nthreads=1>]           : (Optional) Number of threads to be used");

	addExampleLine("Resolution of two half maps half1.mrc and half2.mrc with a sampling rate of 2 A/px", false);
	addExampleLine("xmipp_angular_resolution_alignment --half1 half1.mrc --half2 half2.mrc --sampling 2 ");
	addExampleLine("Resolution of two half maps half1.mrc and half2.mrc with a sampling rate of 2 A/px and a mask mask.mrc", false);
	addExampleLine("xmipp_angular_resolution_alignment --half1 half1.mrc --half2 half2.mrc --mask mask.mrc --sampling 2 ");
}

void ProgAngResAlign::readParams()
{
	fnhalf1 = getParam("--half1");
	fnhalf2 = getParam("--half2");
	fnOut = getParam("-o");

	sampling = getDoubleParam("--sampling");
	fnmask = getParam("--mask");
	directionalRes = checkParam("--directional_resolution");
	limRad = checkParam("--limit_radius");
	isHelix = checkParam("--helix");
	ang_con = getDoubleParam("--anglecone");
	thrs = getDoubleParam("--threshold");
	
	Nthreads = getIntParam("--threads");
}

void ProgAngResAlign::defineFrequenciesSimple(const MultidimArray<double> &inputVol)
{

	xvoldim = XSIZE(inputVol);
	yvoldim = YSIZE(inputVol);
	zvoldim = ZSIZE(inputVol);

	// Setting the Fouer dimensions
	size_t xdimFourier = XSIZE(inputVol)/2 + 1;
	size_t zdimFourier = zvoldim;
	size_t ydimFourier = yvoldim;

	// Initializing the frequency vectors
	freq_fourier_z.initZeros(zdimFourier);
	freq_fourier_x.initZeros(xdimFourier);
	freq_fourier_y.initZeros(ydimFourier);

	// u is the frequency
	double u;

	// Defining frequency components. First element should be 0, it is set as the smallest number to avoid singularities

	VEC_ELEM(freq_fourier_z,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<zdimFourier; ++k){
		FFT_IDX2DIGFREQ(k,zvoldim, u);
		VEC_ELEM(freq_fourier_z, k) = u;
	}

	VEC_ELEM(freq_fourier_y,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<ydimFourier; ++k){
		FFT_IDX2DIGFREQ(k,yvoldim, u);
		VEC_ELEM(freq_fourier_y, k) = u;
	}

	VEC_ELEM(freq_fourier_x,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<xdimFourier; ++k){
		FFT_IDX2DIGFREQ(k,xvoldim, u);
		VEC_ELEM(freq_fourier_x, k) = u;
	}
	
	//Initializing map with frequencies

	freqMap.initZeros(1, zdimFourier, ydimFourier, xdimFourier);  //Nyquist is 2, we take 1.9 greater than Nyquist

	freqElems.initZeros(xvoldim/2+1);

	// Directional frequencies along each direction
	double uz, uy, ux, uz2, uz2y2;
	long n=0;
	int idx = 0;

	// Ncomps is the number of frequencies lesser than Nyquist
	long Ncomps = 0;
		
	for(size_t k=0; k<zdimFourier; ++k)
	{
		uz = VEC_ELEM(freq_fourier_z, k);
		uz2 = uz*uz;
		for(size_t i=0; i<ydimFourier; ++i)
		{
			uy = VEC_ELEM(freq_fourier_y, i);
			uz2y2 = uz2 + uy*uy;

			for(size_t j=0; j<xdimFourier; ++j)
			{
				ux = VEC_ELEM(freq_fourier_x, j);
				ux = sqrt(uz2y2 + ux*ux);

				DIRECT_MULTIDIM_ELEM(freqMap,n) = 1.9;
				if	(ux<=0.5)
				{
					idx = (int) round(ux * xvoldim);
					++Ncomps;
					DIRECT_MULTIDIM_ELEM(freqElems, idx) += 1;

					DIRECT_MULTIDIM_ELEM(freqMap,n) = 1/ux;

					if ((j == 0) && (uy<0))
					{
						DIRECT_MULTIDIM_ELEM(freqMap,n) = 1.9;
						DIRECT_MULTIDIM_ELEM(freqElems,idx) -= 1;
						--Ncomps;
					}
						
					if ((i == 0) && (j == 0) && (uz<0))
					{
						DIRECT_MULTIDIM_ELEM(freqMap,n) = 1.9;
						DIRECT_MULTIDIM_ELEM(freqElems,idx) -= 1;
						--Ncomps;
					}
						
				}				
				++n;
			}
		}
	}
	real_z1z2.initZeros(Ncomps);
	
}


void ProgAngResAlign::arrangeFSC_and_fscGlobal()
	{
		// cumpos is the the cumulative number of frequencies per shell number
		// First shell has 0 elements
		// second shell has the number of elements of the first shell
		// Third shell has the number of elements of the first+sencond shells and so on
		// freqElems in .h and set in defineFreq function
		cumpos.resizeNoCopy(NZYXSIZE(freqElems));

		DIRECT_MULTIDIM_ELEM(cumpos,0) = 0;
		for (long n = 1; n<NZYXSIZE(cumpos); ++n)
		{
			DIRECT_MULTIDIM_ELEM(cumpos,n) = DIRECT_MULTIDIM_ELEM(cumpos,n-1) + DIRECT_MULTIDIM_ELEM(freqElems,n-1);
		}
 		
		// real_z1z2, absz1_vec and absz2_vec will storage real(z1*conj(z2)), ||z1|| and ||z2|| using the
		//Fourier Coefficients z1, and z2 of both half maps. The storage will be in order, first elements of
		// the vector are the freq 0, next freq 1 and so on. Ncomps is the number of components with frequency
		// lesser than Nyquist (defined in defineFreq)

		absz1_vec = real_z1z2;
		absz2_vec = real_z1z2;
		
		// num and den are de numerator and denominator of the fsc
		MultidimArray<long> pos;
		MultidimArray<double> num, den1, den2, noiseAmp;
		num.initZeros(freqElems);
		pos.resizeNoCopy(num);

		freqidx.resizeNoCopy(real_z1z2);

		// arr2indx is the position of in Fourier space. It is defined in the .h
		// Because we only work with the frequencies lesser than nyquist their position
		// int he Fourier space is lost. arr2indx has the position of each frequency. 
		// It is used at the end of the algorithm to generate the 3DFSC
		freqidx.initZeros();
		pos.initZeros();
		den1 = num;
		den2 = num;

		auto ZdimFT1=(int)ZSIZE(FT1);
		auto YdimFT1=(int)YSIZE(FT1);
		auto XdimFT1=(int)XSIZE(FT1);

		// fx, fy, fz, defined in the .h are the frequencies of each pixel with frequency
		// lesser than Nyquist along each axis. They are used to compute the dot product
		// between each vector position and the direction of the cone to calculate the
		// directional FSC
		fx.resizeNoCopy(real_z1z2);
		fy.resizeNoCopy(real_z1z2);
		fz.resizeNoCopy(real_z1z2);

		long n = 0;
		for (int k=0; k<ZdimFT1; k++)
		{
			double uz = VEC_ELEM(freq_fourier_z, k);
			for (int i=0; i<YdimFT1; i++)
			{
				double uy = VEC_ELEM(freq_fourier_y, i);
				for (int j=0; j<XdimFT1; j++)
				{
					double ux = VEC_ELEM(freq_fourier_x, j);
					
					double iun = DIRECT_MULTIDIM_ELEM(freqMap,n);
					double f = 1/iun;
					++n;

					// Only reachable frequencies
					// To speed up the algorithm, only are considered those voxels with frequency lesser than Nyquist, 0.5. The vector idx_count
					// stores all frequencies lesser than Nyquist. This vector determines the frequency of each component of
					// real_z1z2, absz1_vec, absz2_vec.
					if (f>0.5)
						continue;
					
					// Index of each frequency
					auto idx = (int) round(f * xvoldim);
					
					int idx_count = DIRECT_MULTIDIM_ELEM(cumpos, idx) + DIRECT_MULTIDIM_ELEM(pos, idx); 

					// Storing normalized frequencies
					DIRECT_MULTIDIM_ELEM(fx, idx_count) = (float) ux*iun;
					DIRECT_MULTIDIM_ELEM(fy, idx_count) = (float) uy*iun;
					DIRECT_MULTIDIM_ELEM(fz, idx_count) = (float) uz*iun;

					// In this vector we store the index of each Fourier coefficient, thus in the 
					// directional FSC estimation the calculus of the index is avoided speeding up the calculation
					DIRECT_MULTIDIM_ELEM(freqidx, idx_count) = idx;

					// Fourier coefficients of both halves
					std::complex<double> &z1 = dAkij(FT1, k, i, j);
					std::complex<double> &z2 = dAkij(FT2, k, i, j);

					double absz1 = abs(z1);
					double absz2 = abs(z2);

					DIRECT_MULTIDIM_ELEM(real_z1z2, idx_count) = (float) real(conj(z1)*z2);
					DIRECT_MULTIDIM_ELEM(absz1_vec, idx_count) = (float) absz1*absz1;
					DIRECT_MULTIDIM_ELEM(absz2_vec, idx_count) = (float) absz2*absz2;

					DIRECT_MULTIDIM_ELEM(pos, idx) +=1;
				}
			}
		}
	}

void ProgAngResAlign::fscGlobal(double &threshold, double &resol)
	{
		// num and den are de numerator and denominator of the fsc
		MultidimArray<double> num, den1, den2;
		num.initZeros(freqElems);
		den1 = num;
		den2 = num;

		auto ZdimFT1=(int)ZSIZE(FT1);
		auto YdimFT1=(int)YSIZE(FT1);
		auto XdimFT1=(int)XSIZE(FT1);

		long n = 0;
		for (int k=0; k<ZdimFT1; k++)
		{
			for (int i=0; i<YdimFT1; i++)
			{
				for (int j=0; j<XdimFT1; j++)
				{
					auto iun = DIRECT_MULTIDIM_ELEM(freqMap,n);
					double f = 1/iun;
					++n;

					// Only reachable frequencies
					// To speed up the algorithm, only are considered those voxels with frequency lesser than Nyquist, 0.5. The vector idx_count
					// stores all frequencies lesser than Nyquist. This vector determines the frequency of each component of
					// real_z1z2, absz1_vec, absz2_vec.
					if (f>0.5)
						continue;
					
					// Index of each frequency
					auto idx = (int) round(f * xvoldim);
					
					// Fourier coefficients of both halves
					std::complex<double> &z1 = dAkij(FT1, k, i, j);
					std::complex<double> &z2 = dAkij(FT2, k, i, j);

					double absz1 = abs(z1);
					double absz2 = abs(z2);

					DIRECT_MULTIDIM_ELEM(num, idx) = (float) real(conj(z1)*z2);
					DIRECT_MULTIDIM_ELEM(den1, idx) = (float) absz1*absz1;
					DIRECT_MULTIDIM_ELEM(den2, idx) = (float) absz2*absz2;
				}
			}
		}
		MultidimArray<double> fsc;
		fsc.resizeNoCopy(num);
		fsc.initConstant(1.0);

		//The fsc is stored in a metadata and saved
		bool flagRes = true;
		FOR_ALL_ELEMENTS_IN_ARRAY1D(num)
		{
			double auxfsc = (dAi(num,i))/(sqrt(dAi(den1,i)*dAi(den2,i))+1e-38);
			dAi(fsc,i) = std::max(0.0, auxfsc);

			if (flagRes && (i>2) && (dAi(fsc,i)<=threshold))
			{
				flagRes = false;
				double ff = (double) i / (xvoldim * sampling); // frequency
				resol = 1./ff;
			}
		}


	}


void ProgAngResAlign::fscInterpolation(const MultidimArray<double> &freq, const MultidimArray< double > &frc)
{
	// Here the FSC at 0.143 is obtained by interpolating
	FOR_ALL_ELEMENTS_IN_ARRAY1D(freq)
	{
		auto ff = dAi(frc,i);
		if ( (ff<=thrs) && (i>2) )
		{
			double y2, y1, x2, x1, slope, ny;
			y1 = ff;
			x1 = dAi(freq,i);
			y2 = dAi(frc, i-1);
			x2 = dAi(freq, i-1);
			slope = (y2 - y1)/(x2 - x1);
			ny = y2 - slope*x2;

			double fscResolution;
			fscResolution = (thrs - ny)/slope;
			std::cout << "Resolution " << 1/fscResolution << std::endl;
			break;
		}
	}
}



void ProgAngResAlign::fscDir_fast(MultidimArray<float> &fsc, double rot, double tilt,
				double &thrs, double &resol)
{
	// FSCDIR_FAST: computes the directional FSC along a direction given by the angles rot and tilt.
	// Thus the directional resolution, resol, is estimated with the threshold, thrs.
	// The direcional FSC is stored in a metadata mdRes and in the multidimarray fsc. Later, this multidimarray
	// will be used to estimate the FSO.
	// In addition, the 3dfsc and a normalizationmap (needed to estimate the 3dfsc) are created. For each direction
	// these vectors are updated
	size_t dim = NZYXSIZE(freqElems);
	
	// numerator and denominator of the fsc
	MultidimArray<float> num, den1, den2;	
	num.initZeros(dim);
	den1.initZeros(dim);
	den2.initZeros(dim);

	//Parameter to determine the direction of the cone
	float x_dir, y_dir, z_dir, cosAngle, aux;
	x_dir = sinf(tilt)*cosf(rot);
	y_dir = sinf(tilt)*sinf(rot);
	z_dir = cosf(tilt);

	cosAngle = (float) cos(ang_con);

	// It is multiply by 0.5 because later the weight is
	// cosine = sqrt(exp( -((cosine -1)*(cosine -1))*aux )); 
	// thus the computation of the weight is speeded up
	// aux = 4.0/((cos(ang_con) -1)*(cos(ang_con) -1));
	aux = (4.0/((cosAngle -1)*(cosAngle -1)));//*0.5;

	// Computing directional resolution
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(real_z1z2)
	{
		// frecuencies of the fourier position
		auto ux = DIRECT_MULTIDIM_ELEM(fx, n);
		auto uy = DIRECT_MULTIDIM_ELEM(fy, n);
		auto uz = DIRECT_MULTIDIM_ELEM(fz, n);

		// angle between the position and the direction of the cone
		float cosine = fabs(x_dir*ux + y_dir*uy + z_dir*uz);
		
		// Only points inside the cone
		if (cosine >= cosAngle)	
		{
			// in fact it is sqrt(exp( -((cosine -1)*(cosine -1))*aux )); 
			// For sake of performance The sqrt is avoided dividing the by 2 in aux
			cosine = expf( -((cosine -1)*(cosine -1))*aux); 

			// selecting the frequency of the shell
			size_t idxf = DIRECT_MULTIDIM_ELEM(freqidx, n);

			// computing the numerator and denominator
			dAi(num, idxf) += DIRECT_MULTIDIM_ELEM(real_z1z2, n)  * cosine;
			dAi(den1,idxf) += DIRECT_MULTIDIM_ELEM(absz1_vec, n)  * cosine;
			dAi(den2,idxf) += DIRECT_MULTIDIM_ELEM(absz2_vec, n)  * cosine;
		}
	}

	fsc.resizeNoCopy(num);
	fsc.initConstant(1.0);

	//The fsc is stored in a metadata and saved
	bool flagRes = true;
	FOR_ALL_ELEMENTS_IN_ARRAY1D(num)
	{
		double auxfsc = (dAi(num,i))/(sqrt(dAi(den1,i)*dAi(den2,i))+1e-38);
		dAi(fsc,i) = std::max(0.0, auxfsc);

		if (flagRes && (i>2) && (dAi(fsc,i)<=thrs))
		{
			flagRes = false;
			double ff = (float) i / (xvoldim * sampling); // frequency
			resol = 1./ff;
		}
	}
}


void ProgAngResAlign::generateDirections(Matrix2D<float> &angles, bool alot)
{
	if (alot == true)
	{
		angles.initZeros(2,321);
		MAT_ELEM(angles,0,0)=0.0f;			MAT_ELEM(angles,1,0)=0.0f;
		MAT_ELEM(angles,0,1)=324.0f;		MAT_ELEM(angles,1,1)=63.4349f;
		MAT_ELEM(angles,0,2)=36.0f;			MAT_ELEM(angles,1,2)=63.4349f;
		MAT_ELEM(angles,0,3)=180.0f;		MAT_ELEM(angles,1,3)=63.435f;
		MAT_ELEM(angles,0,4)=252.0f;		MAT_ELEM(angles,1,4)=63.435f;
		MAT_ELEM(angles,0,5)=108.0f;		MAT_ELEM(angles,1,5)=63.435f;
		MAT_ELEM(angles,0,6)=324.0f;		MAT_ELEM(angles,1,6)=31.7175f;
		MAT_ELEM(angles,0,7)=36.0f;			MAT_ELEM(angles,1,7)=31.7175f;
		MAT_ELEM(angles,0,8)=0.0f;			MAT_ELEM(angles,1,8)=58.2825f;
		MAT_ELEM(angles,0,9)=288.0f;		MAT_ELEM(angles,1,9)=58.2825f;
		MAT_ELEM(angles,0,10)=342.0f;		MAT_ELEM(angles,1,10)=90.0f;
		MAT_ELEM(angles,0,11)=306.0f;		MAT_ELEM(angles,1,11)=90.0f;
		MAT_ELEM(angles,0,12)=72.0f;		MAT_ELEM(angles,1,12)=58.2825f;
		MAT_ELEM(angles,0,13)=18.0f;		MAT_ELEM(angles,1,13)=90.0f;
		MAT_ELEM(angles,0,14)=54.0f;		MAT_ELEM(angles,1,14)=90.0f;
		MAT_ELEM(angles,0,15)=90.0f;		MAT_ELEM(angles,1,15)=90.0f;
		MAT_ELEM(angles,0,16)=216.0f;		MAT_ELEM(angles,1,16)=58.282f;
		MAT_ELEM(angles,0,17)=144.0f;		MAT_ELEM(angles,1,17)=58.282f;
		MAT_ELEM(angles,0,18)=180.0f;		MAT_ELEM(angles,1,18)=31.718f;
		MAT_ELEM(angles,0,19)=252.0f;		MAT_ELEM(angles,1,19)=31.718f;
		MAT_ELEM(angles,0,20)=108.0f;		MAT_ELEM(angles,1,20)=31.718f;
		MAT_ELEM(angles,0,21)=346.3862f;	MAT_ELEM(angles,1,21)=43.6469f;
		MAT_ELEM(angles,0,22)=58.3862f;		MAT_ELEM(angles,1,22)=43.6469f;
		MAT_ELEM(angles,0,23)=274.3862f;	MAT_ELEM(angles,1,23)=43.6469f;
		MAT_ELEM(angles,0,24)=0.0f;			MAT_ELEM(angles,1,24)=90.0f;
		MAT_ELEM(angles,0,25)=72.0f;		MAT_ELEM(angles,1,25)=90.0f;
		MAT_ELEM(angles,0,26)=288.0f;		MAT_ELEM(angles,1,26)=90.0f;
		MAT_ELEM(angles,0,27)=225.7323f;	MAT_ELEM(angles,1,27)=73.955f;
		MAT_ELEM(angles,0,28)=153.7323f;	MAT_ELEM(angles,1,28)=73.955f;
		MAT_ELEM(angles,0,29)=216.0f;		MAT_ELEM(angles,1,29)=26.565f;
		MAT_ELEM(angles,0,30)=144.0f;		MAT_ELEM(angles,1,30)=26.565f;
		MAT_ELEM(angles,0,31)=0.0f;			MAT_ELEM(angles,1,31)=26.5651f;
		MAT_ELEM(angles,0,32)=72.0f;		MAT_ELEM(angles,1,32)=26.5651f;
		MAT_ELEM(angles,0,33)=288.0f;		MAT_ELEM(angles,1,33)=26.5651f;
		MAT_ELEM(angles,0,34)=350.2677f;	MAT_ELEM(angles,1,34)=73.9549f;
		MAT_ELEM(angles,0,35)=62.2677f;		MAT_ELEM(angles,1,35)=73.9549f;
		MAT_ELEM(angles,0,36)=278.2677f;	MAT_ELEM(angles,1,36)=73.9549f;
		MAT_ELEM(angles,0,37)=206.2677f;	MAT_ELEM(angles,1,37)=73.955f;
		MAT_ELEM(angles,0,38)=134.2677f;	MAT_ELEM(angles,1,38)=73.955f;
		MAT_ELEM(angles,0,39)=202.3862f;	MAT_ELEM(angles,1,39)=43.647f;
		MAT_ELEM(angles,0,40)=130.3862f;	MAT_ELEM(angles,1,40)=43.647f;
		MAT_ELEM(angles,0,41)=13.6138f;		MAT_ELEM(angles,1,41)=43.6469f;
		MAT_ELEM(angles,0,42)=85.6138f;		MAT_ELEM(angles,1,42)=43.6469f;
		MAT_ELEM(angles,0,43)=301.6138f;	MAT_ELEM(angles,1,43)=43.6469f;
		MAT_ELEM(angles,0,44)=9.7323f;		MAT_ELEM(angles,1,44)=73.9549f;
		MAT_ELEM(angles,0,45)=81.7323f;		MAT_ELEM(angles,1,45)=73.9549f;
		MAT_ELEM(angles,0,46)=297.7323f;	MAT_ELEM(angles,1,46)=73.9549f;
		MAT_ELEM(angles,0,47)=36.0f;		MAT_ELEM(angles,1,47)=90.0f;
		MAT_ELEM(angles,0,48)=324.0f;		MAT_ELEM(angles,1,48)=90.0f;
		MAT_ELEM(angles,0,49)=229.6138f;	MAT_ELEM(angles,1,49)=43.647f;
		MAT_ELEM(angles,0,50)=157.6138f;	MAT_ELEM(angles,1,50)=43.647f;
		MAT_ELEM(angles,0,51)=324.0f;		MAT_ELEM(angles,1,51)=15.8587f;
		MAT_ELEM(angles,0,52)=36.0f;		MAT_ELEM(angles,1,52)=15.8587f;
		MAT_ELEM(angles,0,53)=341.533f;		MAT_ELEM(angles,1,53)=59.6208f;
		MAT_ELEM(angles,0,54)=306.467f;		MAT_ELEM(angles,1,54)=59.6208f;
		MAT_ELEM(angles,0,55)=333.5057f;	MAT_ELEM(angles,1,55)=76.5584f;
		MAT_ELEM(angles,0,56)=314.4943f;	MAT_ELEM(angles,1,56)=76.5584f;
		MAT_ELEM(angles,0,57)=53.533f;		MAT_ELEM(angles,1,57)=59.6208f;
		MAT_ELEM(angles,0,58)=26.4943f;		MAT_ELEM(angles,1,58)=76.5584f;
		MAT_ELEM(angles,0,59)=45.5057f;		MAT_ELEM(angles,1,59)=76.5584f;
		MAT_ELEM(angles,0,60)=197.533f;		MAT_ELEM(angles,1,60)=59.621f;
		MAT_ELEM(angles,0,61)=162.467f;		MAT_ELEM(angles,1,61)=59.621f;
		MAT_ELEM(angles,0,62)=180.0;		MAT_ELEM(angles,1,62)=47.576;
		MAT_ELEM(angles,0,63)=269.533f;		MAT_ELEM(angles,1,63)=59.621;
		MAT_ELEM(angles,0,64)=252.0f;		MAT_ELEM(angles,1,64)=47.576f;
		MAT_ELEM(angles,0,65)=108.0f;		MAT_ELEM(angles,1,65)=47.576f;
		MAT_ELEM(angles,0,66)=324.0f;		MAT_ELEM(angles,1,66)=47.5762f;
		MAT_ELEM(angles,0,67)=36.0f;		MAT_ELEM(angles,1,67)=47.5762f;
		MAT_ELEM(angles,0,68)=18.467f;		MAT_ELEM(angles,1,68)=59.6208f;
		MAT_ELEM(angles,0,69)=170.4943f;	MAT_ELEM(angles,1,69)=76.558f;
		MAT_ELEM(angles,0,70)=117.5057f;	MAT_ELEM(angles,1,70)=76.558f;
		MAT_ELEM(angles,0,71)=189.5057f;	MAT_ELEM(angles,1,71)=76.558f;
		MAT_ELEM(angles,0,72)=242.4943f;	MAT_ELEM(angles,1,72)=76.558f;
		MAT_ELEM(angles,0,73)=261.5057f;	MAT_ELEM(angles,1,73)=76.558f;
		MAT_ELEM(angles,0,74)=98.4943f;		MAT_ELEM(angles,1,74)=76.558f;
		MAT_ELEM(angles,0,75)=234.467f;		MAT_ELEM(angles,1,75)=59.621f;
		MAT_ELEM(angles,0,76)=125.533f;		MAT_ELEM(angles,1,76)=59.621f;
		MAT_ELEM(angles,0,77)=180.0f;		MAT_ELEM(angles,1,77)=15.859f;
		MAT_ELEM(angles,0,78)=252.0f;		MAT_ELEM(angles,1,78)=15.859f;
		MAT_ELEM(angles,0,79)=90.467f;		MAT_ELEM(angles,1,79)=59.621f;
		MAT_ELEM(angles,0,80)=108.0f;		MAT_ELEM(angles,1,80)=15.859f;
		MAT_ELEM(angles,0,81)=0.0f;			MAT_ELEM(angles,1,81)=42.8321f;
		MAT_ELEM(angles,0,82)=72.0f;		MAT_ELEM(angles,1,82)=42.8321f;
		MAT_ELEM(angles,0,83)=288.0f;		MAT_ELEM(angles,1,83)=42.8321f;
		MAT_ELEM(angles,0,84)=4.7693f;		MAT_ELEM(angles,1,84)=81.9488f;
		MAT_ELEM(angles,0,85)=76.7693f;		MAT_ELEM(angles,1,85)=81.9488f;
		MAT_ELEM(angles,0,86)=292.7693f;	MAT_ELEM(angles,1,86)=81.9488f;
		MAT_ELEM(angles,0,87)=220.7693f;	MAT_ELEM(angles,1,87)=81.9488f;
		MAT_ELEM(angles,0,88)=148.7693f;	MAT_ELEM(angles,1,88)=81.9488f;
		MAT_ELEM(angles,0,89)=224.2677f;	MAT_ELEM(angles,1,89)=34.924f;
		MAT_ELEM(angles,0,90)=152.2677f;	MAT_ELEM(angles,1,90)=34.924f;
		MAT_ELEM(angles,0,91)=13.5146f;		MAT_ELEM(angles,1,91)=20.3172f;
		MAT_ELEM(angles,0,92)=85.5146f;		MAT_ELEM(angles,1,92)=20.3172f;
		MAT_ELEM(angles,0,93)=301.5146f;	MAT_ELEM(angles,1,93)=20.3172f;
		MAT_ELEM(angles,0,94)=346.1363f;	MAT_ELEM(angles,1,94)=66.7276f;
		MAT_ELEM(angles,0,95)=58.1363f;		MAT_ELEM(angles,1,95)=66.7276f;
		MAT_ELEM(angles,0,96)=274.1363f;	MAT_ELEM(angles,1,96)=66.7276f;
		MAT_ELEM(angles,0,97)=197.8362f;	MAT_ELEM(angles,1,97)=75.105f;
		MAT_ELEM(angles,0,98)=269.8362f;	MAT_ELEM(angles,1,98)=75.105f;
		MAT_ELEM(angles,0,99)=125.8362f;	MAT_ELEM(angles,1,99)=75.105f;
		MAT_ELEM(angles,0,100)=199.6899f;	MAT_ELEM(angles,1,100)=51.609f;
		MAT_ELEM(angles,0,101)=127.6899f;	MAT_ELEM(angles,1,101)=51.609f;
		MAT_ELEM(angles,0,102)=334.8124f;	MAT_ELEM(angles,1,102)=45.0621f;
		MAT_ELEM(angles,0,103)=46.8124f;	MAT_ELEM(angles,1,103)=45.0621f;
		MAT_ELEM(angles,0,104)=175.3133f;	MAT_ELEM(angles,1,104)=83.2562f;
		MAT_ELEM(angles,0,105)=247.3133f;	MAT_ELEM(angles,1,105)=83.2562f;
		MAT_ELEM(angles,0,106)=103.3133f;	MAT_ELEM(angles,1,106)=83.2562f;
		MAT_ELEM(angles,0,107)=229.8637f;	MAT_ELEM(angles,1,107)=66.728f;
		MAT_ELEM(angles,0,108)=157.8637f;	MAT_ELEM(angles,1,108)=66.728f;
		MAT_ELEM(angles,0,109)=202.4854f;	MAT_ELEM(angles,1,109)=20.317f;
		MAT_ELEM(angles,0,110)=130.4854f;	MAT_ELEM(angles,1,110)=20.317f;
		MAT_ELEM(angles,0,111)=16.3101f;	MAT_ELEM(angles,1,111)=51.6091f;
		MAT_ELEM(angles,0,112)=88.3101f;	MAT_ELEM(angles,1,112)=51.6091f;
		MAT_ELEM(angles,0,113)=304.3101f;	MAT_ELEM(angles,1,113)=51.6091f;
		MAT_ELEM(angles,0,114)=18.1638f;	MAT_ELEM(angles,1,114)=75.1046f;
		MAT_ELEM(angles,0,115)=306.1638f;	MAT_ELEM(angles,1,115)=75.1046f;
		MAT_ELEM(angles,0,116)=40.6867f;	MAT_ELEM(angles,1,116)=83.2562f;
		MAT_ELEM(angles,0,117)=328.6867f;	MAT_ELEM(angles,1,117)=83.2562f;
		MAT_ELEM(angles,0,118)=241.1876f;	MAT_ELEM(angles,1,118)=45.062f;
		MAT_ELEM(angles,0,119)=97.1876f;	MAT_ELEM(angles,1,119)=45.062f;
		MAT_ELEM(angles,0,120)=169.1876f;	MAT_ELEM(angles,1,120)=45.062f;
		MAT_ELEM(angles,0,121)=351.7323f;	MAT_ELEM(angles,1,121)=34.9243f;
		MAT_ELEM(angles,0,122)=63.7323f;	MAT_ELEM(angles,1,122)=34.9243f;
		MAT_ELEM(angles,0,123)=279.7323f;	MAT_ELEM(angles,1,123)=34.9243f;
		MAT_ELEM(angles,0,124)=355.2307f;	MAT_ELEM(angles,1,124)=81.9488f;
		MAT_ELEM(angles,0,125)=67.2307f;	MAT_ELEM(angles,1,125)=81.9488f;
		MAT_ELEM(angles,0,126)=283.2307f;	MAT_ELEM(angles,1,126)=81.9488f;
		MAT_ELEM(angles,0,127)=216.0f;		MAT_ELEM(angles,1,127)=73.733f;
		MAT_ELEM(angles,0,128)=144.0f;		MAT_ELEM(angles,1,128)=73.733f;
		MAT_ELEM(angles,0,129)=207.7323f;	MAT_ELEM(angles,1,129)=34.924f;
		MAT_ELEM(angles,0,130)=135.7323f;	MAT_ELEM(angles,1,130)=34.924f;
		MAT_ELEM(angles,0,131)=346.4854f;	MAT_ELEM(angles,1,131)=20.3172f;
		MAT_ELEM(angles,0,132)=58.4854f;	MAT_ELEM(angles,1,132)=20.3172f;
		MAT_ELEM(angles,0,133)=274.4854f;	MAT_ELEM(angles,1,133)=20.3172f;
		MAT_ELEM(angles,0,134)=341.8362f;	MAT_ELEM(angles,1,134)=75.1046f;
		MAT_ELEM(angles,0,135)=53.8362f;	MAT_ELEM(angles,1,135)=75.1046f;
		MAT_ELEM(angles,0,136)=202.1363f;	MAT_ELEM(angles,1,136)=66.728f;
		MAT_ELEM(angles,0,137)=130.1363f;	MAT_ELEM(angles,1,137)=66.728f;
		MAT_ELEM(angles,0,138)=190.8124f;	MAT_ELEM(angles,1,138)=45.062f;
		MAT_ELEM(angles,0,139)=262.8124f;	MAT_ELEM(angles,1,139)=45.062f;
		MAT_ELEM(angles,0,140)=118.8124f;	MAT_ELEM(angles,1,140)=45.062f;
		MAT_ELEM(angles,0,141)=343.6899f;	MAT_ELEM(angles,1,141)=51.6091f;
		MAT_ELEM(angles,0,142)=55.6899f;	MAT_ELEM(angles,1,142)=51.6091f;
		MAT_ELEM(angles,0,143)=271.6899f;	MAT_ELEM(angles,1,143)=51.6091f;
		MAT_ELEM(angles,0,144)=184.6867f;	MAT_ELEM(angles,1,144)=83.2562f;
		MAT_ELEM(angles,0,145)=256.6867f;	MAT_ELEM(angles,1,145)=83.2562f;
		MAT_ELEM(angles,0,146)=112.6867f;	MAT_ELEM(angles,1,146)=83.2562f;
		MAT_ELEM(angles,0,147)=234.1638f;	MAT_ELEM(angles,1,147)=75.105f;
		MAT_ELEM(angles,0,148)=90.1638f;	MAT_ELEM(angles,1,148)=75.105f;
		MAT_ELEM(angles,0,149)=162.1638f;	MAT_ELEM(angles,1,149)=75.105f;
		MAT_ELEM(angles,0,150)=229.5146f;	MAT_ELEM(angles,1,150)=20.317f;
		MAT_ELEM(angles,0,151)=157.5146f;	MAT_ELEM(angles,1,151)=20.317f;
		MAT_ELEM(angles,0,152)=25.1876f;	MAT_ELEM(angles,1,152)=45.0621f;
		MAT_ELEM(angles,0,153)=313.1876f;	MAT_ELEM(angles,1,153)=45.0621f;
		MAT_ELEM(angles,0,154)=13.8637f;	MAT_ELEM(angles,1,154)=66.7276f;
		MAT_ELEM(angles,0,155)=85.8637f;	MAT_ELEM(angles,1,155)=66.7276f;
		MAT_ELEM(angles,0,156)=301.8637f;	MAT_ELEM(angles,1,156)=66.7276f;
		MAT_ELEM(angles,0,157)=31.3133f;	MAT_ELEM(angles,1,157)=83.2562f;
		MAT_ELEM(angles,0,158)=319.3133f;	MAT_ELEM(angles,1,158)=83.2562f;
		MAT_ELEM(angles,0,159)=232.3101f;	MAT_ELEM(angles,1,159)=51.609f;
		MAT_ELEM(angles,0,160)=160.3101f;	MAT_ELEM(angles,1,160)=51.609f;
		MAT_ELEM(angles,0,161)=8.2677f;		MAT_ELEM(angles,1,161)=34.9243f;
		MAT_ELEM(angles,0,162)=80.2677f;	MAT_ELEM(angles,1,162)=34.9243f;
		MAT_ELEM(angles,0,163)=296.2677f;	MAT_ELEM(angles,1,163)=34.9243f;
		MAT_ELEM(angles,0,164)=0.0f;		MAT_ELEM(angles,1,164)=73.733f;
		MAT_ELEM(angles,0,165)=72.0f;			MAT_ELEM(angles,1,165)=73.733f;
		MAT_ELEM(angles,0,166)=288.0f;		MAT_ELEM(angles,1,166)=73.733f;
		MAT_ELEM(angles,0,167)=211.2307f;	MAT_ELEM(angles,1,167)=81.9488f;
		MAT_ELEM(angles,0,168)=139.2307f;	MAT_ELEM(angles,1,168)=81.9488f;
		MAT_ELEM(angles,0,169)=216.0f;		MAT_ELEM(angles,1,169)=42.832f;
		MAT_ELEM(angles,0,170)=144.0f;		MAT_ELEM(angles,1,170)=42.832f;
		MAT_ELEM(angles,0,171)=0.0f;		MAT_ELEM(angles,1,171)=12.9432f;
		MAT_ELEM(angles,0,172)=72.0f;		MAT_ELEM(angles,1,172)=12.9432f;
		MAT_ELEM(angles,0,173)=288.0f;		MAT_ELEM(angles,1,173)=12.9432f;
		MAT_ELEM(angles,0,174)=337.2786f;	MAT_ELEM(angles,1,174)=68.041f;
		MAT_ELEM(angles,0,175)=49.2786f; 	MAT_ELEM(angles,1,175)=68.041f;
		MAT_ELEM(angles,0,176)=193.2786f;	MAT_ELEM(angles,1,176)=68.041f;
		MAT_ELEM(angles,0,177)=265.2786f;	MAT_ELEM(angles,1,177)=68.041f;
		MAT_ELEM(angles,0,178)=121.2786f;	MAT_ELEM(angles,1,178)=68.041f;
		MAT_ELEM(angles,0,179)=189.4537f;	MAT_ELEM(angles,1,179)=53.278f;
		MAT_ELEM(angles,0,180)=261.4537f;	MAT_ELEM(angles,1,180)=53.278f;
		MAT_ELEM(angles,0,181)=117.4537f;	MAT_ELEM(angles,1,181)=53.278f;
		MAT_ELEM(angles,0,182)=333.4537f;	MAT_ELEM(angles,1,182)=53.2783f;
		MAT_ELEM(angles,0,183)=45.4537f; 	MAT_ELEM(angles,1,183)=53.2783f;
		MAT_ELEM(angles,0,184)=180.0f;		MAT_ELEM(angles,1,184)=76.378f;
		MAT_ELEM(angles,0,185)=252.0f;		MAT_ELEM(angles,1,185)=76.378f;
		MAT_ELEM(angles,0,186)=108.0f;		MAT_ELEM(angles,1,186)=76.378f;
		MAT_ELEM(angles,0,187)=238.7214f;	MAT_ELEM(angles,1,187)=68.041f;
		MAT_ELEM(angles,0,188)=94.7214f;	MAT_ELEM(angles,1,188)=68.041f;
		MAT_ELEM(angles,0,189)=166.7214f;	MAT_ELEM(angles,1,189)=68.041f;
		MAT_ELEM(angles,0,190)=216.0f;		MAT_ELEM(angles,1,190)=12.943f;
		MAT_ELEM(angles,0,191)=144.0f;		MAT_ELEM(angles,1,191)=12.943f;
		MAT_ELEM(angles,0,192)=26.5463f;	MAT_ELEM(angles,1,192)=53.2783f;
		MAT_ELEM(angles,0,193)=314.5463f;	MAT_ELEM(angles,1,193)=53.2783f;
		MAT_ELEM(angles,0,194)=22.7214f;	MAT_ELEM(angles,1,194)=68.041f;
		MAT_ELEM(angles,0,195)=310.7214f;	MAT_ELEM(angles,1,195)=68.041f;
		MAT_ELEM(angles,0,196)=36.0f;		MAT_ELEM(angles,1,196)=76.3782f;
		MAT_ELEM(angles,0,197)=324.0f;		MAT_ELEM(angles,1,197)=76.3782f;
		MAT_ELEM(angles,0,198)=242.5463f;	MAT_ELEM(angles,1,198)=53.278f;
		MAT_ELEM(angles,0,199)=98.5463f;	MAT_ELEM(angles,1,199)=53.278f;
		MAT_ELEM(angles,0,200)=170.5463f;	MAT_ELEM(angles,1,200)=53.278f;
		MAT_ELEM(angles,0,201)=336.7264f;	MAT_ELEM(angles,1,201)=37.1611f;
		MAT_ELEM(angles,0,202)=48.7264f;	MAT_ELEM(angles,1,202)=37.1611f;
		MAT_ELEM(angles,0,203)=351.0f;		MAT_ELEM(angles,1,203)=90.0f;
		MAT_ELEM(angles,0,204)=63.0f;		MAT_ELEM(angles,1,204)=90.0f;
		MAT_ELEM(angles,0,205)=279.0f;		MAT_ELEM(angles,1,205)=90.0f;
		MAT_ELEM(angles,0,206)=221.1634f;	MAT_ELEM(angles,1,206)=66.042f;
		MAT_ELEM(angles,0,207)=149.1634f;	MAT_ELEM(angles,1,207)=66.042f;
		MAT_ELEM(angles,0,208)=196.498f;	MAT_ELEM(angles,1,208)=27.943f;
		MAT_ELEM(angles,0,209)=268.498f;	MAT_ELEM(angles,1,209)=27.943f;
		MAT_ELEM(angles,0,210)=124.498f;	MAT_ELEM(angles,1,210)=27.943f;
		MAT_ELEM(angles,0,211)=340.498f;	MAT_ELEM(angles,1,211)=27.9429f;
		MAT_ELEM(angles,0,212)=52.498f;		MAT_ELEM(angles,1,212)=27.9429f;
		MAT_ELEM(angles,0,213)=346.0516f;	MAT_ELEM(angles,1,213)=81.9568f;
		MAT_ELEM(angles,0,214)=58.0516f;	MAT_ELEM(angles,1,214)=81.9568f;
		MAT_ELEM(angles,0,215)=274.0516f;	MAT_ELEM(angles,1,215)=81.9568f;
		MAT_ELEM(angles,0,216)=210.8366f;	MAT_ELEM(angles,1,216)=66.042f;
		MAT_ELEM(angles,0,217)=138.8366f;	MAT_ELEM(angles,1,217)=66.042f;
		MAT_ELEM(angles,0,218)=192.7264f;	MAT_ELEM(angles,1,218)=37.161f;
		MAT_ELEM(angles,0,219)=264.7264f;	MAT_ELEM(angles,1,219)=37.161f;
		MAT_ELEM(angles,0,220)=120.7264f;	MAT_ELEM(angles,1,220)=37.161f;
		MAT_ELEM(angles,0,221)=6.0948f;		MAT_ELEM(angles,1,221)=50.7685f;
		MAT_ELEM(angles,0,222)=78.0948f;	MAT_ELEM(angles,1,222)=50.7685f;
		MAT_ELEM(angles,0,223)=294.0948f;	MAT_ELEM(angles,1,223)=50.7685f;
		MAT_ELEM(angles,0,224)=13.9484f;	MAT_ELEM(angles,1,224)=81.9568f;
		MAT_ELEM(angles,0,225)=85.9484f;	MAT_ELEM(angles,1,225)=81.9568f;
		MAT_ELEM(angles,0,226)=301.9484f;	MAT_ELEM(angles,1,226)=81.9568f;
		MAT_ELEM(angles,0,227)=45.0f;		MAT_ELEM(angles,1,227)=90.0f;
		MAT_ELEM(angles,0,228)=333.0f;		MAT_ELEM(angles,1,228)=90.0f;
		MAT_ELEM(angles,0,229)=239.2736f;	MAT_ELEM(angles,1,229)=37.161f;
		MAT_ELEM(angles,0,230)=95.2736f;	MAT_ELEM(angles,1,230)=37.161f;
		MAT_ELEM(angles,0,231)=167.2736f;	MAT_ELEM(angles,1,231)=37.161f;
		MAT_ELEM(angles,0,232)=324.0f;		MAT_ELEM(angles,1,232)=7.9294f;
		MAT_ELEM(angles,0,233)=36.0f;		MAT_ELEM(angles,1,233)=7.9294f;
		MAT_ELEM(angles,0,234)=332.6069f;	MAT_ELEM(angles,1,234)=61.2449f;
		MAT_ELEM(angles,0,235)=315.3931f;	MAT_ELEM(angles,1,235)=61.2449f;
		MAT_ELEM(angles,0,236)=328.9523f;	MAT_ELEM(angles,1,236)=69.9333f;
		MAT_ELEM(angles,0,237)=319.0477f;	MAT_ELEM(angles,1,237)=69.9333f;
		MAT_ELEM(angles,0,238)=44.6069f;	MAT_ELEM(angles,1,238)=61.2449f;
		MAT_ELEM(angles,0,239)=31.0477f;	MAT_ELEM(angles,1,239)=69.9333f;
		MAT_ELEM(angles,0,240)=40.9523f;	MAT_ELEM(angles,1,240)=69.9333f;
		MAT_ELEM(angles,0,241)=188.6069f;	MAT_ELEM(angles,1,241)=61.245f;
		MAT_ELEM(angles,0,242)=171.3931f;	MAT_ELEM(angles,1,242)=61.245f;
		MAT_ELEM(angles,0,243)=180.0f;		MAT_ELEM(angles,1,243)=55.506f;
		MAT_ELEM(angles,0,244)=260.6069f;	MAT_ELEM(angles,1,244)=61.245f;
		MAT_ELEM(angles,0,245)=252.0f;		MAT_ELEM(angles,1,245)=55.506f;
		MAT_ELEM(angles,0,246)=108.0f;		MAT_ELEM(angles,1,246)=55.506f;
		MAT_ELEM(angles,0,247)=324.0f;		MAT_ELEM(angles,1,247)=39.6468f;
		MAT_ELEM(angles,0,248)=36.0f;		MAT_ELEM(angles,1,248)=39.6468f;
		MAT_ELEM(angles,0,249)=9.299f;		MAT_ELEM(angles,1,249)=58.6205f;
		MAT_ELEM(angles,0,250)=278.701f;	MAT_ELEM(angles,1,250)=58.6205f;
		MAT_ELEM(angles,0,251)=166.1881f;	MAT_ELEM(angles,1,251)=83.2609f;
		MAT_ELEM(angles,0,252)=121.8119f;	MAT_ELEM(angles,1,252)=83.2609f;
		MAT_ELEM(angles,0,253)=81.299f;		MAT_ELEM(angles,1,253)=58.6205f;
		MAT_ELEM(angles,0,254)=193.8119f;	MAT_ELEM(angles,1,254)=83.2609f;
		MAT_ELEM(angles,0,255)=238.1881f;	MAT_ELEM(angles,1,255)=83.2609f;
		MAT_ELEM(angles,0,256)=265.8119f;	MAT_ELEM(angles,1,256)=83.2609f;
		MAT_ELEM(angles,0,257)=94.1881f;	MAT_ELEM(angles,1,257)=83.2609f;
		MAT_ELEM(angles,0,258)=225.299f;	MAT_ELEM(angles,1,258)=58.621f;
		MAT_ELEM(angles,0,259)=134.701f;	MAT_ELEM(angles,1,259)=58.621f;
		MAT_ELEM(angles,0,260)=180.0f;		MAT_ELEM(angles,1,260)=23.788f;
		MAT_ELEM(angles,0,261)=252.0f;		MAT_ELEM(angles,1,261)=23.788f;
		MAT_ELEM(angles,0,262)=108.0f;		MAT_ELEM(angles,1,262)=23.788f;
		MAT_ELEM(angles,0,263)=353.9052f;	MAT_ELEM(angles,1,263)=50.7685f;
		MAT_ELEM(angles,0,264)=65.9052f; 	MAT_ELEM(angles,1,264)=50.7685f;
		MAT_ELEM(angles,0,265)=281.9052f;	MAT_ELEM(angles,1,265)=50.7685f;
		MAT_ELEM(angles,0,266)=9.0f;		MAT_ELEM(angles,1,266)=90.0f;
		MAT_ELEM(angles,0,267)=81.0f;		MAT_ELEM(angles,1,267)=90.0f;
		MAT_ELEM(angles,0,268)=297.0f;		MAT_ELEM(angles,1,268)=90.0f;
		MAT_ELEM(angles,0,269)=229.9484f;	MAT_ELEM(angles,1,269)=81.9568f;
		MAT_ELEM(angles,0,270)=157.9484f;	MAT_ELEM(angles,1,270)=81.9568f;
		MAT_ELEM(angles,0,271)=235.502f;	MAT_ELEM(angles,1,271)=27.943f;
		MAT_ELEM(angles,0,272)=91.502f;		MAT_ELEM(angles,1,272)=27.943f;
		MAT_ELEM(angles,0,273)=163.502f;	MAT_ELEM(angles,1,273)=27.943f;
		MAT_ELEM(angles,0,274)=19.502f;		MAT_ELEM(angles,1,274)=27.9429f;
		MAT_ELEM(angles,0,275)=307.502f;	MAT_ELEM(angles,1,275)=27.9429f;
		MAT_ELEM(angles,0,276)=354.8366f;	MAT_ELEM(angles,1,276)=66.0423f;
		MAT_ELEM(angles,0,277)=66.8366f;	MAT_ELEM(angles,1,277)=66.0423f;
		MAT_ELEM(angles,0,278)=282.8366f;	MAT_ELEM(angles,1,278)=66.0423f;
		MAT_ELEM(angles,0,279)=202.0516f;	MAT_ELEM(angles,1,279)=81.9568f;
		MAT_ELEM(angles,0,280)=130.0516f;	MAT_ELEM(angles,1,280)=81.9568f;
		MAT_ELEM(angles,0,281)=209.9052f;	MAT_ELEM(angles,1,281)=50.768f;
		MAT_ELEM(angles,0,282)=137.9052f;	MAT_ELEM(angles,1,282)=50.768f;
		MAT_ELEM(angles,0,283)=23.2736f;	MAT_ELEM(angles,1,283)=37.1611f;
		MAT_ELEM(angles,0,284)=311.2736f;	MAT_ELEM(angles,1,284)=37.1611f;
		MAT_ELEM(angles,0,285)=5.1634f;		MAT_ELEM(angles,1,285)=66.0423f;
		MAT_ELEM(angles,0,286)=77.1634f;	MAT_ELEM(angles,1,286)=66.0423f;
		MAT_ELEM(angles,0,287)=293.1634f;	MAT_ELEM(angles,1,287)=66.0423f;
		MAT_ELEM(angles,0,288)=27.0f;		MAT_ELEM(angles,1,288)=90.0f;
		MAT_ELEM(angles,0,289)=315.0f;		MAT_ELEM(angles,1,289)=90.0f;
		MAT_ELEM(angles,0,290)=222.0948f;	MAT_ELEM(angles,1,290)=50.768f;
		MAT_ELEM(angles,0,291)=150.0948f;	MAT_ELEM(angles,1,291)=50.768f;
		MAT_ELEM(angles,0,292)=324.0f;		MAT_ELEM(angles,1,292)=23.7881f;
		MAT_ELEM(angles,0,293)=36.0f;		MAT_ELEM(angles,1,293)=23.7881f;
		MAT_ELEM(angles,0,294)=350.701f;	MAT_ELEM(angles,1,294)=58.6205f;
		MAT_ELEM(angles,0,295)=297.299f;	MAT_ELEM(angles,1,295)=58.6205f;
		MAT_ELEM(angles,0,296)=337.8119f;	MAT_ELEM(angles,1,296)=83.2609f;
		MAT_ELEM(angles,0,297)=310.1881f;	MAT_ELEM(angles,1,297)=83.2609f;
		MAT_ELEM(angles,0,298)=62.701f;		MAT_ELEM(angles,1,298)=58.6205f;
		MAT_ELEM(angles,0,299)=22.1881f;	MAT_ELEM(angles,1,299)=83.2609f;
		MAT_ELEM(angles,0,300)=49.8119f;	MAT_ELEM(angles,1,300)=83.2609f;
		MAT_ELEM(angles,0,301)=206.701f;	MAT_ELEM(angles,1,301)=58.621f;
		MAT_ELEM(angles,0,302)=153.299f;	MAT_ELEM(angles,1,302)=58.621f;
		MAT_ELEM(angles,0,303)=180.0f;		MAT_ELEM(angles,1,303)=39.647f;
		MAT_ELEM(angles,0,304)=252.0f;		MAT_ELEM(angles,1,304)=39.647f;
		MAT_ELEM(angles,0,305)=108.0f;		MAT_ELEM(angles,1,305)=39.647f;
		MAT_ELEM(angles,0,306)=324.0f;		MAT_ELEM(angles,1,306)=55.5056f;
		MAT_ELEM(angles,0,307)=36.0f;		MAT_ELEM(angles,1,307)=55.5056f;
		MAT_ELEM(angles,0,308)=27.3931f;	MAT_ELEM(angles,1,308)=61.2449f;
		MAT_ELEM(angles,0,309)=175.0477f;	MAT_ELEM(angles,1,309)=69.933f;
		MAT_ELEM(angles,0,310)=112.9523f;	MAT_ELEM(angles,1,310)=69.933f;
		MAT_ELEM(angles,0,311)=184.9523f;	MAT_ELEM(angles,1,311)=69.933f;
		MAT_ELEM(angles,0,312)=247.0477f;	MAT_ELEM(angles,1,312)=69.933f;
		MAT_ELEM(angles,0,313)=256.9523f;	MAT_ELEM(angles,1,313)=69.933f;
		MAT_ELEM(angles,0,314)=103.0477f;	MAT_ELEM(angles,1,314)=69.933f;
		MAT_ELEM(angles,0,315)=243.3931f;	MAT_ELEM(angles,1,315)=61.245f;
		MAT_ELEM(angles,0,316)=116.6069f;	MAT_ELEM(angles,1,316)=61.245f;
		MAT_ELEM(angles,0,317)=180.0f;		MAT_ELEM(angles,1,317)=7.929f;
		MAT_ELEM(angles,0,318)=252.0f;		MAT_ELEM(angles,1,318)=7.929f;
		MAT_ELEM(angles,0,319)=99.3931f;	MAT_ELEM(angles,1,319)=61.245f;
		MAT_ELEM(angles,0,320)=108.0f;		MAT_ELEM(angles,1,320)=7.929f;
	}
	else
	{
		angles.initZeros(2,81);
		MAT_ELEM(angles, 0, 0) = 0.000000f;	 	 MAT_ELEM(angles, 1, 0) = 0.000000f;
		MAT_ELEM(angles, 0, 1) = 36.000000f;	 MAT_ELEM(angles, 1, 1) = 15.858741f;
		MAT_ELEM(angles, 0, 2) = 36.000000f;	 MAT_ELEM(angles, 1, 2) = 31.717482f;
		MAT_ELEM(angles, 0, 3) = 36.000000f;	 MAT_ELEM(angles, 1, 3) = 47.576224f;
		MAT_ELEM(angles, 0, 4) = 36.000000f;	 MAT_ELEM(angles, 1, 4) = 63.434965f;
		MAT_ELEM(angles, 0, 5) = 62.494295f;	 MAT_ELEM(angles, 1, 5) = -76.558393f;
		MAT_ELEM(angles, 0, 6) = 54.000000f;	 MAT_ELEM(angles, 1, 6) = 90.000000f;
		MAT_ELEM(angles, 0, 7) = 45.505705f;	 MAT_ELEM(angles, 1, 7) = 76.558393f;
		MAT_ELEM(angles, 0, 8) = 108.000000f;	 MAT_ELEM(angles, 1, 8) = 15.858741f;
		MAT_ELEM(angles, 0, 9) = 108.000000f;	 MAT_ELEM(angles, 1, 9) = 31.717482f;
		MAT_ELEM(angles, 0, 10) = 108.000000f;	 MAT_ELEM(angles, 1, 10) = 47.576224f;
		MAT_ELEM(angles, 0, 11) = 108.000000f;	 MAT_ELEM(angles, 1, 11) = 63.434965f;
		MAT_ELEM(angles, 0, 12) = 134.494295f;	 MAT_ELEM(angles, 1, 12) = -76.558393f;
		MAT_ELEM(angles, 0, 13) = 126.000000f;	 MAT_ELEM(angles, 1, 13) = 90.000000f;
		MAT_ELEM(angles, 0, 14) = 117.505705f;	 MAT_ELEM(angles, 1, 14) = 76.558393f;
		MAT_ELEM(angles, 0, 15) = 144.000000f;	 MAT_ELEM(angles, 1, 15) = -15.858741f;
		MAT_ELEM(angles, 0, 16) = 144.000000f;	 MAT_ELEM(angles, 1, 16) = -31.717482f;
		MAT_ELEM(angles, 0, 17) = 144.000000f;	 MAT_ELEM(angles, 1, 17) = -47.576224f;
		MAT_ELEM(angles, 0, 18) = 144.000000f;	 MAT_ELEM(angles, 1, 18) = -63.434965f;
		MAT_ELEM(angles, 0, 19) = 170.494295f;	 MAT_ELEM(angles, 1, 19) = 76.558393f;
		MAT_ELEM(angles, 0, 20) = 162.000000f;	 MAT_ELEM(angles, 1, 20) = 90.000000f;
		MAT_ELEM(angles, 0, 21) = 153.505705f;	 MAT_ELEM(angles, 1, 21) = -76.558393f;
		MAT_ELEM(angles, 0, 22) = 72.000000f;	 MAT_ELEM(angles, 1, 22) = -15.858741f;
		MAT_ELEM(angles, 0, 23) = 72.000000f;	 MAT_ELEM(angles, 1, 23) = -31.717482f;
		MAT_ELEM(angles, 0, 24) = 72.000000f;	 MAT_ELEM(angles, 1, 24) = -47.576224f;
		MAT_ELEM(angles, 0, 25) = 72.000000f;	 MAT_ELEM(angles, 1, 25) = -63.434965f;
		MAT_ELEM(angles, 0, 26) = 98.494295f;	 MAT_ELEM(angles, 1, 26) = 76.558393f;
		MAT_ELEM(angles, 0, 27) = 90.000000f;	 MAT_ELEM(angles, 1, 27) = 90.000000f;
		MAT_ELEM(angles, 0, 28) = 81.505705f;	 MAT_ELEM(angles, 1, 28) = -76.558393f;
		MAT_ELEM(angles, 0, 29) = 0.000000f;	 MAT_ELEM(angles, 1, 29) = -15.858741f;
		MAT_ELEM(angles, 0, 30) = 0.000000f;	 MAT_ELEM(angles, 1, 30) = -31.717482f;
		MAT_ELEM(angles, 0, 31) = 0.000000f;	 MAT_ELEM(angles, 1, 31) = -47.576224f;
		MAT_ELEM(angles, 0, 32) = 0.000000f;	 MAT_ELEM(angles, 1, 32) = -63.434965f;
		MAT_ELEM(angles, 0, 33) = 26.494295f;	 MAT_ELEM(angles, 1, 33) = 76.558393f;
		MAT_ELEM(angles, 0, 34) = 18.000000f;	 MAT_ELEM(angles, 1, 34) = 90.000000f;
		MAT_ELEM(angles, 0, 35) = 9.505705f;	 MAT_ELEM(angles, 1, 35) = -76.558393f;
		MAT_ELEM(angles, 0, 36) = 12.811021f;	 MAT_ELEM(angles, 1, 36) = 42.234673f;
		MAT_ELEM(angles, 0, 37) = 18.466996f;	 MAT_ELEM(angles, 1, 37) = 59.620797f;
		MAT_ELEM(angles, 0, 38) = 0.000000f;	 MAT_ELEM(angles, 1, 38) = 90.000000f;
		MAT_ELEM(angles, 0, 39) = 8.867209f;	 MAT_ELEM(angles, 1, 39) = 75.219088f;
		MAT_ELEM(angles, 0, 40) = 72.000000f;	 MAT_ELEM(angles, 1, 40) = 26.565058f;
		MAT_ELEM(angles, 0, 41) = 59.188979f;	 MAT_ELEM(angles, 1, 41) = 42.234673f;
		MAT_ELEM(angles, 0, 42) = 84.811021f;	 MAT_ELEM(angles, 1, 42) = 42.234673f;
		MAT_ELEM(angles, 0, 43) = 53.533003f;	 MAT_ELEM(angles, 1, 43) = 59.620797f;
		MAT_ELEM(angles, 0, 44) = 72.000000f;	 MAT_ELEM(angles, 1, 44) = 58.282544f;
		MAT_ELEM(angles, 0, 45) = 90.466996f;	 MAT_ELEM(angles, 1, 45) = 59.620797f;
		MAT_ELEM(angles, 0, 46) = 72.000000f;	 MAT_ELEM(angles, 1, 46) = 90.000000f;
		MAT_ELEM(angles, 0, 47) = 63.132791f;	 MAT_ELEM(angles, 1, 47) = 75.219088f;
		MAT_ELEM(angles, 0, 48) = 80.867209f;	 MAT_ELEM(angles, 1, 48) = 75.219088f;
		MAT_ELEM(angles, 0, 49) = 144.000000f;	 MAT_ELEM(angles, 1, 49) = 26.565058f;
		MAT_ELEM(angles, 0, 50) = 131.188979f;	 MAT_ELEM(angles, 1, 50) = 42.234673f;
		MAT_ELEM(angles, 0, 51) = 156.811021f;	 MAT_ELEM(angles, 1, 51) = 42.234673f;
		MAT_ELEM(angles, 0, 52) = 125.533003f;	 MAT_ELEM(angles, 1, 52) = 59.620797f;
		MAT_ELEM(angles, 0, 53) = 144.000000f;	 MAT_ELEM(angles, 1, 53) = 58.282544f;
		MAT_ELEM(angles, 0, 54) = 162.466996f;	 MAT_ELEM(angles, 1, 54) = 59.620797f;
		MAT_ELEM(angles, 0, 55) = 144.000000f;	 MAT_ELEM(angles, 1, 55) = 90.000000f;
		MAT_ELEM(angles, 0, 56) = 135.132791f;	 MAT_ELEM(angles, 1, 56) = 75.219088f;
		MAT_ELEM(angles, 0, 57) = 152.867209f;	 MAT_ELEM(angles, 1, 57) = 75.219088f;
		MAT_ELEM(angles, 0, 58) = 180.000000f;	 MAT_ELEM(angles, 1, 58) = -26.565058f;
		MAT_ELEM(angles, 0, 59) = 167.188979f;	 MAT_ELEM(angles, 1, 59) = -42.234673f;
		MAT_ELEM(angles, 0, 60) = 180.000000f;	 MAT_ELEM(angles, 1, 60) = -58.282544f;
		MAT_ELEM(angles, 0, 61) = 161.533003f;	 MAT_ELEM(angles, 1, 61) = -59.620797f;
		MAT_ELEM(angles, 0, 62) = 171.132791f;	 MAT_ELEM(angles, 1, 62) = -75.219088f;
		MAT_ELEM(angles, 0, 63) = 108.000000f;	 MAT_ELEM(angles, 1, 63) = -26.565058f;
		MAT_ELEM(angles, 0, 64) = 120.811021f;	 MAT_ELEM(angles, 1, 64) = -42.234673f;
		MAT_ELEM(angles, 0, 65) = 95.188979f;	 MAT_ELEM(angles, 1, 65) = -42.234673f;
		MAT_ELEM(angles, 0, 66) = 126.466996f;	 MAT_ELEM(angles, 1, 66) = -59.620797f;
		MAT_ELEM(angles, 0, 67) = 108.000000f;	 MAT_ELEM(angles, 1, 67) = -58.282544f;
		MAT_ELEM(angles, 0, 68) = 89.533003f;	 MAT_ELEM(angles, 1, 68) = -59.620797f;
		MAT_ELEM(angles, 0, 69) = 108.000000f;	 MAT_ELEM(angles, 1, 69) = 90.000000f;
		MAT_ELEM(angles, 0, 70) = 116.867209f;	 MAT_ELEM(angles, 1, 70) = -75.219088f;
		MAT_ELEM(angles, 0, 71) = 99.132791f;	 MAT_ELEM(angles, 1, 71) = -75.219088f;
		MAT_ELEM(angles, 0, 72) = 36.000000f;	 MAT_ELEM(angles, 1, 72) = -26.565058f;
		MAT_ELEM(angles, 0, 73) = 48.811021f;	 MAT_ELEM(angles, 1, 73) = -42.234673f;
		MAT_ELEM(angles, 0, 74) = 23.188979f;	 MAT_ELEM(angles, 1, 74) = -42.234673f;
		MAT_ELEM(angles, 0, 75) = 54.466996f;	 MAT_ELEM(angles, 1, 75) = -59.620797f;
		MAT_ELEM(angles, 0, 76) = 36.000000f;	 MAT_ELEM(angles, 1, 76) = -58.282544f;
		MAT_ELEM(angles, 0, 77) = 17.533003f;	 MAT_ELEM(angles, 1, 77) = -59.620797f;
		MAT_ELEM(angles, 0, 78) = 36.000000f;	 MAT_ELEM(angles, 1, 78) = 90.000000f;
		MAT_ELEM(angles, 0, 79) = 44.867209f;	 MAT_ELEM(angles, 1, 79) = -75.219088f;
		MAT_ELEM(angles, 0, 80) = 27.132791f;	 MAT_ELEM(angles, 1, 80) = -75.219088f;
	}

	angles *= PI/180.0;
}


void ProgAngResAlign::readData(MultidimArray<double> &half1, MultidimArray<double> &half2)
{
	std::cout << "Reading data..." << std::endl;
	Image<double> imgHalf1, imgHalf2, inMask;
	imgHalf1.read(fnhalf1);
	imgHalf2.read(fnhalf2);

	half1 = imgHalf1();
	half2 = imgHalf2();

	maxRadius = XSIZE(half1)*0.5;

	if (fnmask!= "")
	{
		Image<double> inMask;
		inMask.read(fnmask);
		auto mapMask = inMask();

		// Check maximum radius
		size_t xdim = XSIZE(mapMask)*0.5;
		size_t ydim = YSIZE(mapMask)*0.5;
		size_t zdim = ZSIZE(mapMask)*0.5;

		size_t Nelems;
		if (isHelix)
		{
			Nelems = round(sqrt(xdim*xdim + ydim*ydim)+1);
		}
		else
		{
			Nelems = round(sqrt(xdim*xdim + ydim*ydim + zdim*zdim)+1);
		}

		MultidimArray<double> radiusElems, rr, radiusTheoretical;
		radiusElems.initZeros(Nelems);
		rr.initZeros(Nelems);
		radiusTheoretical.initZeros(Nelems);
		

		long n=0;
		if (isHelix)
		{
			for(size_t k=0; k<ZSIZE(mapMask); ++k)
			{
				for(size_t i=0; i<YSIZE(mapMask); ++i)
				{
					size_t yy = (i-ydim);
					yy *= yy;
					
					for(size_t j=0; j<XSIZE(mapMask); ++j)
					{
						size_t xx = (j-xdim);
						xx *= xx;
						int rad = round(sqrt(xx + yy));
						auto idx = (int) round(rad);
						dAi(radiusTheoretical, idx) += 1;

						if (DIRECT_MULTIDIM_ELEM(mapMask, n)>0)
						{
				
							dAi(radiusElems, idx) += 1;
							
							dAi(rr, idx) = rad;
						}
						++n;
					}
				}
			}
		}
		else
		{
			for(size_t k=0; k<ZSIZE(mapMask); ++k)
			{
				size_t zz = (k-zdim);
				zz *= zz;
				
				for(size_t i=0; i<YSIZE(mapMask); ++i)
				{
					size_t yy = (i-ydim);
					yy *= yy;
					
					for(size_t j=0; j<XSIZE(mapMask); ++j)
					{
						size_t xx = (j-xdim);
						xx *= xx;
						int rad = round(sqrt(xx + yy + zz));
						auto idx = (int) round(rad);
						dAi(radiusTheoretical, idx) += 1;

						if (DIRECT_MULTIDIM_ELEM(mapMask, n)>0)
						{
				
							dAi(radiusElems, idx) += 1;
							
							dAi(rr, idx) = rad;
						}
						++n;
					}
				}
			}
		}
		

		if (limRad)
		{
			double radiusThreshold;

			radiusThreshold = 0.75;


			for (size_t k=0; k<radiusElems.nzyxdim; k++)
			{
				if (dAi(radiusElems, k) > radiusThreshold * dAi(radiusTheoretical, k))
					maxRadius = k;
			}

			std::cout << "maxRadius = -> " << maxRadius << std::endl;
		}	


		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mapMask)
		{

			double valmask = DIRECT_MULTIDIM_ELEM(mapMask, n);
			DIRECT_MULTIDIM_ELEM(half1, n) = DIRECT_MULTIDIM_ELEM(half1, n) * valmask;
			DIRECT_MULTIDIM_ELEM(half2, n) = DIRECT_MULTIDIM_ELEM(half2, n) * valmask;
		}
	}
}


void ProgAngResAlign::generateShellMask(MultidimArray<double> &shellMask, size_t shellNum, bool ishelix)
{
	double sigma2 = 5*5;
	double inv2sigma2;
	inv2sigma2 = 1/(2*sigma2);
	if (ishelix)
	{
		FOR_ALL_ELEMENTS_IN_ARRAY3D(shellMask)
		{
			double radius = sqrt(i*i + j*j);
			radius = radius - shellNum;
			A3D_ELEM(shellMask, k, i, j) = exp(-(radius*radius)*inv2sigma2);
		}
	}else
	{
		FOR_ALL_ELEMENTS_IN_ARRAY3D(shellMask)
		{
			double radius = sqrt(k*k + i*i + j*j);
			radius = radius - shellNum;
			A3D_ELEM(shellMask, k, i, j) = exp(-(radius*radius)*inv2sigma2);
		}
	}
}


void ProgAngResAlign::applyShellMask(const MultidimArray<double> &half1, const MultidimArray<double> &half2,
									 const MultidimArray<double> &shellMask,
									 MultidimArray<double> &half1_aux, 
									 MultidimArray<double> &half2_aux)
{
		half1_aux = half1;
		half2_aux = half2;

		// Applying the mask
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(shellMask)
		{
			double valmask = DIRECT_MULTIDIM_ELEM(shellMask, n);
			DIRECT_MULTIDIM_ELEM(half1_aux, n) = DIRECT_MULTIDIM_ELEM(half1, n) * valmask;
			DIRECT_MULTIDIM_ELEM(half2_aux, n) = DIRECT_MULTIDIM_ELEM(half2, n) * valmask;
		}

		half1_aux.setXmippOrigin();
		half2_aux.setXmippOrigin();

		// fftshift must by applied before computing the fft. This will be computed later
		CenterFFT(half1_aux, true);
		CenterFFT(half2_aux, true);
}


void ProgAngResAlign::run()
	{
		std::cout << "Starting ... " << std::endl << std::endl;

		//This read the data and applies a fftshift to in the next step compute the fft
		// The maps half1_aux, and half2_aux are used to recover the initial maps after masking in each iteration
		MultidimArray<double> half1, half2;
    	MultidimArray<double> &phalf1 = half1, &phalf2 = half2;
    	readData(phalf1, phalf2);

		// Defining frequencies freq_fourier_x,y,z and freqMap
		// The number of frequencies in each shell freqElem is determined
        defineFrequenciesSimple(phalf1);

		//Define shell Mask
		MultidimArray<double> shellMask;
		shellMask.resizeNoCopy(phalf1);

		// Generating the set of directions to be analyzed
		// And converting the cone angle to radians
    	generateDirections(angles, false);
    	ang_con = ang_con*PI/180.0;

		MultidimArray<double> half1_aux; 
		MultidimArray<double> half2_aux;

		//Generate shell mask and computing radial resolution
		
		MetaDataVec mdOut;

		//std::cout << "max radius  = " << maxRadius << std::endl;

		for (size_t shellRadius = 0; shellRadius<maxRadius; shellRadius++) 
		{
			shellMask.initZeros();
			shellMask.setXmippOrigin();
			generateShellMask(shellMask, shellRadius, isHelix);

			// Image<double> img;
			// img() = shellMask;
			// FileName fn;
			// fn = formatString("mk_%i.mrc", shellRadius);
			// img.write(fn);

			applyShellMask(half1, half2, shellMask, half1_aux, half2_aux);

					//Computing the FFT
			FourierTransformer transformer2(FFTW_BACKWARD), transformer1(FFTW_BACKWARD);
			//transformer1.setThreadsNumber(Nthreads);
			//transformer2.setThreadsNumber(Nthreads);

			transformer1.FourierTransform(half1_aux, FT1, false);
			transformer2.FourierTransform(half2_aux, FT2, false);

			// Storing the shell of both maps as vectors global
			// The global FSC is also computed
			
			arrangeFSC_and_fscGlobal();


			double res;
			if (directionalRes)
			{
				std::vector<double> radialResolution(angles.mdimx);

				for (size_t k = 0; k<angles.mdimx; k++)
				{
					float rot  = MAT_ELEM(angles, 0, k);
					float tilt = MAT_ELEM(angles, 1, k);

					double resInterp = -1;	
					MultidimArray<float> fsc;

					// Estimating the direction FSC along the direction given by rot and tilt
					fscDir_fast(fsc, rot, tilt, thrs, resInterp);

					radialResolution[k] = resInterp;
				}

				std::sort(radialResolution.begin(),radialResolution.end());

				res = radialResolution[round(0.5*angles.mdimx)];
				radialResolution.clear();
			}
			else
			{
				fscGlobal(thrs, res);
			}

			std::cout << "Radius "<< shellRadius << "/" << maxRadius << " " << res << "A" << std::endl;

			MDRowVec row;
			row.setValue(MDL_RESOLUTION_FRC, res);
			row.setValue(MDL_IDX, shellRadius);
			mdOut.addRow(row);
			
		}

		mdOut.write(fnOut);
}
