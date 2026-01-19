/***************************************************************************
 *
 * Authors:     Jose Luis Vilas (jlvilas@cnb.csic.es)
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
#include "monogenic_signal.h"
#include "core/xmipp_image.h"
#include "core/xmipp_fftw.h"
#include <random>
#include <algorithm>


// This function takes as input a "mask" and returns the radius and the volumen "vol" of the
// protein. The radius is defines has the distance from the center of the cube to the farthest
// point of the protein measured from the center. The volumen "vol" represent the number of
// voxels of the mask
void Monogenic::proteinRadiusVolumeAndShellStatistics(const MultidimArray<int> &mask, int &radius,
		long &vol)
{
	vol = 0;
	radius = 0;
	FOR_ALL_ELEMENTS_IN_ARRAY3D(mask)
	{

		if (A3D_ELEM(mask, k, i, j) == 1)
		{		
                        int R2 = (k*k + i*i + j*j);
			++vol;
			if (R2>radius)
				radius = R2;
		}
	}
	radius = sqrt(radius);
	std::cout << "                                     " << std::endl;
	std::cout << "The protein has a radius of "<< radius << " px " << std::endl;
}


// FINDCLIFFVALUE: This function determines the radius of noise, "radiuslimit". It means given
// a map, "inputmap", the radius measured from the origin for which the map is masked with a
// spherical mask is detected. Outside of this sphere there is no noise. Once the all voxels of  
// of the mask with corresponding radiues greater than "radiuslimit" has been set to -1, the 
// parameter "rsmooth" does not affect to the output of this function, it is only used to 
// provide information to the user when the "radiuslimit" is close to the boxsize. Note that 
// the perimeter of the box is sometimes smoothed when a Fourier Transform is carried out. 
// To prevent this situation this parameter is provided, but it is only informative via 
// the standard output
void Monogenic::findCliffValue(MultidimArray<double> &inputmap,
		int &radius, int &radiuslimit, MultidimArray<int> &mask, double rsmooth)
{
	double criticalZ = icdf_gauss(0.95);
	radiuslimit = XSIZE(inputmap)/2;
	double last_mean=0;
	double last_std2=1e-38;

	for (int rad = radius; rad<radiuslimit; rad++)
	{
		double sum=0;
		double sum2=0;
		double N=0;
		int sup;
		int inf;
		inf = rad*rad;
		sup = (rad + 1)*(rad + 1);
		FOR_ALL_ELEMENTS_IN_ARRAY3D(inputmap)
		{
			double aux = k*k + i*i + j*j;
			if ( (aux<sup) && (aux>=inf) )
			{
				double value = A3D_ELEM(inputmap, k, i, j);;
				sum += value;
				sum2 += value*value;
				N++;
			}
		}
		double mean = sum/N;
		double std2 = sum2/N - mean*mean;

		if (std2/last_std2<0.01)
		{
			radiuslimit = rad - 1;
			break;
		}

		last_mean = mean, last_std2 = std2;
	}

	std::cout << "There is no noise beyond a radius of " << radiuslimit << " px " << std::endl;
	std::cout << "Regions with a radius greater than " << radiuslimit << " px will not be considered" << std::endl;

	double raux = (radiuslimit - rsmooth);
	if (raux<=radius)
	{
		std::cout << "Warning: the boxsize is very close to "
				"the protein size please provide a greater box" << std::endl;
	}

	raux *= raux;

	FOR_ALL_ELEMENTS_IN_ARRAY3D(mask)
	{
		double aux = k*k + i*i + j*j;
		if (aux>=raux)
			A3D_ELEM(mask, k, i, j) = -1;
	}

}

//FOURIERFREQVECTOR: It defines a vector, freq_fourier, that contains
// the frequencies of the Fourier direction. Where dimarrayFourier is the
// number of components of the vector, and dimarrayReal is the dimensions
// of the map along the direction for which the fft is computed
Matrix1D<double> Monogenic::fourierFreqVector(size_t dimarrayFourier, size_t dimarrayReal)
{
        double u;
        Matrix1D<double> freq_fourier;
	      freq_fourier.initZeros(dimarrayFourier);
        VEC_ELEM(freq_fourier,0) = 1e-38; //A really low value to represent 0 avooiding singularities
	for(size_t k=1; k<dimarrayFourier; ++k){
		FFT_IDX2DIGFREQ(k,dimarrayReal, u);
		VEC_ELEM(freq_fourier, k) = u;
	}
	return freq_fourier;
}


//TODO: Use macros to avoid repeating code
//FOURIERFREQS_3D: Determine the map of frequencies in the Fourier Space as an output.
// Also, the accessible frequencies along each direction are calculated, they are
// "freq_fourier_x", "freq_fourier_y", and "freq_fourier_z".
MultidimArray<double> Monogenic::fourierFreqs_3D(const MultidimArray< std::complex<double> > &myfftV,
		const MultidimArray<double> &inputVol,
		Matrix1D<double> &freq_fourier_x,
		Matrix1D<double> &freq_fourier_y,
		Matrix1D<double> &freq_fourier_z)
{
	freq_fourier_z = fourierFreqVector(ZSIZE(myfftV), ZSIZE(inputVol));
        freq_fourier_y = fourierFreqVector(YSIZE(myfftV), YSIZE(inputVol));
        freq_fourier_x = fourierFreqVector(XSIZE(myfftV), XSIZE(inputVol));

	MultidimArray<double> iu;

	iu.initZeros(myfftV);

	double uz;
	double uy;
	double ux;
	double uz2;
	double u2;
	double uz2y2;
	long n=0;
	//  TODO: Take ZSIZE(myfftV) out of the loop
	//	TODO: Use freq_fourier_x instead of calling FFT_IDX2DIGFREQ

	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		FFT_IDX2DIGFREQ(k,ZSIZE(inputVol),uz);
		uz2 = uz*uz;
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			FFT_IDX2DIGFREQ(i,YSIZE(inputVol),uy);
			uz2y2 = uz2 + uy*uy;

			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				FFT_IDX2DIGFREQ(j,XSIZE(inputVol), ux);
				u2 = uz2y2 + ux*ux;

				if ((k != 0) || (i != 0) || (j != 0))
				{
					DIRECT_MULTIDIM_ELEM(iu,n) = 1/sqrt(u2);
				}
				else
				{
					DIRECT_MULTIDIM_ELEM(iu,n) = 1e38;
				}
				++n;
			}
		}
	}

	return iu;
}


//RESOLUTION2EVAL: Determines the resoltion to be analzed in the estimation
//of the local resolution. These resolution are freq, freqL (digital units)
// being freqL the tail of the raise cosine centered at freq. The parameter
// resolution is the frequency freq in converted into Angstrom. The parameters
//minRes and maxRes, determines the limits of the resolution range to be
// analyzed (in Angstrom). Sampling is the sampling rate in A/px, and step,
// is the resolution step for which analyze the local resolution.
void Monogenic::resolution2eval(int &count_res, double step,
		double &resolution, double &last_resolution,
		double &freq, double &freqL,
		int &last_fourier_idx,
		int &volsize,
		bool &continueIter,	bool &breakIter,
		double &sampling, double &maxRes)
{
//TODO: simplify this function
	resolution = maxRes - count_res*step;

	freq = sampling/resolution;
	++count_res;

	double Nyquist = 2*sampling;
	double aux_frequency;
	int fourier_idx;

	DIGFREQ2FFT_IDX(freq, volsize, fourier_idx);

	FFT_IDX2DIGFREQ(fourier_idx, volsize, aux_frequency);

	freq = aux_frequency;

	if (fourier_idx == last_fourier_idx)
	{
		continueIter = true;
		return;
	}

	last_fourier_idx = fourier_idx;
	resolution = sampling/aux_frequency;


	if (count_res == 0){
		last_resolution = resolution;
        }

	if (resolution<Nyquist)// || (resolution > last_resolution) )
	{
		breakIter = true;
		return;
	}

	freqL = sampling/(resolution + step);

	int fourier_idx_2;

	DIGFREQ2FFT_IDX(freqL, volsize, fourier_idx_2);

	if (fourier_idx_2 == fourier_idx)
	{
		if (fourier_idx > 0){
			FFT_IDX2DIGFREQ(fourier_idx - 1, volsize, freqL);
		}
		else{
			freqL = sampling/(resolution + step);
		}
	}

}


// AMPLITUDEMONOSIG3D_LPF: Estimates the monogenic amplitude of a HPF map, myfftV is the
// Fourier transform of that map. The monogenic amplitude is "amplitude", and the
// parameters "fftVRiesz", "fftVRiesz_aux", "VRiesz", are auxiliar variables that will
// be defined inside the function (they are input to speed up monores algorithm).
// Freq, freqL and freqH, are the frequency of the HPF and the tails of a raise cosine.
// Note that once the monogenic amplitude is estimated, it is low pass filtered to obtain
// a smooth version and avoid rippling. freq_fourier_x, freq_fourier_y, freq_fourier_z,
// are the accesible Fourier frequencies along each axis.
void Monogenic::amplitudeMonoSig3D_LPF(const MultidimArray< std::complex<double> > &myfftV,
		FourierTransformer &transformer_inv,
		MultidimArray< std::complex<double> > &fftVRiesz,
		MultidimArray< std::complex<double> > &fftVRiesz_aux, MultidimArray<double> &VRiesz,
		double freq, double freqH, double freqL, MultidimArray<double> &iu,
		Matrix1D<double> &freq_fourier_x, Matrix1D<double> &freq_fourier_y,
		Matrix1D<double> &freq_fourier_z, MultidimArray<double> &amplitude,
		int count, FileName fnDebug)
{
//FIXME: use atf.h
//	FourierTransformer transformer_inv;

	fftVRiesz.initZeros(myfftV);
	fftVRiesz_aux.initZeros(myfftV);
	std::complex<double> J(0,1);

	// Filter the input volume and add it to amplitude
	long n=0;
	double ideltal=PI/(freq-freqH);

	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				double iun=DIRECT_MULTIDIM_ELEM(iu,n);
				double un=1.0/iun;
				if (freqH<=un && un<=freq)
				{
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) *= 0.5*(1+cos((un-freq)*ideltal));//H;
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = -J;
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= DIRECT_MULTIDIM_ELEM(fftVRiesz, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= iun;
				} else if (un>freq)
				{
					DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = -J;
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= DIRECT_MULTIDIM_ELEM(fftVRiesz, n);
					DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= iun;
				}
				++n;
			}
		}
	}

	transformer_inv.inverseFourierTransform(fftVRiesz, amplitude);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n) *= DIRECT_MULTIDIM_ELEM(amplitude,n);

	//TODO: create a macro with these kind of code
	// Calculate first component of Riesz vector
	double ux;
	n=0;
	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				ux = VEC_ELEM(freq_fourier_x,j);
				DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = ux*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				++n;
			}
		}
	}

	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n)+=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);


	// Calculate second and third component of Riesz vector
	n=0;
	double uy;
	double uz;
	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		uz = VEC_ELEM(freq_fourier_z,k);
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			uy = VEC_ELEM(freq_fourier_y,i);
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = uz*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = uy*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				++n;
			}
		}
	}

	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n) += DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);

	transformer_inv.inverseFourierTransform(fftVRiesz_aux, VRiesz);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
	{
		DIRECT_MULTIDIM_ELEM(amplitude,n)+=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);
		DIRECT_MULTIDIM_ELEM(amplitude,n)=sqrt(DIRECT_MULTIDIM_ELEM(amplitude,n));
	}

	transformer_inv.FourierTransform(amplitude, fftVRiesz, false);

	double raised_w = PI/(freqL-freq);
	n=0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(fftVRiesz)
	{
		double un=1.0/DIRECT_MULTIDIM_ELEM(iu,n);
		if (freqL>=un && un>=freq)
		{
			DIRECT_MULTIDIM_ELEM(fftVRiesz,n) *= 0.5*(1 + cos(raised_w*(un-freq)));
		}
		else
		{
			if (un>freqL)
			{
				DIRECT_MULTIDIM_ELEM(fftVRiesz,n) = 0;
			}
		}
	}

	transformer_inv.inverseFourierTransform();
//
//	saveImg = amplitude;
//	FileName iternumber;
//	iternumber = formatString("_Amplitude_new_%i.vol", count);
//	saveImg.write(fnDebug+iternumber);
//	saveImg.clear();
}


// STATISTICSINBINARYMASK2: Estimates the staticstics of two maps:
// Signal map "volS" and Noise map "volN". The signal statistics
// are obtained by mean of a binary mask. The results are the mean
// and variance of noise and signal, as well as the number of voxels
// of signal and noise NS and NN respectively. The thr95 represents
// the percentile 95 of noise distribution.
void Monogenic::statisticsInBinaryMask2(const MultidimArray<double> &volS,
		const MultidimArray<double> &volN,
		MultidimArray<int> &mask, double &meanS, double &sdS2,
		double &meanN, double &sdN2, double &significance, double &thr95, double &NS, double &NN)
{
	double sumS = 0;
	double sumS2 = 0;
	double sumN = 0;
	double sumN2 = 0;
	NN = 0;
	NS = 0;
	std::vector<double> noiseValues;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(volS)
	{
		if (DIRECT_MULTIDIM_ELEM(mask, n)>0)
		{
			double amplitudeValue=DIRECT_MULTIDIM_ELEM(volS, n);
			sumS  += amplitudeValue;
			sumS2 += amplitudeValue*amplitudeValue;
			++NS;
		}
		if (DIRECT_MULTIDIM_ELEM(mask, n)>=0) //BE CAREFULL WITH THE =
		{
			double amplitudeValueN=DIRECT_MULTIDIM_ELEM(volN, n);
			noiseValues.push_back(amplitudeValueN);
			sumN  += amplitudeValueN;
			sumN2 += amplitudeValueN*amplitudeValueN;
			++NN;
		}
	}

	std::sort(noiseValues.begin(),noiseValues.end());
	thr95 = noiseValues[size_t(noiseValues.size()*significance)];
	meanS = sumS/NS;
	meanN = sumN/NN;
	sdS2 = sumS2/NS - meanS*meanS;
	sdN2 = sumN2/NN - meanN*meanN;
}


// STATISTICSINOUTBINARYMASK2: Estimates the staticstics of a single
// map. The signal statistics are obtained by mean of a binary mask.
// The results are the mean and variance of noise and signal, as well
// as the number of voxels of signal and noise NS and NN respectively.
// The thr95 represents the percentile 95 of noise distribution.
void Monogenic::statisticsInOutBinaryMask2(const MultidimArray<double> &volS,
		MultidimArray<int> &mask, double &meanS, double &sdS2,
		double &meanN, double &sdN2, double &significance, double &thr95, double &NS, double &NN)
{
	double sumS = 0;
	double sumS2 = 0;
	double sumN = 0;
	double sumN2 = 0;
	NN = 0;
	NS = 0;

	std::vector<double> noiseValues;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(volS)
	{
		if (DIRECT_MULTIDIM_ELEM(mask, n)>=1)
		{
			double amplitudeValue=DIRECT_MULTIDIM_ELEM(volS, n);
			sumS  += amplitudeValue;
			sumS2 += amplitudeValue*amplitudeValue;
			++NS;
		}
		if (DIRECT_MULTIDIM_ELEM(mask, n)==0)
		{
			double amplitudeValueN=DIRECT_MULTIDIM_ELEM(volS, n);
			noiseValues.push_back(amplitudeValueN);
			sumN  += amplitudeValueN;
			sumN2 += amplitudeValueN*amplitudeValueN;
			++NN;
		}
	}

	std::sort(noiseValues.begin(),noiseValues.end());
	thr95 = noiseValues[size_t(noiseValues.size()*significance)];
	meanS = sumS/NS;
	meanN = sumN/NN;
	sdS2 = sumS2/NS - meanS*meanS;
	sdN2 = sumN2/NN - meanN*meanN;
}


// SETLOCALRESOLUTIONHALFMAPS: Set the local resolution of a voxel, by
// determining if the monogenic amplitude "amplitudeMS" is higher than
// the threshold of noise "thresholdNoise". Thus the local resolution
// map "plocalResolution" is set, with the resolution value "resolution"
// or with "resolution_2". "resolution" is the resolution for with
// the hypohtesis test is carried out, and "resolution_2" is the resolution
// of the analisys two loops ago (see monores method)
void Monogenic::setLocalResolutionHalfMaps(const MultidimArray<double> &amplitudeMS,
		MultidimArray<int> &pMask, MultidimArray<double> &plocalResolutionMap,
		double thresholdNoise, double resolution, double resolution_2)
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitudeMS)
		{
			if (DIRECT_MULTIDIM_ELEM(pMask, n)>=1)
				if (DIRECT_MULTIDIM_ELEM(amplitudeMS, n)>thresholdNoise)
				{
					DIRECT_MULTIDIM_ELEM(pMask, n) = 1;
					DIRECT_MULTIDIM_ELEM(plocalResolutionMap, n) = resolution;
				}
				else{
					DIRECT_MULTIDIM_ELEM(pMask, n) += 1;
					if (DIRECT_MULTIDIM_ELEM(pMask, n) >2)
					{
						DIRECT_MULTIDIM_ELEM(pMask, n) = 0;
						DIRECT_MULTIDIM_ELEM(plocalResolutionMap, n) = resolution_2;
					}
				}
		}
	}

// SETLOCALRESOLUTION: Set the local resolution of a voxel, by
// determining if the monogenic amplitude "amplitudeMS" is higher than
// the threshold of noise "thresholdNoise". Thus the local resolution
// map "plocalResolution" is set, with the resolution value "resolution"
// or with "resolution_2". "resolution" is the resolution for with
// the hypohtesis test is carried out, and "resolution_2" is the resolution
// of the analisys two loops ago (see monores method)
void Monogenic::setLocalResolutionMap(const MultidimArray<double> &amplitudeMS,
		MultidimArray<int> &pMask, MultidimArray<double> &plocalResolutionMap,
		double thresholdNoise, double resolution, double resolution_2)
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitudeMS)
		{
			if (DIRECT_MULTIDIM_ELEM(pMask, n)>=1)
				if (DIRECT_MULTIDIM_ELEM(amplitudeMS, n)>thresholdNoise)
				{
					DIRECT_MULTIDIM_ELEM(pMask, n) = 1;
					DIRECT_MULTIDIM_ELEM(plocalResolutionMap, n) = resolution;//sampling/freq;
				}
				else{
					DIRECT_MULTIDIM_ELEM(pMask, n) += 1;
					if (DIRECT_MULTIDIM_ELEM(pMask, n) >2)
					{
						DIRECT_MULTIDIM_ELEM(pMask, n) = -1;
						DIRECT_MULTIDIM_ELEM(plocalResolutionMap, n) = resolution_2;//maxRes - counter*R_;
					}
				}
		}
	}


// MONOGENICAMPLITUDE_3D_FOURIER: Given the fourier transform of a map
// "myfftV", this function computes the monogenic amplitude "amplitude" 
// iu is the inverse of the frequency in Fourier space.
void Monogenic::monogenicAmplitude_3D_Fourier(const MultidimArray< std::complex<double> > &myfftV,
		MultidimArray<double> &iu, MultidimArray<double> &amplitude, int numberOfThreads)
{
	Matrix1D<double> freq_fourier_z;
	Matrix1D<double> freq_fourier_y;
	Matrix1D<double> freq_fourier_x;

	iu = fourierFreqs_3D(myfftV, amplitude, freq_fourier_x, freq_fourier_y, freq_fourier_z);

	// Filter the input volume and add it to amplitude
	MultidimArray< std::complex<double> > fftVRiesz;
	MultidimArray< std::complex<double> > fftVRiesz_aux;
	fftVRiesz.initZeros(myfftV);
	fftVRiesz_aux.initZeros(myfftV);
	std::complex<double> J(0,1);

	long n=0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(myfftV)
	{
		//double H=0.5*(1+cos((un-w1)*ideltal));
		DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
		DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = -J;
		DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= DIRECT_MULTIDIM_ELEM(fftVRiesz, n);
		DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= DIRECT_MULTIDIM_ELEM(iu, n);

	}
	MultidimArray<double> VRiesz;
	VRiesz.resizeNoCopy(amplitude);

	FourierTransformer transformer_inv;
        transformer_inv.setThreadsNumber(numberOfThreads);
	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n) = DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);

	// Calculate first component of Riesz vector
	double uz;
	double uy;
	double ux;
	n=0;
	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				ux = VEC_ELEM(freq_fourier_x,j);
				DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = ux*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				++n;
			}
		}
	}

	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n) += DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);

	// Calculate second and third components of Riesz vector
	n=0;
	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		uz = VEC_ELEM(freq_fourier_z,k);
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			uy = VEC_ELEM(freq_fourier_y,i);
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = uy*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = uz*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				++n;
			}
		}
	}
	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n)+=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);

	transformer_inv.inverseFourierTransform(fftVRiesz_aux, VRiesz);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
	{
		DIRECT_MULTIDIM_ELEM(amplitude,n) += DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);
		DIRECT_MULTIDIM_ELEM(amplitude,n) = sqrt(DIRECT_MULTIDIM_ELEM(amplitude,n));
	}
}


//ADDNOISE: This function add gaussian with mean = double mean and standard deviation
// equal to  double stddev to a map given by "vol"
void Monogenic::addNoise(MultidimArray<double> &vol, double mean, double stddev)
{

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::normal_distribution<double> dist(mean, stddev);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(vol)
		DIRECT_MULTIDIM_ELEM(vol,n) += dist(generator);
}


//CREATEDATATEST: This function generates fringe pattern (rings) map returned 
// as a multidimArray, with  dimensions given by xdim, ydim and zdim. The 
// wavelength of the pattern is given by double wavelength
MultidimArray<double> Monogenic::createDataTest(size_t xdim, size_t ydim, size_t zdim,
		double wavelength)
{
	int siz_z;
	int siz_y;
	int siz_x;
	double x;
	double y;
	double z;

	siz_z = xdim/2;
	siz_y = ydim/2;
	siz_x = xdim/2;
	MultidimArray<double> testmap;
	testmap.initZeros(zdim, ydim, xdim);

	long n=0;
	for(int k=0; k<zdim; ++k)
	{
		z = (k - siz_z);
		z= z*z;
		for(int i=0; i<ydim; ++i)
		{
			y = (i - siz_y);
			y = y*y;
			y = z + y;
			for(int j=0; j<xdim; ++j)
			{
				x = (j - siz_x);
				x = x*x;
				x = sqrt(x + y);
				DIRECT_MULTIDIM_ELEM(testmap, n) = cos(2*PI/wavelength*x);
				++n;
			}
		}
	}

	FileName fn;
	fn = formatString("fringes.vol");
	Image<double> saveImg;
	saveImg() = testmap;
	saveImg.write(fn);

	return testmap;
}

