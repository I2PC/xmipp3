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
#ifndef _MONOGENIC_HH
#define _MONOGENIC_HH

#include <complex>
#include "core/matrix1d.h"

template<typename T>
class MultidimArray;
class FourierTransformer;

/** Routines for working with monogenic signals
*/
/**@defgroup Monogenic Monogenic Signal
   @ingroup DataLibrary */
//@{
class Monogenic
{
public:
	// PROTEINRADIUSVOLUMEANDSHELLSTATISTICS: This function takes as input a "mask" and returns
	// the radius and the volumen "vol" of the protein. The radius is defines has the distance
	// from the center of the cube to the farthest point of the protein measured from the center.
	// The volumen "vol" represent the number of voxels of the mask.
	void proteinRadiusVolumeAndShellStatistics(const MultidimArray<int> &mask, int &radius,
			long &vol);


	// FINDCLIFFVALUE: This function determines the radius of noise, "radiuslimit". It means given
	// a map, "inputmap", the radius measured from the origin for which the map is masked with a
        // spherical mask is detected. Outside of this sphere there is no noise. Once the all voxels of  
	// of the mask with corresponding radiues greater than "radiuslimit" has been set to -1, the 
	// parameter "rsmooth" does not affect to the output of this function, it is only used to 
	// provide information to the user when the "radiuslimit" is close to the boxsize. Note that 
	// the perimeter of the box is sometimes smoothed when a Fourier Transform is carried out. 
	// To prevent this situation this parameter is provided, but it is only informative via 
	// the standard output
	void findCliffValue(MultidimArray<double> &inputmap,
			int &radius, int &radiuslimit, MultidimArray<int> &mask, double rsmooth);


        //FOURIERFREQVECTOR: It defines a vector, freq_fourier, that contains
        // the frequencies of the Fourier direction. Where dimarrayFourier is the
        // number of components of the vector, and dimarrayReal is the dimensions
        // of the map along the direction for which the fft is computed
        Matrix1D<double> fourierFreqVector(size_t dimarrayFourier, size_t dimarrayReal);

	//FOURIERFREQS_3D: Determine the map of frequencies in the Fourier Space as an output.
	// Also, the accesible frequencies along each direction are calculated, they are
	// "freq_fourier_x", "freq_fourier_y", and "freq_fourier_z".
	MultidimArray<double> fourierFreqs_3D(const MultidimArray< std::complex<double> > &myfftV,
			const MultidimArray<double> &inputVol,
			Matrix1D<double> &freq_fourier_x,
			Matrix1D<double> &freq_fourier_y,
			Matrix1D<double> &freq_fourier_z);

	//RESOLUTION2EVAL: Determines the resoltion to be analzed in the estimation
	//of the local resolution. These resolution are freq, freqL (digital units)
	// being freqL the tail of the raise cosine centered at freq. The parameter
	// resolution is the frequency freq in converted into Angstrom. The parameters
	//minRes and maxRes, determines the limits of the resolution range to be
	// analyzed (in Angstrom). Sampling is the sampling rate in A/px, and step,
	// is the resolution step for which analyze the local resolution.
	void resolution2eval(int &count_res, double step,
			double &resolution, double &last_resolution,
			double &freq, double &freqL,
			int &last_fourier_idx,
			int &volsize,
			bool &continueIter,	bool &breakIter,
			double &sampling, double &maxRes);

	// AMPLITUDEMONOSIG3D_LPF: Estimates the monogenic amplitude of a HPF map, myfftV is the
	// Fourier transform of that map. The monogenic amplitude is "amplitude", and the
	// parameters "fftVRiesz", "fftVRiesz_aux", "VRiesz", are auxiliar variables that will
	// be defined inside the function (they are input to speed up monores algorithm).
	// Freq, freqL and freqH, are the frequency of the HPF and the tails of a raise cosine.
	// Note that once the monogenic amplitude is estimated, it is low pass filtered to obtain
	// a smooth version and avoid rippling. freq_fourier_x, freq_fourier_y, freq_fourier_z,
	// are the accesible Fourier frequencies along each axis.
	void amplitudeMonoSig3D_LPF(const MultidimArray< std::complex<double> > &myfftV,
			FourierTransformer &transformer_inv,
			MultidimArray< std::complex<double> > &fftVRiesz,
			MultidimArray< std::complex<double> > &fftVRiesz_aux, MultidimArray<double> &VRiesz,
			double freq, double freqH, double freqL, MultidimArray<double> &iu,
			Matrix1D<double> &freq_fourier_x, Matrix1D<double> &freq_fourier_y,
			Matrix1D<double> &freq_fourier_z, MultidimArray<double> &amplitude,
			int count, FileName fnDebug);

	// STATISTICSINBINARYMASK2: Estimates the staticstics of two maps:
	// Signal map "volS" and Noise map "volN". The signal statistics
	// are obtained by mean of a binary mask. The results are the mean
	// and variance of noise and signal, as well as the number of voxels
	// of signal and noise NS and NN respectively. The thr95 represents
	// the percentile 95 of noise distribution.
	void statisticsInBinaryMask2(const MultidimArray<double> &volS,
			const MultidimArray<double> &volN,
			MultidimArray<int> &mask, double &meanS, double &sdS2,
			double &meanN, double &sdN2, double &significance, double &thr95, double &NS, double &NN);

	// STATISTICSINOUTBINARYMASK2: Estimates the staticstics of a single
	// map. The signal statistics are obtained by mean of a binary mask.
	// The results are the mean and variance of noise and signal, as well
	// as the number of voxels of signal and noise NS and NN respectively.
	// The thr95 represents the percentile 95 of noise distribution.
	void statisticsInOutBinaryMask2(const MultidimArray<double> &volS,
			MultidimArray<int> &mask, double &meanS, double &sdS2,
			double &meanN, double &sdN2, double &significance, double &thr95, double &NS, double &NN);

	// SETLOCALRESOLUTIONHALFMAPS: Set the local resolution of a voxel, by
	// determining if the monogenic amplitude "amplitudeMS" is higher than
	// the threshold of noise "thresholdNoise". Thus the local resolution
	// map "plocalResolution" is set, with the resolution value "resolution"
	// or with "resolution_2". "resolution" is the resolution for with
	// the hypohtesis test is carried out, and "resolution_2" is the resolution
	// of the analisys two loops ago (see monores method)
	void setLocalResolutionHalfMaps(const MultidimArray<double> &amplitudeMS,
			MultidimArray<int> &pMask, MultidimArray<double> &plocalResolutionMap,
			double thresholdNoise, double resolution, double resolution_2);

	// SETLOCALRESOLUTIONMAP: Set the local resolution of a voxel, by
	// determining if the monogenic amplitude "amplitudeMS" is higher than
	// the threshold of noise "thresholdNoise". Thus the local resolution
	// map "plocalResolution" is set, with the resolution value "resolution"
	// or with "resolution_2". "resolution" is the resolution for with
	// the hypohtesis test is carried out, and "resolution_2" is the resolution
	// of the analisys two loops ago (see monores method)
	void setLocalResolutionMap(const MultidimArray<double> &amplitudeMS,
		MultidimArray<int> &pMask, MultidimArray<double> &plocalResolutionMap,
		double thresholdNoise, double resolution, double resolution_2);

	// MONOGENICAMPLITUDE_3D_FOURIER: Given the fourier transform of a map
	// "myfftV", this function computes the monogenic amplitude "amplitude" 
	// iu is the inverse of the frequency in Fourier space.
	void monogenicAmplitude_3D_Fourier(const MultidimArray< std::complex<double> > &myfftV,
			MultidimArray<double> &iu, MultidimArray<double> &amplitude, int numberOfThreads);

	//ADDNOISE: This function add gaussian with mean = double mean and standard deviation
	// equal to  double stddev to a map given by "vol"
	void addNoise(MultidimArray<double> &vol, double mean, double stddev);

	//CREATEDATATEST: This function generates fringe pattern (rings) map returned 
	// as a multidimArray, with  dimensions given by xdim, ydim and zdim. The 
	// wavelength of the pattern is given by double wavelength
	MultidimArray<double> createDataTest(size_t xdim, size_t ydim, size_t zdim,
		double wavelength);

};

//@}
#endif
