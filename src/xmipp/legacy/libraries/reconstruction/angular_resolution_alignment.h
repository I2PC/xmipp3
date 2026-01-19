/***************************************************************************
 *
 * Authors:    Jose Luis Vilas (jlvilas@cnb.csic.es)
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

#ifndef _PROG_ANGULAR_RESOLUTION_ALIGNMENT
#define _PROG_ANGULAR_RESOLUTION_ALIGNMENT

#include <core/xmipp_program.h>
#include <core/matrix2d.h>
#include <core/xmipp_fftw.h>
#include <core/xmipp_filename.h>
#include <core/metadata_vec.h>


class ProgAngResAlign : public XmippProgram
{
private:
    // Filenames
    FileName fnhalf1, fnhalf2, fnmask, fnOut;

    // Double Params
    double sampling, ang_con, thrs;
    
    // Maximum radius to be analyzed
    int maxRadius;

    // Number of threads in the paralellization
    int Nthreads;
       
    // Bool params
    bool isHelix, limRad, directionalRes;

    // Matrix2d for the projection angles
    Matrix2D<float> angles;

    // Int params
    size_t xvoldim, yvoldim, zvoldim, fscshellNum;

	// Frequency vectors and frequency map
	Matrix1D<double> freq_fourier_x;
	Matrix1D<double> freq_fourier_y;
	Matrix1D<double> freq_fourier_z;
	MultidimArray<float> fx, fy, fz;
    MultidimArray< double > freqMap;

	// Half maps
	MultidimArray< std::complex< double > > FT1, FT2;
	MultidimArray<float> real_z1z2, absz1_vec, absz2_vec;

	//Access indices
	MultidimArray<long> freqElems, cumpos, freqidx, arr2indx;


private:
        /* This function determines the frequency map in Fourier space. 
        * The input is the original map inputVol. The output is stored in
        * the global variable freqMap (defined in the .h)
        * FreqMap is in Fourier space an each voxel defines the frequency
        * value associated to that position
        */
        void defineFrequenciesSimple(const MultidimArray<double> &inputVol);

        /* This function tales two half maps and a mask. The half maps
        * are multiplied by the mask returning two masked half maps half1_aux and half2_aux 
        */
        void applyShellMask(const MultidimArray<double> &half1, const MultidimArray<double> &half2,
									 const MultidimArray<double> &shellMask,
									 MultidimArray<double> &half1_aux, 
									 MultidimArray<double> &half2_aux);

        /* Estimates the global FSC between two half maps FT1 and FT2 (in Fourier Space)
         * ARRANGEFSC_AND_FSCGLOBAL: This function estimates the global resolution FSC FSC=real(z1*conj(z2)/(||z1||·||z2||)
         * and precomputes the products z1*conj(z2),  ||z1||, ||z2||, to calculate faster the directional FSC. The input are
         * the Fourier transform of two half maps (FT1, FT2) defined in the .h, the sampling_rate, the used threshold (thrs)
         * to that define the resolution (FSC-threshold). The output are: 1) The resolution of the map in Angstrom (In the terminal).
         * 2) three vectors defined in the .h, real_z1z2, absz1_vec, absz2_vec, with z1*conj(z2),  ||z1||, ||z2||, defined in
         * float to speed up the computation and reduce the use of memory. These vector make use of the two half maps FT1, and FT2.
         */
		void arrangeFSC_and_fscGlobal();

        /* This function estiamtes the global FSC between two half maps. The half maps are defined in the .h.
        */
        void fscGlobal(double &threshold, double &resol);

        /* Defines a Matrix2D with coordinates Rot and tilt achieving a uniform coverage of the
        * projection sphere. Bool alot = True, implies a dense coverage */
        void generateDirections(Matrix2D<float> &angles, bool alot);

        /* The half maps are masked with two mask (the one provided by the user and a second one defined by the algorithm)
        * This function implements the second one. This mask is a radial mask with a Gaussian profile centered at
        * a specific radius. If the map is a helix the mask will be a gaussian cylinder, if not it will be a gassian shell
        */
        void generateShellMask(MultidimArray<double> &shellMask, size_t shellNum, bool ishelix);

        /* Estimates the directional FSC between two half maps FT1 and FT2 (in Fourier Space)
        * requires the sampling rate, and the frequency vectors,  */
		void fscDir_fast(MultidimArray<float> &fsc, double rot, double tilt,
						              double &thrs, double &resol);

        /* PREPAREDATA: Data are prepared to be taken by the algorithm. 
        * The half maps will be read and stored in the multidimarray half1, and half2.
        * Also de mask (if provided) is read and stored in mask (defined in .h) */
        void readData(MultidimArray<double> &half1, MultidimArray<double> &half2);

        /* FSCINTERPOLATION: The exact resolution of the the FSC = thrs is estimated. thrs is a global variable */
        void fscInterpolation(const MultidimArray<double> &freq, const MultidimArray< double > &frc);

public:
        /* Defining the params and help of the algorithm */
        void defineParams();

        /* It reads the input parameters */
        void readParams();

        /* Run the program */
        void run();
};

#endif
