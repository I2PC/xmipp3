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

#ifndef _PROG_CTF_WIENER2D_CORRECTION
#define _PROG_CTF_WIENER2D_CORRECTION

#include <core/xmipp_program.h>
#include <core/xmipp_filename.h>
#include <core/metadata_vec.h>
#include <core/xmipp_image.h>


class ProgCTFWiener2DCorrection : public XmippProgram
{
private:
    /* Filenames of the input subtomograms (.xmd file)
	and output file (volume of the average) */
    FileName fnIn, fnOut;

    /* Sampling rate */
    double sampling;

    /* Wiener constant */
    double wc;

    /* Defocus accuracy */
    double sigmaDf;

    /* Number of threads */
    int nthreads;

public:
    /* Creating a gaussian mask to weight the CTF corrected images */
    void gaussianMask(MultidimArray<double> &cumMask, 
					 MultidimArray<double> &tiMask,  
					 MultidimArray<double> &ptrImg, int x0, int stripeSize);

    /* Defining the params and help of the algorithm */
    void defineParams();

    /* It reads the input parameters */
    void readParams();

    /* Run the program */
    void run();
};

#endif
