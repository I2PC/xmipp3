/***************************************************************************
 *
 * Authors:    Federico P. de Isidro-Gomez (federico.pdeisidro@astx.com)
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

#ifndef _PROG_FSCOH
#define _PROG_FSCOH

#include "core/metadata_vec.h"
#include "core/xmipp_program.h"
#include "core/xmipp_image.h"
#include "core/xmipp_filename.h"
 #include "core/xmipp_fftw.h"

#define VERBOSE_OUTPUT
// #define DEBUG_DIM
// #define DEBUG_FREQUENCY_MAP
#define DEBUG_OUTPUT_FILES
// #define DEBUG_FOURIER_SHELL_NORMALIZE

/**@defgroup ProgFSCoh Calculates statistical map
   @ingroup ReconsLibrary */
//@{
/** Calculate statistical map from a pool of input maps and weight input volume*/

class ProgFSCoh: public XmippProgram
{
 public:
    // Input params
    FileName fn_mapPool;               // Input metadata with map pool for analysis
    FileName fn_oroot;                 // Location for saving output maps
    double sampling_rate;              // Sapling rate of input maps

    // Volume dimensions
    bool dimInitialized = false;
    size_t Xdim;
    size_t Ydim;
    size_t Zdim;
    size_t Ndim;

    // Data variables
    MultidimArray<double> freqMap;                      // Frequency mapping in Fourier space
    MultidimArray<double> FSCoh;                        // Fourier Shell Coherence
    MultidimArray<double> FSCoh_num;                    // Fourier Shell Coherence numerator
    MultidimArray<double> FSCoh_den;                    // Fourier Shell Coherence denominator
    MultidimArray<std::complex<double>> FSCoh_map;      // Complex map components
    MultidimArray<double> FSCoh_map2;                   // Squared map components
    FileName fn_V;                                      // Filename for each input volume from pool
    Image<double> V;                                    // Each input volume from pool

    // Particle metadata
    MetaDataVec mapPoolMD;
    MDRowVec row;

    // Filtering variables
    int indexThr;

    // Fourier transformer
    FourierTransformer ft;


public:

    // ---------------------- IN/OUT METHODS -----------------------------
    // Define parameters
    void defineParams() override;
    // Read argument
    void readParams() override;
    // Show
    void show() const override;

    // ----------------------- MAIN METHODS ------------------------------
    void run();

    // ----------------------- CORE METHODS ------------------------------
    void fourierShellCoherence(MetaDataVec mapPoolMD);
    void calculateResolutionThreshold();

    // ---------------------- UTILS METHODS ------------------------------
    void composefreqMap();
    void normalizeMap(MultidimArray<double> &vol);
    void normalizeFTMap(MultidimArray<std::complex<double>> &volFT);
    void fourierShellNormalization(MultidimArray<std::complex<double>> &volFT);

};
//@}
#endif