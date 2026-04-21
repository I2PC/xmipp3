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

#ifndef _PROG_CLASSIFY_MAP_CLUSTER
#define _PROG_CLASSIFY_MAP_CLUSTER

#include "core/metadata_vec.h"
#include "core/xmipp_program.h"
#include "core/xmipp_image.h"
#include "core/xmipp_filename.h"
#include "reconstruction/resolution_fscoh.h"
#include "core/matrix2d.h"
#include "core/matrix1d.h"

#define VERBOSE_OUTPUT
// #define DEBUG_DIM
// #define DEBUG_FREQUENCY_MAP
#define DEBUG_OUTPUT_FILES
#define DEBUG_MDS
#define DEBUG_HIERARCHICAL_CLUSTERING

/**@defgroup ProgClassifyMapCluster Map clusterin algorithm based on FSC distance
   @ingroup ReconsLibrary */
//@{
/** Cluster map based in FSC distance*/

class ProgClassifyMapCluster: public XmippProgram
{
 public:
    // Input params
    FileName fn_mapPool;                    // Input metadata with map pool for analysis
    FileName fn_mapPool_statistical;        // Input metadata with map pool for statistical map calculation
    FileName fn_oroot;                      // Location for saving output maps
    double sampling_rate;                   // Sampling rate of input maps

    // Volume dimensions
    bool dimInitialized = false;
    size_t Xdim;
    size_t Ydim;
    size_t Zdim;
    size_t Ndim;
    size_t Xdim_ft;
	size_t Ydim_ft;
	size_t Zdim_ft;
	size_t Ndim_ft;

    // Data variables
    Matrix2D<double> distanceMatrix;  // Matrix for saving pairwise distance between maps
    struct Cluster
    {
        std::vector<int> points;  // original indices
        int id;                   // cluster id (for linkage matrix)
    };                                  // Data strutures for hierarchical clustering
    FourierTransformer ft;              // Fourier transformer
    MultidimArray<double> freqMap;                      // Frequency mapping in Fourier space
    MultidimArray<double> FSC;                        // Fourier Shell Coherence
    MultidimArray<double> FSC_num;                    // Fourier Shell Coherence numerator
    MultidimArray<double> FSC_den1;                    // Fourier Shell Coherence denominator
    MultidimArray<double> FSC_den2;                    // Fourier Shell Coherence denominator

    Image<std::complex<double>> referenceMapPool_ft;         // Reference map pool Fourier Transform

    FileName fn_V;                          // Filename for each input volume from pool
    Image<double> V;                        // Each input volume from pool
    MultidimArray<std::complex<double>> V_ft;       // Each input volume from pool Fourier Transform

    // Particle metadata
    MetaDataVec mapPoolMD;
    MDRowVec row;

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
    void calculateDistanceFSC(double &distance, int i1, int i2);
    void classicalMDS(Matrix2D<double>& D, Matrix2D<double>& B, Matrix1D<double>& eigenvals, Matrix2D<double>& eigenvecs);
    void kmeans(Matrix2D<double>& X, int k, int maxIter, Matrix1D<int>& labels);
    void hierarchicalClusteringLinkage_LW(const Matrix2D<double>& D, Matrix2D<double>& Z);

    // ---------------------- UTILS METHODS ------------------------------

    // Generate side info
    void generateSideInfo();
    void composefreqMap();
    void normalizeFTMap(MultidimArray<std::complex<double>> &volFT);
};
//@}
#endif