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

/**@defgroup ProgClassifyMapCluster Calculates statistical map
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
    FileName fn_mask;                       // Protein mask filename
    double sampling_rate;                   // Sampling rate of input maps
    double protein_radius;                  // Protein radius
    double significance_thr;                // Significance Z-score threshold

    // Side info variables
    FileName fn_out_avg_map;
    FileName fn_out_std_map;
    FileName fn_out_median_map;
    FileName fn_out_mad_map;
    bool proteinMaskProvided = false;

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
    FourierTransformer ft;              // Fourier transformer
    MultidimArray<double> freqMap;                      // Frequency mapping in Fourier space
    MultidimArray<double> FSC;                        // Fourier Shell Coherence
    MultidimArray<double> FSC_num;                    // Fourier Shell Coherence numerator
    MultidimArray<double> FSC_den1;                    // Fourier Shell Coherence denominator
    MultidimArray<double> FSC_den2;                    // Fourier Shell Coherence denominator

    Image<double> referenceMapPool;         // Reference map pool
    Image<std::complex<double>> referenceMapPool_ft;         // Reference map pool Fourier Transform
    Image<double> medianMap;                // Median volume
    Image<double> MADMap;                   // MAD volume
    Image<double> V_ZscoresMAD;             // Each z-scores map using MAD from pool

    FileName fn_V;                          // Filename for each input volume from pool
    Image<double> V;                        // Each input volume from pool
    MultidimArray<std::complex<double>> V_ft;       // Each input volume from pool Fourier Transform
    Image<double> V_Zscores;                // Each z-scores map from pool
    Image<double> avgVolume;                // Average volume
    Image<double> stdVolume;                // Standard deviation volume
    Image<double> avgDiffVolume;            // Average difference volume
    MultidimArray<int> ROI_mask;            // Mask for focus analysis if protein radius provided
    MultidimArray<int> coincidentMask;      // Mask for coincident regions between each input map and the statiscal pool
    MultidimArray<int> differentMask;       // Mask for different regions between each input map and the statiscal pool
    MultidimArray<int> positiveMask;                // Mask for positive values in each input map
    MultidimArray<int> positiveMask_dilated;        // Mask for positive values in each input map + backgorund
    MultidimArray<double> distanceCoincidentMask;
    MultidimArray<double> distanceDifferentMask;

    // Calculated parameters
    std::vector<double> histogramEqualizationParameters;
    double equalizationParam;
    double partialOccupancyFactor;
    std::vector<double> zScoreAccumulator;

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
    // Write output statistical map
    void writeStatisticalMap();
    // Write Z-scores map
    void writeZscoresMap(FileName fnIn);
    // Write weighted map
    void writeWeightedMap(FileName fnIn);
    // Write maks
    void writeMask(FileName fnIn);

    // ----------------------- MAIN METHODS ------------------------------
    void run();

    // ----------------------- CORE METHODS ------------------------------
    void calculateDistanceFSC(double &distance, int i1, int i2);
    void classicalMDS(Matrix2D<double>& D, Matrix2D<double>& B, Matrix1D<double>& eigenvals, Matrix2D<double>& eigenvecs);
    void kmeans(Matrix2D<double>& X, int k, int maxIter, Matrix1D<int>& labels);


    void calculateFSCoh();
    void preprocessMap(FileName fnIn);
    void processStaticalMap();
    void computeStatisticalMaps();
    void calculateAvgDiffMap();
    void computeSigmaNormMAD(double& sigmaNorm);
    void computeSigmaNormIQR(double& sigmaNorm);
    void weightMap();

    // ---------------------- UTILS METHODS ------------------------------

    // Generate side info
    void generateSideInfo();
    void composefreqMap();
    void normalizeFTMap(MultidimArray<std::complex<double>> &volFT);


    void createRadiusMask();
    void generateDistanceMask(MultidimArray<int>& mask, MultidimArray<double>& maskDistance, double tao);

    double t_cdf(double t, int nu);
    double t_p_value(double t_stat, int nu);
    double percentile(const std::vector<double>& values, double p);


    // Methdos for new approach
    double median(std::vector<double> v);
    void computeMedianMap();
    void writeMedianMap();
    void writeMadMap();
    void computemMADMap();
    void calculateZscoreMADMap();
    void writeZscoresMADMap(FileName fnIn);

};
//@}
#endif