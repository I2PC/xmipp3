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

#ifndef _PROG_STATISTICAL_MAP
#define _PROG_STATISTICAL_MAP

#include "core/metadata_vec.h"
#include "core/xmipp_program.h"
#include "core/xmipp_image.h"
#include "core/xmipp_filename.h"
#include "reconstruction/resolution_fscoh.h"

#define VERBOSE_OUTPUT
#define DEBUG_DIM
// #define DEBUG_FREQUENCY_MAP
#define DEBUG_STAT_MAP
#define DEBUG_WEIGHT_MAP
#define DEBUG_WRITE_OUTPUT
#define DEBUG_OUTPUT_FILES
#define DEBUG_PERCENTILE

/**@defgroup ProgStatisticalMap Calculates statistical map
   @ingroup ReconsLibrary */
//@{
/** Calculate statistical map from a pool of input maps and weight input volume*/

class ProgStatisticalMap: public XmippProgram
{
 public:
    // Input params
    FileName fn_mapPool;                    // Input metadata with map pool for analysis
    FileName fn_mapPool_statistical;        // Input metadata with map pool for statistical map calculation
    FileName fn_oroot;                      // Location for saving output maps
    double sampling_rate;                   // Sampling rate of input maps
    double protein_radius;                  // Protein radius
    double significance_thr;                // Significance Z-score threshold

    // Side info variables
    FileName fn_out_avg_map;
    FileName fn_out_std_map;
    FileName fn_out_coincident_maks;
    FileName fn_out_different_maks;

    // Volume dimensions
    bool dimInitialized = false;
    size_t Xdim;
    size_t Ydim;
    size_t Zdim;
    size_t Ndim;

    // Data variables
    FileName fn_V;                          // Filename for each input volume from pool
    Image<double> V;                        // Each input volume from pool
    Image<double> V_Zscores;                // Each z-scores map from pool
    Image<double> avgVolume;                // Average volume
    Image<double> stdVolume;                // Standard deviation volume
    Image<double> avgDiffVolume;            // Average difference volume
    MultidimArray<int> proteinRadiusMask;   // Mask for focus analysis if protein radius provided
    MultidimArray<int> coincidentMask;      // Mask for coincident regions between each input map and the statiscal pool
    MultidimArray<int> differentMask;       // Mask for different regions between each input map and the statiscal pool

    // Calculated parameters
    double percentileThr = 99.99;  // *** Esto vamos a querer que sea un parametro, lo estoy viendo venir
    std::vector<double> histogramEqualizationParameters;
    double equalizationParam;
    double partialOccupancyFactor;

    // Particle metadata
    MetaDataVec mapPoolMD;
    MDRowVec row;

    // FSCoh
    ProgFSCoh fscoh;

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
    void writeMask();

    // ----------------------- MAIN METHODS ------------------------------
    void run();

    // ----------------------- CORE METHODS ------------------------------
    void calculateFSCoh();
    void preprocessMap(FileName fnIn);
    void processStaticalMap();
    void processStaticalMapDixon();
    void computeStatisticalMaps();
    void calculateAvgDiffMap();
    void calculateZscoreMap();
    void calculateDixonMap();
    void weightMap();

    // ---------------------- UTILS METHODS ------------------------------
    // Generate side info
    void generateSideInfo();
    double normal_cdf(double z);
    void createRadiusMask();

    double t_cdf(double t, int nu);
    double t_p_value(double t_stat, int nu);
    double percentile(MultidimArray<double>& data, double p);

};
//@}
#endif