/***************************************************************************
 *
 * Authors:     Federico P. de Isidro-Gomez (federico.pdeisidro@astx.com)
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

 #include "statistical_map.h"
 #include "core/metadata_extension.h"
 #include "core/multidim_array.h"
 #include "core/xmipp_image_base.h"
 #include "core/xmipp_fftw.h"
 #include "data/filters.h"
 #include "data/morphology.h"
 #include <iostream>
 #include <string>
 #include <chrono>
 #include <cmath>
 #include <numeric>



// I/O methods ===================================================================
void ProgStatisticalMap::readParams()
{
    fn_mapPool = getParam("-i");
    fn_mapPool_statistical = getParam("--input_mapPool");
    fn_oroot = getParam("--oroot");
    sampling_rate = getDoubleParam("--sampling_rate");
    protein_radius = getDoubleParam("--protein_radius");
    significance_thr = getDoubleParam("--significance_thr");
}

void ProgStatisticalMap::show() const
{
    if (!verbose)
        return;
	std::cout
    << "RUNNING IN MAD (MEAN AVERAGE DEVIATION) MODE!!!!!!!!!!!!!!!" << std::endl
	<< "Input metadata with map pool for analysis:\t" << fn_mapPool << std::endl
	<< "Input metadata with map pool for statistical map calculation:\t" << fn_mapPool_statistical << std::endl
	<< "Output location for statistical volumes:\t" << fn_oroot << std::endl
	<< "Sampling rate:\t" << sampling_rate << std::endl
	<< "Protein radius:\t" << protein_radius << std::endl
	<< "Significance Z-score threshold:\t" << significance_thr << std::endl;
}

void ProgStatisticalMap::defineParams()
{
	//Usage
    addUsageLine("This algorithm computes a statistical map that characterize the input map pool for posterior comparison \
                  to new map pool to characterize the likelyness of its densities.");

    //Parameters
    addParamsLine("-i <i=\"\">                                : Input metadata containing volumes to analyze against the calculated statical map.");
    addParamsLine("--input_mapPool <input_mapPool=\"\">       : Input metadata containing map pool for statistical map calculation.");
    addParamsLine("--oroot <oroot=\"\">                       : Location for saving output.");
    addParamsLine("--sampling_rate <sampling_rate=1.0>        : Sampling rate of the input of maps.");
    addParamsLine("[--protein_radius <protein_radius=-1>]     : Protein raius (in Angstroms). This is used to restrain the number of pixeles considered in the analysis. By default no pixel is removed from analysis.");
    addParamsLine("[--significance_thr <significance_thr=3>] : Z-score threshold to consider a region significantly different.");
}

void ProgStatisticalMap::writeStatisticalMap() 
{
    avgVolume.write(fn_out_avg_map);
    stdVolume.write(fn_out_std_map);
    #ifdef DEBUG_WRITE_OUTPUT
    std::cout << "Statistical map saved at: " << fn_out_avg_map << " and " << fn_out_std_map<<std::endl;
    #endif
}

void ProgStatisticalMap::writeMedianMap() 
{
    medianMap.write(fn_out_median_map);

    #ifdef DEBUG_WRITE_OUTPUT
    std::cout << "Median map saved at: " << fn_out_median_map << std::endl;
    #endif
}

void ProgStatisticalMap::writeMadMap() 
{
    MADMap.write(fn_out_mad_map);

    #ifdef DEBUG_WRITE_OUTPUT
    std::cout << "MAD map saved at: " << fn_out_mad_map << std::endl;
    #endif
}

void ProgStatisticalMap::writeZscoresMap(FileName fnIn)
{
    // ---- Basic sanity checks ----
    if (fn_oroot.empty())
        throw std::runtime_error("Output root directory 'fn_oroot' is empty.");

    // ---- Extract base name (without path, without extension) ----
    size_t lastSlashPos = fnIn.find_last_of("/\\");
    if (lastSlashPos == FileName::npos) {
        lastSlashPos = static_cast<size_t>(-1); // so lastSlashPos + 1 == 0
    }

    size_t lastDotPos = fnIn.find_last_of('.');
    if (lastDotPos == FileName::npos || lastDotPos <= lastSlashPos) {
        // No extension or dot is part of the path segment: treat whole filename as base
        lastDotPos = fnIn.size();
    }

    FileName base = fnIn.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1);

    // ---- Build the output directory prefix with exactly one separator ----
    const bool rootHasSep = (fn_oroot.back() == '/' || fn_oroot.back() == '\\');
    const FileName sep = rootHasSep ? "" : FileName("/");

    // ---- Try candidates until we find a free name ----
    FileName fnOut;
    for (int counter = 0;; ++counter) {
        if (counter == 0) {
            fnOut = fn_oroot + sep + base + "_Zscores.mrc";
        } else {
            fnOut = fn_oroot + sep + base + "_Zscores_" + std::to_string(counter) + ".mrc";
        }

        // If file does not exist, we can write to it
        std::ifstream test(fnOut);
        if (!test.good()) {
            break;
        }
        // else keep looping and try the next suffix
    }

    // ---- Write output Z-scores volume ----
       V_Zscores.write(fnOut);

    #ifdef DEBUG_WRITE_OUTPUT
    std::cout << "    Z-scores map saved at: " << fnOut << std::endl;
    #endif
}

void ProgStatisticalMap::writeZscoresMADMap(FileName fnIn)
{
    // ---- Basic sanity checks ----
    if (fn_oroot.empty())
        throw std::runtime_error("Output root directory 'fn_oroot' is empty.");

    // ---- Extract base name (without path, without extension) ----
    size_t lastSlashPos = fnIn.find_last_of("/\\");
    if (lastSlashPos == FileName::npos) {
        lastSlashPos = static_cast<size_t>(-1); // ensures substr starts at 0
    }

    size_t lastDotPos = fnIn.find_last_of('.');
    if (lastDotPos == FileName::npos || lastDotPos <= lastSlashPos) {
        // No extension or dot is part of the path segment: treat whole filename as base
        lastDotPos = fnIn.size();
    }

    FileName base = fnIn.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1);

    // ---- Build the output directory prefix with exactly one separator ----
    const bool rootHasSep = (fn_oroot.back() == '/' || fn_oroot.back() == '\\');
    const FileName sep = rootHasSep ? "" : FileName("/");

    // ---- Try candidates until we find a free name ----
    FileName fnOut;
    for (int counter = 0;; ++counter) {
        if (counter == 0) {
            fnOut = fn_oroot + sep + base + "_ZscoresMAD.mrc";
        } else {
            fnOut = fn_oroot + sep + base + "_ZscoresMAD_" + std::to_string(counter) + ".mrc";
        }

        std::ifstream test(fnOut);
        if (!test.good()) {
            break; // free name found
        }
        // else keep looping and try the next suffix
    }

    // ---- Write output Z-scores MAD volume ----
    V_ZscoresMAD.write(fnOut);

    #ifdef DEBUG_WRITE_OUTPUT
    std::cout << "    Z-scores MAD map saved at: " << fnOut << std::endl;
       #endif
}

void ProgStatisticalMap::writeWeightedMap(FileName fnIn) 
{
    // Compose filename
    size_t lastSlashPos = fnIn.find_last_of("/\\");
    size_t lastDotPos = fnIn.find_last_of('.');

    FileName newFileName = fnIn.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1) + "_weighted.mrc";
    FileName fnOut = fn_oroot + (fn_oroot.back() == '/' || fn_oroot.back() == '\\' ? "" : "/") + newFileName;

    // Check if file already existes (the same pool map might contain to identical filenames
    int counter = 1;
    while (std::ifstream(fnOut)) 
    {
        fnOut = fn_oroot + (fn_oroot.back() == '/' || fn_oroot.back() == '\\' ? "" : "/") + fnIn.substr(fnIn.find_last_of("/\\") + 1, fnIn.find_last_of('.') - fnIn.find_last_of("/\\") - 1) + "_weighted_" + std::to_string(counter++) + ".mrc";
    }

    //Write output weighted volume
    V.write(fnOut);
}


void ProgStatisticalMap::writeMask(FileName fnIn)
{
    // Compose filename base
    size_t lastSlashPos = fnIn.find_last_of("/\\");
    size_t lastDotPos   = fnIn.find_last_of('.');

    const FileName baseName = fnIn.substr(
        lastSlashPos == FileName::npos ? 0 : lastSlashPos + 1,
        (lastDotPos == FileName::npos ? fnIn.size() : lastDotPos) -
        (lastSlashPos == FileName::npos ? 0 : lastSlashPos + 1)
    );

    const bool rootHasSep = !fn_oroot.empty() &&
                            (fn_oroot.back() == '/' || fn_oroot.back() == '\\');
    const FileName rootPrefix = fn_oroot + (rootHasSep ? "" : "/");

    // 1) Coincident Mask
    FileName fn_out_coincident_mask = rootPrefix + baseName + "_coincidentMask.mrc";
    {
        int counter = 1;
        while (std::ifstream(fn_out_coincident_mask))
            fn_out_coincident_mask = rootPrefix + baseName + "_coincidentMask_" + std::to_string(counter++) + ".mrc";
    }

    // 2) Different Mask
    FileName fn_out_different_mask = rootPrefix + baseName + "_differentMask.mrc";
    {
        int counter = 1;
        while (std::ifstream(fn_out_different_mask))
            fn_out_different_mask = rootPrefix + baseName + "_differentMask_" + std::to_string(counter++) + ".mrc";
    }

    // 3) Coincident Distance Mask
    FileName fn_out_coincident_dist = rootPrefix + baseName + "_coincidentDistance.mrc";
    {
        int counter = 1;
        while (std::ifstream(fn_out_coincident_dist))
            fn_out_coincident_dist = rootPrefix + baseName + "_coincidentDistance_" + std::to_string(counter++) + ".mrc";
    }

    // 4) Different Distance Mask
    FileName fn_out_different_dist = rootPrefix + baseName + "_differentDistance.mrc";
    {
        int counter = 1;
        while (std::ifstream(fn_out_different_dist))
            fn_out_different_dist = rootPrefix + baseName + "_differentDistance_" + std::to_string(counter++) + ".mrc";
    }

    // Binary masks (int)
    {
        Image<int> saveMask;
        saveMask() = coincidentMask;
        saveMask.write(fn_out_coincident_mask);

        saveMask() = differentMask;
        saveMask.write(fn_out_different_mask);
    }

    // Distance masks (double)
    {
        Image<double> saveDist;
        saveDist() = distanceCoincidentMask;
        saveDist.write(fn_out_coincident_dist);

        saveDist() = distanceDifferentMask;
        saveDist.write(fn_out_different_dist);
    }

    #ifdef DEBUG_WRITE_OUTPUT
    std::cout << "Coincident mask saved at: " << fn_out_coincident_mask << std::endl;
    std::cout << "Different  mask saved at: " << fn_out_different_mask << std::endl;
    std::cout << "Coincident distance mask saved at: " << fn_out_coincident_dist << std::endl;
    std::cout << "Different  distance mask saved at: " << fn_out_different_dist << std::endl;
    #endif
}



// Main method ===================================================================
void ProgStatisticalMap::run()
{
	auto t1 = std::chrono::high_resolution_clock::now();
    show();

    calculateFSCoh();

    // ---
    // Calculate statistical map
    // ---
    #ifdef VERBOSE_OUTPUT
    std::cout << "\n\n---Analyzing input map pool for statistical characterization---" << std::endl;
    #endif
    
    mapPoolMD.read(fn_mapPool_statistical);
    Ndim = mapPoolMD.size();
    size_t volCounter = 0;

    for (const auto& row : mapPoolMD)
	{
        row.getValue(MDL_IMAGE, fn_V);

        #ifdef DEBUG_STAT_MAP
        std::cout << "Processing volume " << fn_V << " from statistical map pool..." << std::endl;
        #endif

        V.clear();
        V.read(fn_V); 

        if (!dimInitialized)
        {
            // Generate side info
            generateSideInfo();

            #ifdef DEBUG_DIM
            std::cout 
            << "Xdim: " << Xdim << std::endl
            << "Ydim: " << Ydim << std::endl
            << "Zdim: " << Zdim << std::endl
            << "Ndim: " << Ndim << std::endl;
            #endif

            referenceMapPool().initZeros(Ndim, Zdim, Ydim, Xdim);
            avgVolume().initZeros(Zdim, Ydim, Xdim);
            stdVolume().initZeros(Zdim, Ydim, Xdim);
            // avgDiffVolume().initZeros(Zdim, Ydim, Xdim);
            medianMap().initZeros(Zdim, Ydim, Xdim);
            MADMap().initZeros(Zdim, Ydim, Xdim);
            V_ZscoresMAD().initZeros(Zdim, Ydim, Xdim);

            dimInitialized = true;
        }

        preprocessMap(fn_V);
        processStaticalMap();

        // Load preprocess map in reference map pool
        for(size_t k = 0; k < Zdim; k++)
	    {
            for(size_t j = 0; j <Xdim; j++)
            {
                for(size_t i = 0; i < Ydim; i++)
                {
                    DIRECT_NZYX_ELEM(referenceMapPool(), volCounter, k, i, j) = DIRECT_ZYX_ELEM(V(), k, i, j);
                }
            }
        }

        volCounter++;
    }

    // Calculate median map
    computeMedianMap();
    writeMedianMap();

    computemMADMap();
    writeMadMap();

    computeStatisticalMaps();
    writeStatisticalMap();

    // calculateAvgDiffMap();

    #ifdef DEBUG_STAT_MAP
    std::cout << "Statistical map succesfully calculated!" << std::endl;
    #endif
    
    // ---
    // Calculate Z-score maps from statistical map pool for histogram equalization
    // ---
    #ifdef VERBOSE_OUTPUT
    std::cout << "\n\n---Analyzing input map pool for histogram equalization---" << std::endl;
    #endif

    volCounter = 0;

    for (const auto& row : mapPoolMD)
	{
        row.getValue(MDL_IMAGE, fn_V);

        #ifdef DEBUG_WEIGHT_MAP
        std::cout << "Anayzing volume " << fn_V << " against statistical map for histogram equalization..." << std::endl;
        #endif

        // Load preprocess map from reference map pool
        V().initZeros(Zdim, Ydim, Xdim);

        for(size_t k = 0; k < Zdim; k++)
	    {
            for(size_t j = 0; j <Xdim; j++)
            {
                for(size_t i = 0; i < Ydim; i++)
                {
                     DIRECT_ZYX_ELEM(V(), k, i, j) = DIRECT_NZYX_ELEM(referenceMapPool(), volCounter, k, i, j);
                }
            }
        }

        V_Zscores().initZeros(Zdim, Ydim, Xdim);
        differentMask.initZeros(Zdim, Ydim, Xdim);
        coincidentMask.initZeros(Zdim, Ydim, Xdim);
        V_ZscoresMAD().initZeros(Zdim, Ydim, Xdim);
        calculateZscoreMADMap();
        writeZscoresMADMap(fn_V);

        writeZscoresMap(fn_V);
        // writeMask(fn_V);

        volCounter++;
    }

    // ---
    // Compare input maps against statistical map
    // ---
    #ifdef VERBOSE_OUTPUT
    std::cout << "\n\n---Comparing input map pool against statistical map---" << std::endl;
    #endif

    mapPoolMD.read(fn_mapPool);

    for (const auto& row : mapPoolMD)
	{
        row.getValue(MDL_IMAGE, fn_V);

        #ifdef DEBUG_WEIGHT_MAP
        std::cout << "Anayzing volume " << fn_V << " against statistical map..." << std::endl;
        #endif

        V.clear();
        V.read(fn_V);

        preprocessMap(fn_V);

        V_Zscores().initZeros(Zdim, Ydim, Xdim);
        coincidentMask.initZeros(Zdim, Ydim, Xdim);
        differentMask.initZeros(Zdim, Ydim, Xdim);
        V_ZscoresMAD().initZeros(Zdim, Ydim, Xdim);
        
        calculateZscoreMADMap();
        writeZscoresMADMap(fn_V);
        writeZscoresMap(fn_V);

        weightMap();
        writeWeightedMap(fn_V);
        writeMask(fn_V);
    }

    #ifdef DEBUG_WEIGHT_MAP
    std::cout << "Input maps succesfully analyzed!" << std::endl;
    #endif

    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

 	std::cout << "Execution time: " << ms_int.count() << " ms" << std::endl;
}


// Core methods ===================================================================
void ProgStatisticalMap::calculateFSCoh()
{
	// Initialize FSCoh detector
    fscoh.fn_mapPool = fn_mapPool_statistical;
    fscoh.fn_oroot = fn_oroot;
    fscoh.sampling_rate = sampling_rate;

	#ifdef VERBOSE_OUTPUT
	std::cout << "----- Calculate FSCoh" << std::endl;
	#endif

	fscoh.run();

	#ifdef VERBOSE_OUTPUT
	std::cout << "----- FSCoh caluclated successfully!" << std::endl;
	#endif
}

void ProgStatisticalMap::preprocessMap(FileName fnIn)
{
    std::cout << "    Preprocessing input map..." << std::endl;

    // LPF map up to coherent resolution threshold (remove uncoherent frequencies)
    FourierTransformer ft;
    MultidimArray<std::complex<double>> V_ft;
	ft.FourierTransform(V(), V_ft, false);

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V_ft)
    {
        if (DIRECT_MULTIDIM_ELEM(fscoh.freqMap, n) > fscoh.indexThr)
        {
            DIRECT_MULTIDIM_ELEM(V_ft,  n) = 0;
        }
    }

    ft.inverseFourierTransform();

    std::cout << "    Low-pass filtering applied up to frequency index: " << fscoh.indexThr << std::endl;

    // Create mask with only positive values (std>1 in protein radius)
    double foo;
    double std;
    V().computeAvgStdev_within_binary_mask(proteinRadiusMask, foo, std);

    positiveMask.initZeros(Zdim, Ydim, Xdim);

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        if (DIRECT_MULTIDIM_ELEM(V(), n) > std && DIRECT_MULTIDIM_ELEM(proteinRadiusMask, n) > 0)
        {
            DIRECT_MULTIDIM_ELEM(positiveMask, n) = 1;
        }
    }
   
    std::cout << "    Positive density mask created." << std::endl;

    // Normalize map on positive densities dividing by std
    V().computeAvgStdev_within_binary_mask(positiveMask, foo, std);
    std::cout << "    Normalizing map by stdev value of positive densities: " << std << std::endl;

    // Normalize map
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        DIRECT_MULTIDIM_ELEM(V(), n) = DIRECT_MULTIDIM_ELEM(V(), n) / std;
    }

    #ifdef DEBUG_OUTPUT_FILES

    // Build base name (no path, no extension)
    const size_t s = fnIn.find_last_of("/\\");
    const size_t d = fnIn.find_last_of('.');
    const size_t start = (s == FileName::npos ? 0 : s + 1);
    const size_t end = (d == FileName::npos || d <= s ? fnIn.size() : d);
    const FileName base = fnIn.substr(start, end - start);

    // Build output with unique suffix
    const bool hasSep = (!fn_oroot.empty() && (fn_oroot.back() == '/' || fn_oroot.back() == '\\'));
    const FileName sep = hasSep ? "" : "/";
    FileName fnOut;
    for (int i = 0;; ++i) {
        fnOut = fn_oroot + sep + base + "_preprocess" + (i ? "_" + std
            ::to_string(i) : "") + ".mrc";
        std::ifstream f(fnOut);
        if (!f.good()) break;
    }

    V.write(fnOut);
    #endif

}

void ProgStatisticalMap::processStaticalMap()
{ 
    std::cout << "    Processing input map for statistical map calculation..." << std::endl;

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        // Reuse avg and std maps for sum and sum^2 (memory efficient)
        double value = DIRECT_MULTIDIM_ELEM(V(),n);
        DIRECT_MULTIDIM_ELEM(avgVolume(),n) += value;   // sum
        DIRECT_MULTIDIM_ELEM(stdVolume(),n) += value * value;   // sum squared
    }
}

void ProgStatisticalMap::computeMedianMap()
{ 
    std::cout << "    Calculating median map..." << std::endl;

    std::vector<double> voxelValues;
    voxelValues.reserve(Ndim);

    // Iterate for every voxel in the volume's shape
    for(size_t k = 0; k < Zdim; k++)
	{
        for(size_t j = 0; j <Xdim; j++)
		{
			for(size_t i = 0; i < Ydim; i++)
            {
                if (DIRECT_ZYX_ELEM(proteinRadiusMask, k, i, j) > 0)
                {
                    for (size_t n = 0; n < Ndim; n++)
                    {
                        voxelValues.push_back(DIRECT_NZYX_ELEM(referenceMapPool(), n, k, i, j));
                    }

                    DIRECT_ZYX_ELEM(medianMap(), k, i, j) = median(voxelValues);
                    voxelValues.clear();
                }
            }
        }
    }
}

void ProgStatisticalMap::computemMADMap()
{ 
    std::cout << "    Calculating MAD map..." << std::endl;

    std::vector<double> voxelValues;
    voxelValues.reserve(Ndim);

    // Iterate for every voxel in the volume's shape
    for(size_t k = 0; k < Zdim; k++)
	{
        for(size_t j = 0; j <Xdim; j++)
		{
			for(size_t i = 0; i < Ydim; i++)
            {
                if (DIRECT_ZYX_ELEM(proteinRadiusMask, k, i, j) > 0)
                {
                    for (size_t n = 0; n < Ndim; n++)
                    {
                        voxelValues.push_back(abs(DIRECT_NZYX_ELEM(referenceMapPool(), n, k, i, j) - DIRECT_ZYX_ELEM(medianMap(), k, i, j)));
                    }

                    DIRECT_ZYX_ELEM(MADMap(), k, i, j) = median(voxelValues);
                    voxelValues.clear();
                }                
            }
        }
    }
}


void ProgStatisticalMap::computeStatisticalMaps()
{
    std::cout << "Computing statisical map..." << std::endl;

    // Compute mean and standard deviation maps
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(avgVolume())
    {
        double sum  = DIRECT_MULTIDIM_ELEM(avgVolume(),n);
        double sum2 = DIRECT_MULTIDIM_ELEM(stdVolume(),n);
        double mean = sum/Ndim;

        DIRECT_MULTIDIM_ELEM(avgVolume(),n) = mean;
        DIRECT_MULTIDIM_ELEM(stdVolume(),n) = sqrt(sum2/Ndim - mean*mean);
    }

    // Update positive mask from average map for posterior analysis (std>1 in protein radius)
    positiveMask.initZeros(Zdim, Ydim, Xdim);

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(avgVolume())
    {
        // As maps are normalized to std=1 the comparison is direct
        if (DIRECT_MULTIDIM_ELEM(avgVolume(), n) > 1 && DIRECT_MULTIDIM_ELEM(proteinRadiusMask, n) > 0)
        {
            DIRECT_MULTIDIM_ELEM(positiveMask, n) = 1;
        }
    }

    #ifdef DEBUG_OUTPUT_FILES
    Image<int> saveImage;
    std::string debugFileFn = fn_oroot + "positiveMask.mrc";
    saveImage() = positiveMask;
    saveImage.write(debugFileFn);
    #endif   
}

void ProgStatisticalMap::calculateAvgDiffMap()
{
    for (const auto& row : mapPoolMD)
	{
        row.getValue(MDL_IMAGE, fn_V);
        V.read(fn_V); 

        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(avgDiffVolume())
        {
            DIRECT_MULTIDIM_ELEM(avgDiffVolume(),n) =  DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n);
        }
    }

    avgDiffVolume() /= Ndim;

    avgDiffVolume.write(fn_oroot + "statsMap_avgDiff.mrc");
}

void ProgStatisticalMap::computeSigmaNormMAD(double& sigmaNorm) 
{
    // Calculate diff map
    MultidimArray<double> diffMap;
    diffMap.initZeros(Zdim, Ydim, Xdim);

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(diffMap)
    {
        if (DIRECT_MULTIDIM_ELEM(proteinRadiusMask, n) > 0)
        {
            DIRECT_MULTIDIM_ELEM(diffMap, n) = DIRECT_MULTIDIM_ELEM(V(), n) - DIRECT_MULTIDIM_ELEM(medianMap(), n);
        }
    }

    // Calculate abosule difference map median
    double median;
    diffMap.computeMedian_within_binary_mask(proteinRadiusMask, median);

    // Compute absolute deviations from the median
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(diffMap)
    {
        if (DIRECT_MULTIDIM_ELEM(proteinRadiusMask, n) > 0)
        {
            DIRECT_MULTIDIM_ELEM(diffMap, n) = std::fabs(DIRECT_MULTIDIM_ELEM(diffMap, n) - median);
        }
    }

    // Compute MAD
    double mad;
    diffMap.computeMedian_within_binary_mask(proteinRadiusMask, mad);
    
    // Scale MAD to estimate sigma under normality
    // For a normal distribution, MAD ≈ 0.6745 * sigma.
    // So sigma ≈ MAD / 0.6745 ≈ MAD * 1.4826.
    sigmaNorm = 1.4826 * mad;

    #ifdef DEBUG_SIGMA_NORM
    std::cout << "    Sigma normalization factor (MAD): " << sigmaNorm << std::endl;
    #endif
}


void ProgStatisticalMap::computeSigmaNormIQR(double& sigmaNorm) {
     // Calculate diff map
    std::vector<double> diffs;

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(coincidentMask)
    {
        if (DIRECT_MULTIDIM_ELEM(coincidentMask, n) > 0)
        {
            diffs.push_back(DIRECT_MULTIDIM_ELEM(V(), n) - DIRECT_MULTIDIM_ELEM(avgVolume(), n));
        }
    }

    // Sort map values
    std::sort(diffs.begin(), diffs.end());

    // Calculate IQR
    size_t n = diffs.size();
    double q1 = diffs[n/4];
    double q3 = diffs[(3*n)/4];
    double iqr = q3 - q1;

    sigmaNorm = iqr / 1.349;

    #ifdef DEBUG_SIGMA_NORM
    std::cout << "    Sigma normalization factor (IQR): " << sigmaNorm << std::endl;
    #endif
}

void ProgStatisticalMap::calculateZscoreMADMap()
{
    std::cout << "    Calculating Zscore MAD map..." << std::endl;

    double mapMAD;
    double foo;
    // computeSigmaNormMAD(mapMAD);
    MADMap().computeAvgStdev_within_binary_mask(positiveMask, mapMAD, foo);
    mapMAD = mapMAD * 1.4826; // Scale MAD to estimate sigma under normality

    std::cout << "    Global MAD value: " << mapMAD << std::endl;

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        if (DIRECT_MULTIDIM_ELEM(proteinRadiusMask,n) > 0)
        {
            double mad_local = DIRECT_MULTIDIM_ELEM(MADMap(), n);
            mad_local = mad_local * 1.4826; // Scale MAD to estimate sigma under normality
            double median_local  = DIRECT_MULTIDIM_ELEM(medianMap(), n);
            double val = DIRECT_MULTIDIM_ELEM(V(), n);

            double zscoreMAD = 0.0;
            double zscore = 0.0;
            
            zscoreMAD = (val - median_local) / sqrt(mapMAD*mapMAD + mad_local * mad_local);
            // zscoreMAD = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / sqrt(mapMAD*mapMAD + mad_local * mad_local);
            zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / sqrt(mapMAD*mapMAD + DIRECT_MULTIDIM_ELEM(stdVolume(),n)*DIRECT_MULTIDIM_ELEM(stdVolume(),n));
            // zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / sqrt(mad_local*mad_local + DIRECT_MULTIDIM_ELEM(stdVolume(),n)*DIRECT_MULTIDIM_ELEM(stdVolume(),n));
            // zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / sqrt(DIRECT_MULTIDIM_ELEM(stdVolume(),n)*DIRECT_MULTIDIM_ELEM(stdVolume(),n));

            DIRECT_MULTIDIM_ELEM(V_ZscoresMAD(), n) = zscoreMAD;
            DIRECT_MULTIDIM_ELEM(V_Zscores(), n) = zscore;
        }
    }

    // Calculate different mask
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V_Zscores())
    {
        if (DIRECT_MULTIDIM_ELEM(proteinRadiusMask,n) > 0)
        {
            if (DIRECT_MULTIDIM_ELEM(V_Zscores(), n) > 3.0)
            {
                DIRECT_MULTIDIM_ELEM(differentMask,n) = 1;
            }
        }
    }

    // Calculate coincident mask via "cosine average"
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        if (DIRECT_MULTIDIM_ELEM(positiveMask,n) > 0)
        {
            double num = (DIRECT_MULTIDIM_ELEM(V(),n) * DIRECT_MULTIDIM_ELEM(avgVolume(), n));
            double dem = sqrt((DIRECT_MULTIDIM_ELEM(V(),n) * DIRECT_MULTIDIM_ELEM(V(), n)) + (DIRECT_MULTIDIM_ELEM(avgVolume(),n) * DIRECT_MULTIDIM_ELEM(avgVolume(), n)));
            double div = num / dem;

            // Using 2.1213 as threshold corresponds a consistent intensity of 3 standard deviations between both maps
            if (div > 2.1213)
            {
                DIRECT_MULTIDIM_ELEM(coincidentMask,n) = 1;
            }             
        }
    }

    ///////////////////////////////////////////////////////////////////////////////


// // --- Empirical null (median + MAD) on full masked z-map ---
// std::vector<double> tmp;

// FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V_Zscores())
// {
//     if (DIRECT_MULTIDIM_ELEM(proteinRadiusMask, n) > 0)
//         tmp.push_back(DIRECT_MULTIDIM_ELEM(V_Zscores, n));
// }

// double z_threshold = std::numeric_limits<double>::quiet_NaN();
// const size_t N = tmp.size();
// if (N == 0) {
//     // handle empty mask
// } else {
//     // Median (mu0)
//     std::vector<double> w = tmp;
//     std::nth_element(w.begin(), w.begin() + N/2, w.end());
//     double mu0 = w[N/2];
//     if (N % 2 == 0) {
//         std::nth_element(w.begin(), w.begin() + N/2 - 1, w.end());
//         mu0 = 0.5 * (mu0 + w[N/2 - 1]);
//     }

//     // MAD -> sigma0
//     std::vector<double> absdev(N);
//     for (size_t i = 0; i < N; ++i)
//         absdev[i] = std::abs(tmp[i] - mu0);

//     std::nth_element(absdev.begin(), absdev.begin() + N/2, absdev.end());
//     double mad = absdev[N/2];
//     if (N % 2 == 0) {
//         std::nth_element(absdev.begin(), absdev.begin() + N/2 - 1, absdev.end());
//         mad = 0.5 * (mad + absdev[N/2 - 1]);
//     }

//     double sigma0 = 1.4826 * mad;
//     if (sigma0 < 1e-12) sigma0 = 1e-12;

//     // Choose sigma_ref
//     // Option 1 (recommended): from calibration rigid maps (pass it in)
//     // double sigma_ref = sigma_ref_from_rigid;
//     // Option 2 (no calibration available): do not be more permissive than classic z
//     double sigma_ref = 1.0;

//     double sigma0_star = std::max(sigma0, sigma_ref);

//     // Final threshold in ORIGINAL z* space (one-tailed positive)
//     z_threshold = mu0 + 3.0 * sigma0_star;

//     std::cout << "    z-threshold (empirical-null, one-tailed, 3σ): " << z_threshold
//               << "  [mu0=" << mu0 << ", sigma0=" << sigma0
//               << ", sigma_ref=" << sigma_ref << "]\n";
// }

 ///////////////////////////////////////////////////////////////////////////////

    std::cout << "    Zscore MAD map calculated successfully!" << std::endl;
}



void ProgStatisticalMap::weightMap()
{ 
    std::cout << "    Calculating weighted map..." << std::endl;

    // Parámetros
    const double EPS   = 1e-6;  // excluir fuera/borde
    const double K_EXP = 2.0;   // agresividad del peso: prueba 2 o 4 si quieres más efecto

    // 1) Distancias RAW (truco: tao=1.0 => d_norm = d_raw)
    generateDistanceMask(coincidentMask, distanceCoincidentMask, 1.0); // d_raw
    generateDistanceMask(differentMask,  distanceDifferentMask,  1.0); // d_raw

    // 2) Construye vectores con distancias RAW > EPS dentro de cada ROI (filtrado fuera del percentile)
    std::vector<double> distCoinPos; distCoinPos.reserve(4096);
    std::vector<double> distDiffPos; distDiffPos.reserve(4096);

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(distanceCoincidentMask)
    {
        if (DIRECT_MULTIDIM_ELEM(coincidentMask, n) > 0) {
            double d = DIRECT_MULTIDIM_ELEM(distanceCoincidentMask, n);
            if (d > EPS) distCoinPos.push_back(d);
        }
    }
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(distanceDifferentMask)
    {
        if (DIRECT_MULTIDIM_ELEM(differentMask, n) > 0) {
            double d = DIRECT_MULTIDIM_ELEM(distanceDifferentMask, n);
            if (d > EPS) distDiffPos.push_back(d);
        }
    }

    // 3) tau = min(P95_tight, P95_loose) sobre distancias RAW > 0
    double tauA = distCoinPos.empty() ? 0.0 : percentile(distCoinPos, 95.0);
    double tauB = distDiffPos.empty() ? 0.0 : percentile(distDiffPos, 95.0);
    double tao  = std::min(tauA, tauB);

    // Salvaguardas: si queda <= 1, forzamos algo > 1 para evitar saturación total
    if (tao <= 1.0 + 1e-12) {
        // Opción conservadora: forzar sqrt(2)
        tao = 1.41421356237; // sqrt(2)
        std::cerr << "  [WARN] tau (P95 tight-anchored) <= 1; forcing tau = sqrt(2)\n";
    }
    std::cout << "  Calculated tau (tight-anchored P95): " << tao << std::endl;

    // 4) Medias ponderadas con w = min(1, (d_raw/tau)^k), excluyendo d_raw <= EPS
    double coincident_num = 0.0, coincident_den = 0.0;
    double different_num  = 0.0, different_den  = 0.0;

    // Diagnóstico: porcentaje de pesos saturados (w==1)
    size_t C_total=0, C_full=0, D_total=0, D_full=0;

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        // ROI coincidente
        if (DIRECT_MULTIDIM_ELEM(coincidentMask, n) > 0)
        {
            double d_raw = DIRECT_MULTIDIM_ELEM(distanceCoincidentMask, n);
            if (d_raw > EPS)
            {
                double d_norm = d_raw / tao;
                double w = std::min(1.0, std::pow(d_norm, K_EXP));
                coincident_num += w * DIRECT_MULTIDIM_ELEM(V(), n);
                coincident_den += w;

                ++C_total; if (w >= 1.0 - 1e-12) ++C_full;
            }
        }

        // ROI diferente
        if (DIRECT_MULTIDIM_ELEM(differentMask, n) > 0)
        {
            double d_raw = DIRECT_MULTIDIM_ELEM(distanceDifferentMask, n);
            if (d_raw > EPS)
            {
                double d_norm = d_raw / tao;
                double w = std::min(1.0, std::pow(d_norm, K_EXP));
                different_num += w * DIRECT_MULTIDIM_ELEM(V(), n);
                different_den += w;

                ++D_total; if (w >= 1.0 - 1e-12) ++D_full;
            }
        }
    }

    double coincident_avg = (coincident_den > 0.0) ? (coincident_num / coincident_den)
                                                   : std::numeric_limits<double>::quiet_NaN();
    double different_avg  = (different_den  > 0.0) ? (different_num  / different_den)
                                                   : std::numeric_limits<double>::quiet_NaN();
    double partialOccupancyFactor = different_avg / coincident_avg;

    std::cout << "  coincident_avg ---------------------> " << coincident_avg << std::endl;
    std::cout << "  different_avg  ---------------------> " << different_avg  << std::endl;
    std::cout << "  partialOccupancyFactor -------------> " << partialOccupancyFactor << std::endl;

    // Diagnóstico de saturación
    if (C_total > 0 || D_total > 0) {
        double pctC = (C_total>0) ? (100.0 * C_full / (double)C_total) : 0.0;
        double pctD = (D_total>0) ? (100.0 * D_full / (double)D_total) : 0.0;
        std::cout << "  [diag] coincident: %w==1 = " << pctC << "% (" << C_full << "/" << C_total << ")\n";
        std::cout << "  [diag] different : %w==1 = " << pctD << "% (" << D_full << "/" << D_total << ")\n";
    }

    // 5) Tu resta final
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        DIRECT_MULTIDIM_ELEM(V(),n) =
            DIRECT_MULTIDIM_ELEM(V(),n) - (DIRECT_MULTIDIM_ELEM(avgVolume(),n) * (1 - partialOccupancyFactor));
    }
}


// Utils methods ===================================================================
void ProgStatisticalMap::generateSideInfo()
{
    Xdim = XSIZE(V());
    Ydim = YSIZE(V());
    Zdim = ZSIZE(V());

    fn_out_avg_map = fn_oroot + "statsMap_avg.mrc";
    fn_out_std_map = fn_oroot + "statsMap_std.mrc";
    fn_out_median_map = fn_oroot + "statsMap_median.mrc";
    fn_out_mad_map = fn_oroot + "statsMap_mad.mrc";

    createRadiusMask();
}

void ProgStatisticalMap::createRadiusMask()
{
    proteinRadiusMask.initZeros(Zdim, Ydim, Xdim);

    if (protein_radius > 0) // If mas radius is provided
    {
        double radiusInPx = protein_radius / sampling_rate;

        // Directional radius along each direction
        double half_Xdim = (Xdim * 1.0) / 2;
        double half_Ydim = (Ydim * 1.0) / 2;
        double half_Zdim = (Zdim * 1.0) / 2;
        double uz;
        double uy;
        double ux;
        double uz2;
        double uz2y2;
        long n=0;

        for(size_t k=0; k<Zdim; ++k)
        {
            uz = k - half_Zdim;
            uz2 = uz*uz;
            
            for(size_t i=0; i<Ydim; ++i)
            {
                uy = i - half_Ydim;
                uz2y2 = uz2 + uy*uy;

                for(size_t j=0; j<Xdim; ++j)
                {
                    ux = j - half_Xdim;
                    ux = sqrt(uz2y2 + ux*ux);

                    if (ux < radiusInPx)
                    {
                        DIRECT_MULTIDIM_ELEM(proteinRadiusMask,n) = 1;
                    }

                    ++n;
                }
            }
        }
    }
    else // If no mask radius is provided, use full map
    {
        proteinRadiusMask.initConstant(1);
    }

    #ifdef DEBUG_OUTPUT_FILES
    Image<int> saveImage;
    std::string debugFileFn = fn_oroot + "proteinRadiusMask.mrc";
    saveImage() = proteinRadiusMask;
    saveImage.write(debugFileFn);
    #endif   
} 


double ProgStatisticalMap::median(std::vector<double> v) 
{
    if (v.empty()) {
        throw std::invalid_argument("Cannot compute median of an empty vector");
    }

    std::sort(v.begin(), v.end());
    size_t n = v.size();

    if (n % 2 == 1) {
        // Odd number of elements
        return static_cast<double>(v[n / 2]);
    } else {
        // Even number of elements
        return (static_cast<double>(v[n / 2 - 1]) + static_cast<double>(v[n / 2])) / 2.0;
    }
}


// ------------------------------------------------------------
// 1D Euclidean Distance Transform (Felzenszwalb & Huttenlocher)
// Entrada: f (0 en background, +INF en foreground)
// Salida:  d = distancia^2 a 0 más cercano a lo largo de la línea
// Complejidad: O(n)
// ------------------------------------------------------------
template<typename T>
static inline void edt_1d(const T* f, T* d, int n)
{
    const T INF = std::numeric_limits<T>::infinity();
    std::vector<int> v(n);
    std::vector<T> z(n + 1);

    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = +INF;

    auto sq = [](T x) { return x * x; };

    for (int q = 1; q < n; ++q)
    {
        T s;
        while (true)
        {
            int p = v[k];
            // Intersección entre parábolas centradas en p y q
            s = ((f[q] + sq((T)q)) - (f[p] + sq((T)p))) / (2 * (q - p));
            if (s <= z[k])
            {
                if (k == 0) break;
                --k;
            }
            else break;
        }
        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = +INF;
    }

    int kk = 0;
    for (int q = 0; q < n; ++q)
    {
        while (z[kk + 1] < q) ++kk;
        T dx = (T)q - (T)v[kk];
        d[q] = dx * dx + f[v[kk]];
    }
}

// ------------------------------------------------------------
// EDT 3D exacto (pases separables X -> Y -> Z) usando tu layout ZYX
// mask: 0 = fondo (fuera ROI), !=0 = interior ROI
// dist2: distancias^2 al fondo tras 3 pases
// ------------------------------------------------------------
static inline void edt3d_exact_zyx(
    MultidimArray<int>& mask,
    MultidimArray<double>& dist2, // salida (dist^2)
    size_t Zdim, size_t Ydim, size_t Xdim)
{
    const double INF = std::numeric_limits<double>::infinity();

    // 0) Inicializa dist2: 0 en fondo, INF en ROI
    dist2.initZeros(Zdim, Ydim, Xdim);
    for (size_t k = 0; k < Zdim; ++k)
        for (size_t j = 0; j < Xdim; ++j)
            for (size_t i = 0; i < Ydim; ++i)
                DIRECT_ZYX_ELEM(dist2, k, i, j) =
                    (DIRECT_ZYX_ELEM(mask, k, i, j) == 0) ? 0.0 : INF;

    // Buffers temporales para líneas (longitud = max dimensión)
    const int Lmax = (int)std::max({ Zdim, Ydim, Xdim });
    std::vector<double> line(Lmax), out(Lmax);

    // --- Pase X: para cada (z,y), línea a lo largo de x=j=0..Xdim-1
    for (size_t k = 0; k < Zdim; ++k)
    {
        for (size_t i = 0; i < Ydim; ++i)
        {
            for (size_t j = 0; j < Xdim; ++j)
                line[(int)j] = DIRECT_ZYX_ELEM(dist2, k, i, j);

            edt_1d(line.data(), out.data(), (int)Xdim);

            for (size_t j = 0; j < Xdim; ++j)
                DIRECT_ZYX_ELEM(dist2, k, i, j) = out[(int)j];
        }
    }

    // --- Pase Y: para cada (z,x), línea a lo largo de y=i=0..Ydim-1
    for (size_t k = 0; k < Zdim; ++k)
    {
        for (size_t j = 0; j < Xdim; ++j)
        {
            for (size_t i = 0; i < Ydim; ++i)
                line[(int)i] = DIRECT_ZYX_ELEM(dist2, k, i, j);

            edt_1d(line.data(), out.data(), (int)Ydim);

            for (size_t i = 0; i < Ydim; ++i)
                DIRECT_ZYX_ELEM(dist2, k, i, j) = out[(int)i];
        }
    }

    // --- Pase Z: para cada (y,x), línea a lo largo de z=k=0..Zdim-1
    for (size_t i = 0; i < Ydim; ++i)
    {
        for (size_t j = 0; j < Xdim; ++j)
        {
            for (size_t k = 0; k < Zdim; ++k)
                line[(int)k] = DIRECT_ZYX_ELEM(dist2, k, i, j);

            edt_1d(line.data(), out.data(), (int)Zdim);

            for (size_t k = 0; k < Zdim; ++k)
                DIRECT_ZYX_ELEM(dist2, k, i, j) = out[(int)k];
        }
    }
}

// ------------------------------------------------------------
// Tu método re‑hecho con EDT exacto y misma firma/sintaxis
// ------------------------------------------------------------
void ProgStatisticalMap::generateDistanceMask(
    MultidimArray<int>&    mask,
    MultidimArray<double>& maskDistance,
    double tao)
{
    // 1) Distancia euclídea exacta al fondo (al cuadrado)
    MultidimArray<double> dist2;
    edt3d_exact_zyx(mask, dist2, Zdim, Ydim, Xdim);

    // 2) Normaliza por tao y deja 0 fuera de ROI
    maskDistance.initZeros(Zdim, Ydim, Xdim);

    for (size_t k = 0; k < Zdim; ++k)
    {
        std::cout << "    Generating distance mask slice " << (k + 1)
                  << " of " << Zdim << "\r" << std::flush;

        for (size_t j = 0; j < Xdim; ++j)
        {
            for (size_t i = 0; i < Ydim; ++i)
            {
                if (DIRECT_ZYX_ELEM(mask, k, i, j) == 0)
                {
                    DIRECT_ZYX_ELEM(maskDistance, k, i, j) = 0.0;
                }
                else
                {
                    const double d2 = DIRECT_ZYX_ELEM(dist2, k, i, j);
                    const double d  = (d2 > 0.0) ? std::sqrt(d2) : 0.0;
                    DIRECT_ZYX_ELEM(maskDistance, k, i, j) =
                        (tao > 0.0) ? (d / tao) : 0.0;
                }
            }
        }
    }
    std::cout << std::endl;
}


double ProgStatisticalMap::percentile(const std::vector<double>& values, double p)
{
    // Caso trivial
    if (values.empty()) return 0.0;

    // Clamp p a [0, 100]
    if (p <= 0.0) return *std::min_element(values.begin(), values.end());
    if (p >= 100.0) return *std::max_element(values.begin(), values.end());

    // Copia y ordena (si prefieres evitar O(n log n), puedo darte versión con nth_element)
    std::vector<double> v = values;
    std::sort(v.begin(), v.end());

    const double pos  = (p / 100.0) * (static_cast<double>(v.size() - 1));
    const size_t i0   = static_cast<size_t>(std::floor(pos));
    const size_t i1   = static_cast<size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(i0);

    const double v0 = v[i0];
    const double v1 = v[i1];

    // Interpolación lineal
    return v0 * (1.0 - frac) + v1 * frac;
}
