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

void ProgStatisticalMap::writePercentileMap(FileName fnIn) 
{
    // Compose filename
    size_t lastSlashPos = fnIn.find_last_of("/\\");
    size_t lastDotPos = fnIn.find_last_of('.');

    FileName newFileName = fnIn.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1) + "_Percentile.mrc";
    FileName fnOut = fn_oroot + (fn_oroot.back() == '/' || fn_oroot.back() == '\\' ? "" : "/") + newFileName;

    // Check if file already existes (the same pool map might contain to identical filenames
    int counter = 1;
    while (std::ifstream(fnOut)) 
    {
        fnOut = fn_oroot + (fn_oroot.back() == '/' || fn_oroot.back() == '\\' ? "" : "/") + fnIn.substr(fnIn.find_last_of("/\\") + 1, fnIn.find_last_of('.') - fnIn.find_last_of("/\\") - 1) + "_Zscores_" + std::to_string(counter++) + fnIn.substr(fnIn.find_last_of('.'));
    }

    //Write output percentile volume
    V_Percentile.write(fnOut);
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
    // Compose filename
    size_t lastSlashPos = fnIn.find_last_of("/\\");
    size_t lastDotPos = fnIn.find_last_of('.');

    // ----------- Coincident Mask -----------
    FileName newFileName = fnIn.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1) + "_coincidentMask.mrc";
    FileName fn_out_coincident_mask = fn_oroot + (fn_oroot.back() == '/' || fn_oroot.back() == '\\' ? "" : "/") + newFileName;

    int counter = 1;
    while (std::ifstream(fn_out_coincident_mask)) 
    {
        fn_out_coincident_mask = fn_oroot + (fn_oroot.back() == '/' || fn_oroot.back() == '\\' ? "" : "/") +
            fnIn.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1) + "_coincidentMask_" + std::to_string(counter++) + ".mrc";
    }

    // ----------- Different Mask -----------
    newFileName = fnIn.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1) + "_differentMask.mrc";
    FileName fn_out_different_mask = fn_oroot + (fn_oroot.back() == '/' || fn_oroot.back() == '\\' ? "" : "/") + newFileName;

    counter = 1;
    while (std::ifstream(fn_out_different_mask)) 
    {
        fn_out_different_mask = fn_oroot + (fn_oroot.back() == '/' || fn_oroot.back() == '\\' ? "" : "/") +
            fnIn.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1) + "_differentMask_" + std::to_string(counter++) + ".mrc";
    }

    // Write output masks
    Image<int> saveMask;
    saveMask() = coincidentMask;
    saveMask.write(fn_out_coincident_mask);
    saveMask() = differentMask;
    saveMask.write(fn_out_different_mask);

    #ifdef DEBUG_WRITE_OUTPUT
    std::cout << "Coincident mask saved at: " << fn_out_coincident_mask << std::endl;
    std::cout << "Different mask saved at: " << fn_out_different_mask << std::endl;
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

    for (const auto& row : mapPoolMD)
	{
        row.getValue(MDL_IMAGE, fn_V);

        #ifdef DEBUG_WEIGHT_MAP
        std::cout << "Anayzing volume " << fn_V << " against statistical map for histogram equalization..." << std::endl;
        #endif

        V.clear();
        V.read(fn_V);

        preprocessMap(fn_V);

        V_Zscores().initZeros(Zdim, Ydim, Xdim);
        // V_Percentile().initZeros(Zdim, Ydim, Xdim);
        // differentMask.initZeros(Zdim, Ydim, Xdim);
        // coincidentMask.initZeros(Zdim, Ydim, Xdim);
        V_ZscoresMAD().initZeros(Zdim, Ydim, Xdim);
        calculateZscoreMADMap();
        writeZscoresMADMap(fn_V);

        // calculateZscoreMap();
        // calculateZscoreMap_GlobalSigma();
        writeZscoresMap(fn_V);
        // writeMask(fn_V);

        // double p = percentile(V_Zscores(), percentileThr);
        // histogramEqualizationParameters.push_back(p);
        
        // FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V_Zscores())
        // {
        //     if (DIRECT_MULTIDIM_ELEM(proteinRadiusMask, n) > 0)
        //     {
        //         zScoreAccumulator.push_back(DIRECT_MULTIDIM_ELEM(V_Zscores(), n));
        //     }
        // }

        // double min;
        // double max;
        // V_Zscores().computeDoubleMinMax(min, max);

        // #ifdef DEBUG_PERCENTILE
        // std::cout << "Max value in Z-score map: " << max << std::endl;
        // #endif

        // histogramEqualizationParameters.push_back(max);        
    }

    // Sort accumulated z-scores for percentile calculation
    // std::sort(zScoreAccumulator.begin(), zScoreAccumulator.end());

    // // Calculate average transformation
    // double sum = std::accumulate(histogramEqualizationParameters.begin(), histogramEqualizationParameters.end(), 0.0);
    // equalizationParam =  sum / histogramEqualizationParameters.size();

    // #ifdef DEBUG_PERCENTILE
    // std::cout << "Equalization parameter: " << equalizationParam << std::endl;
    // #endif

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
        // V_Percentile().initZeros(Zdim, Ydim, Xdim);
        // coincidentMask.initZeros(Zdim, Ydim, Xdim);
        // differentMask.initZeros(Zdim, Ydim, Xdim);
        V_ZscoresMAD().initZeros(Zdim, Ydim, Xdim);
        calculateZscoreMADMap();
        writeZscoresMADMap(fn_V);

        // calculateZscoreMap();
        // calculateZscoreMap_GlobalSigma();
        writeZscoresMap(fn_V);
        // calculatePercetileMap();
        // writePercentileMap(fn_V);

        // weightMap();
        // writeWeightedMap(fn_V);
        // writeMask(fn_V);
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

    // Create mask with only positive values
    positiveMask.initZeros(Zdim, Ydim, Xdim);

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        if (DIRECT_MULTIDIM_ELEM(V(), n) > 0 && DIRECT_MULTIDIM_ELEM(proteinRadiusMask, n) > 0)
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
    
    std::cout << "    Positive density mask created." << std::endl;

    double factor;

    // Normalize map on positive densities dividing by median
    // V().computeMedian_within_binary_mask(positiveMask, factor);
    // std::cout << "    Normalizing map by median value of positive densities: " << factor << std::endl;

    // Normalize map on positive densities dividing by std
    double foo;
    V().computeAvgStdev_within_binary_mask(positiveMask, foo, factor);
    std::cout << "    Normalizing map by stdev value of positive densities: " << factor << std::endl;

    // Normalize map and set negative densities to 0
    // FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    // {
    //     if (DIRECT_MULTIDIM_ELEM(positiveMask, n) > 0)
    //     {
    //         DIRECT_MULTIDIM_ELEM(V(), n) = DIRECT_MULTIDIM_ELEM(V(), n) / factor;
    //     }
    //     else
    //     {
    //         DIRECT_MULTIDIM_ELEM(V(), n) = 0;
    //     }
    // }

    // Normalize map
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        DIRECT_MULTIDIM_ELEM(V(), n) = DIRECT_MULTIDIM_ELEM(V(), n) / factor;
    }

    // Normalize map: mean=0, std=1
    // double avg;
    // double std;
    
    // if (protein_radius < 0)
    // {
    //     V().computeAvgStdev(avg, std);

    //     FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    //     {
    //         DIRECT_MULTIDIM_ELEM(V(), n) = (DIRECT_MULTIDIM_ELEM(V(), n) - avg) / std;
    //     }
    // }
    // else
    // {
    //     V().computeMedian_within_binary_mask(proteinRadiusMask, avg); // Reuse avg variable for median

    //     FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    //     {
    //         if (DIRECT_MULTIDIM_ELEM(proteinRadiusMask, n) > 0)
    //         {
    //             DIRECT_MULTIDIM_ELEM(V(), n) = DIRECT_MULTIDIM_ELEM(V(), n) / std;
    //         }
    //         else
    //         {
    //             DIRECT_MULTIDIM_ELEM(V(), n) = 0;
    //         }
    //     }
    // }
    // #ifdef DEBUG_PREPROCESS
    // std::cout << "    Preprocessed map stats - Mean: " << avg << ", Std: " << std << std::endl;
    // #endif


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

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(avgVolume())
    {
        double sum  = DIRECT_MULTIDIM_ELEM(avgVolume(),n);
        double sum2 = DIRECT_MULTIDIM_ELEM(stdVolume(),n);
        double mean = sum/Ndim;

        DIRECT_MULTIDIM_ELEM(avgVolume(),n) = mean;
        DIRECT_MULTIDIM_ELEM(stdVolume(),n) = sqrt(sum2/Ndim - mean*mean);
        // DIRECT_MULTIDIM_ELEM(stdVolume(),n) = sum2/Ndim - mean*mean;
    }
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


void ProgStatisticalMap::calculateZscoreMap()
{
    std::cout << "    Calculating Zscore map..." << std::endl;

    int numberPx = Xdim*Ydim*Zdim;
    MultidimArray<double> pValuesArray;
    pValuesArray.initZeros(numberPx);

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        // Classic Z-score
        // double zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / DIRECT_MULTIDIM_ELEM(stdVolume(),n);
        double zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / DIRECT_MULTIDIM_ELEM(stdVolume(),n);

        // Average-normalized Z-score
        // double zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) * DIRECT_MULTIDIM_ELEM(avgVolume(),n) / DIRECT_MULTIDIM_ELEM(stdVolume(),n);

        // Add constant to denominator
        // double zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / sqrt(DIRECT_MULTIDIM_ELEM(stdVolume(),n) + 0.5);

        // Multiply by the average volume
        // double zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) * DIRECT_MULTIDIM_ELEM(avgDiffVolume(),n) / sqrt(DIRECT_MULTIDIM_ELEM(stdVolume(),n));

        // Correction factor by the number of maps
        // double zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / (DIRECT_MULTIDIM_ELEM(stdVolume(),n) * sqrt(1 + 1 / Ndim));

        // Ad-hoc sigma normalization factor
        // double sigma_norm = std::percentile();
        // double adjusted_std = sqrt(DIRECT_MULTIDIM_ELEM(stdVolume(),n)*DIRECT_MULTIDIM_ELEM(stdVolume(),n) + sigma_norm * sigma_norm);
        // double zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / adjusted_std;
        
        // Take only positive Z-score (densities that appear in the test map that are not present in the pool)
        // if (zscore > 0)
        // {
        //     DIRECT_MULTIDIM_ELEM(V_Zscores(),n) = zscore;

        //     if (zscore > significance_thr * equalizationParam)
        //     {
        //         DIRECT_MULTIDIM_ELEM(differentMask,n) = 1;
        //     }
        // }

                
        DIRECT_MULTIDIM_ELEM(V_Zscores(),n) = zscore;

        if (zscore > significance_thr * equalizationParam)
        {
            DIRECT_MULTIDIM_ELEM(differentMask,n) = 1;
        }

        // // Convert z-score to one-tailed p-value using standard normal CDF
        // double p = 1 - normal_cdf(zscore);
        // DIRECT_MULTIDIM_ELEM(pValuesArray,n) = p;

        // // Use p values instead of Z-scores
        // DIRECT_MULTIDIM_ELEM(V_Zscores(),n) = p;
    }

    // MultidimArray<double> pValuesArray_sort;
    // pValuesArray.sort(pValuesArray_sort);
    // double pSignificant = 1.0;
    // int q = 0.05; // significance parameter

    // // Benjamini-Hochberg procedure: find the smallest p such that p > (i/m) * q
    // FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(pValuesArray_sort)
    // {
    //     if (DIRECT_MULTIDIM_ELEM(pValuesArray_sort,n) > (n/(numberPx*1.0))*q)
    //     {
    //         pSignificant = DIRECT_MULTIDIM_ELEM(pValuesArray_sort,n);
    //         break;
    //     }
    // }

    // FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V_Zscores())
    // {
    //     if (DIRECT_MULTIDIM_ELEM(V_Zscores(),n) > pSignificant)
    //     {
    //         DIRECT_MULTIDIM_ELEM(V_Zscores(),n) = 0;
    //     }
    // }
    
    // calculate t-statistc
    // FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    // {
    //     double tStat = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / sqrt(DIRECT_MULTIDIM_ELEM(stdVolume(),n)/Ndim);
    //     double pValue = t_p_value(tStat, Ndim-1);

    //     // Invert p-value scale (higher more significant)
    //     DIRECT_MULTIDIM_ELEM(V_Zscores(),n) = 1/pValue;
    //     // if (pValue < 0.05)
    //     // {
    //     //     DIRECT_MULTIDIM_ELEM(V_Zscores(),n) = pValue;
    //     // }
    // }

    MultidimArray<double> differentMask_double;
    differentMask_double.initZeros(Zdim, Ydim, Xdim);

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(differentMask)
    {
        DIRECT_MULTIDIM_ELEM(differentMask_double, n) = 1.0 * DIRECT_MULTIDIM_ELEM(differentMask, n);
    }

    removeSmallComponents(differentMask_double, 5);

    // MultidimArray<double> differentMask_double_tmp = differentMask_double;
    // int neig = 6;   // Neighbourhood
    // int count = 0;  // Min number of empty elements in neighbourhood
    // int size = 1;   // Number of iterations or erosion 
    // closing3D(differentMask_double_tmp, differentMask_double, neig, count, size);

    double epsilon = 1e-5;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(differentMask_double)
    {
        if (DIRECT_MULTIDIM_ELEM(differentMask_double, n) > epsilon)
        {
            DIRECT_MULTIDIM_ELEM(differentMask, n) = 1;
        }
        else
        {
            DIRECT_MULTIDIM_ELEM(differentMask, n) = 0;
        }
    }
}

void ProgStatisticalMap::calculateZscoreMADMap()
{
    std::cout << "    Calculating Zscore MAD map..." << std::endl;

    double mapMAD;
    double foo;
    // computeSigmaNormMAD(mapMAD);
    MADMap().computeAvgStdev_within_binary_mask(proteinRadiusMask, mapMAD, foo);
    mapMAD = mapMAD * 1.4826; // Scale MAD to estimate sigma under normality

    std::cout << "    Global MAD value: " << mapMAD << std::endl;

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        double mad_local = DIRECT_MULTIDIM_ELEM(MADMap(), n);
        mad_local = mad_local * 1.4826; // Scale MAD to estimate sigma under normality
        double median_local  = DIRECT_MULTIDIM_ELEM(medianMap(), n);
        double val = DIRECT_MULTIDIM_ELEM(V(), n);

        double zscoreMAD = 0.0;
        double zscore = 0.0;
        // if (val > 0.0)
        // {
            zscoreMAD = (val - median_local) / sqrt(mapMAD*mapMAD + mad_local * mad_local);
            // zscoreMAD = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / sqrt(mapMAD*mapMAD + mad_local * mad_local);
            zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / sqrt(mapMAD*mapMAD + DIRECT_MULTIDIM_ELEM(stdVolume(),n)*DIRECT_MULTIDIM_ELEM(stdVolume(),n));
            // zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / sqrt(mad_local*mad_local + DIRECT_MULTIDIM_ELEM(stdVolume(),n)*DIRECT_MULTIDIM_ELEM(stdVolume(),n));
            // zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / sqrt(DIRECT_MULTIDIM_ELEM(stdVolume(),n)*DIRECT_MULTIDIM_ELEM(stdVolume(),n));
        // }

        DIRECT_MULTIDIM_ELEM(V_ZscoresMAD(), n) = zscoreMAD;
        DIRECT_MULTIDIM_ELEM(V_Zscores(), n) = zscore;
    }

    std::cout << "    Zscore MAD map calculated successfully!" << std::endl;
}

void ProgStatisticalMap::calculateZscoreMap_GlobalSigma()
{
    std::cout << "    Calculating Zscore map with global sigma correction..." << std::endl;

    // Mask common region between new map and pool using "cosine average"
    size_t numCoincidentPx = 0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        if (DIRECT_MULTIDIM_ELEM(proteinRadiusMask,n) > 0)
        {
            double num = (DIRECT_MULTIDIM_ELEM(V(),n) * DIRECT_MULTIDIM_ELEM(avgVolume(), n));
            double dem = sqrt((DIRECT_MULTIDIM_ELEM(V(),n) * DIRECT_MULTIDIM_ELEM(V(), n)) + (DIRECT_MULTIDIM_ELEM(avgVolume(),n) * DIRECT_MULTIDIM_ELEM(avgVolume(), n)));
            double div = num / dem;

            // Using 2.1213 as threshold corresponds a consistent intensity of 3 standard deviations between both maps
            if (div > 2.1213)
            {
                DIRECT_MULTIDIM_ELEM(coincidentMask,n) = 1;
                numCoincidentPx++;
            }             
        }
    }
    std::cout << "    Number of coincident pixels: " << numCoincidentPx << std::endl;

    double v_avg;
    double v_std;

    V().computeAvgStdev_within_binary_mask(coincidentMask, v_avg, v_std);

    std::cout << "    Average of the coincident region: " << v_avg << std::endl;
    std::cout << "    Std of the coincident region: " << v_std << std::endl;

    // El que ponga segundo es que el que se aplica!!!!!!
    computeSigmaNormIQR(v_std);
    computeSigmaNormMAD(v_std);

    size_t numDifferentPx = 0;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        if (DIRECT_MULTIDIM_ELEM(proteinRadiusMask,n) > 0)
        {
            // Classic Z-score
            // double zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / DIRECT_MULTIDIM_ELEM(stdVolume(),n);

            // Z-score with sigma norm correction
            // double zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / sqrt(v_std*v_std + DIRECT_MULTIDIM_ELEM(stdVolume(),n)*DIRECT_MULTIDIM_ELEM(stdVolume(),n));

            // Z-scores based on local MAD
            double zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / sqrt(v_std*v_std + DIRECT_MULTIDIM_ELEM(stdVolume(),n)*DIRECT_MULTIDIM_ELEM(stdVolume(),n));


            DIRECT_MULTIDIM_ELEM(V_Zscores(),n) = zscore;

            if (zscore > significance_thr)
            {
                DIRECT_MULTIDIM_ELEM(differentMask,n) = 1;
                numDifferentPx++;
            }
        }
    }
    std::cout << "    Number of different pixels: " << numDifferentPx << std::endl;

    // Remove small components and closing 3D in different mask
    MultidimArray<double> differentMask_double;
    differentMask_double.initZeros(Zdim, Ydim, Xdim);

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(differentMask)
    {
        DIRECT_MULTIDIM_ELEM(differentMask_double, n) = 1.0 * DIRECT_MULTIDIM_ELEM(differentMask, n);
    }

    removeSmallComponents(differentMask_double, 5);

    // MultidimArray<double> differentMask_double_tmp = differentMask_double;
    // int neig = 6;   // Neighbourhood
    // int count = 0;  // Min number of empty elements in neighbourhood
    // int size = 1;   // Number of iterations or erosion 
    // closing3D(differentMask_double_tmp, differentMask_double, neig, count, size);

    double epsilon = 1e-5;
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(differentMask_double)
    {
        if (DIRECT_MULTIDIM_ELEM(differentMask_double, n) > epsilon)
        {
            DIRECT_MULTIDIM_ELEM(differentMask, n) = 1;
        }
        else
        {
            DIRECT_MULTIDIM_ELEM(differentMask, n) = 0;
        }
    }
}

void ProgStatisticalMap::calculatePercetileMap()
{
    std::cout << "    Calculating percentile map..." << std::endl;

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V_Zscores())
    {
        if (DIRECT_MULTIDIM_ELEM(proteinRadiusMask, n) > 0)
        {
            auto pos = std::lower_bound(zScoreAccumulator.begin(), zScoreAccumulator.end(), DIRECT_MULTIDIM_ELEM(V_Zscores(), n)) - zScoreAccumulator.begin();
            double percentile = (static_cast<double>(pos) / zScoreAccumulator.size()) * 100.0;
            DIRECT_MULTIDIM_ELEM(V_Percentile(), n) = percentile;
        }
    }
}

void ProgStatisticalMap::weightMap()
{ 
    std::cout << "    Calculating weighted map..." << std::endl;

    // Filter uncoherent frequencies
    // FourierTransformer ft;
    // MultidimArray<std::complex<double>> V_ft;
	// ft.FourierTransform(V(), V_ft, false);

    // FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V_ft)
    // {
    //     if (DIRECT_MULTIDIM_ELEM(fscoh.freqMap, n) > fscoh.indexThr)
    //     {
    //         DIRECT_MULTIDIM_ELEM(V_ft,  n) = 0;
    //     }
    // }

    // ft.inverseFourierTransform();


    // Weight by z-scores
    // FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    // {
    //     DIRECT_MULTIDIM_ELEM(V(),n) *=  DIRECT_MULTIDIM_ELEM(V_Zscores(),n);
    // }


    // Use percentile to set a threshold 
    // MultidimArray<double> sorted_Zscores; 
    // V_Zscores().sort(sorted_Zscores);

    // double percentile_over3sd;
    // FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(sorted_Zscores)
    // {
    //     if (DIRECT_MULTIDIM_ELEM(sorted_Zscores,n) > 3)
    //     {
    //         percentile_over3sd = n / NZYXSIZE(sorted_Zscores);
    //         break;
    //     }
    // }

    // double overall_percentile = percentile_over3sd + equalizationParam;

    // size_t idx = static_cast<size_t>(std::floor(overall_percentile));
    // double threshold = DIRECT_MULTIDIM_ELEM(sorted_Zscores, idx);


    // Use percentile to set a threshold (bis)
    // MultidimArray<double> sorted_Zscores; 
    // V_Zscores().sort(sorted_Zscores);

    // double threshold = DIRECT_MULTIDIM_ELEM(sorted_Zscores, static_cast<size_t>(std::floor(NZYXSIZE(sorted_Zscores) - equalizationParam)));
    // std::cout << "THRESHOLD Z SCORE HALO MAP AT: " << threshold << std::endl;
    

    // Use FDR for outlier pixels
        // FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    // {
    //     DIRECT_MULTIDIM_ELEM(V(),n) =  1 - normal_cdf(DIRECT_MULTIDIM_ELEM(V_Zscores(),n));
    // }

    // Reweight z-score map based on the transofmration that put all z-score under z<1 
    // FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    // {
    //     DIRECT_MULTIDIM_ELEM(V(),n) =  DIRECT_MULTIDIM_ELEM(V_Zscores(),n) / equalizationParam;   
    // }

    // // Mask common region between new map and pool using "cosine average"
    // FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    // {
    //     if (DIRECT_MULTIDIM_ELEM(proteinRadiusMask,n) > 0)
    //     {
    //         double num = (DIRECT_MULTIDIM_ELEM(V(),n) * DIRECT_MULTIDIM_ELEM(avgVolume(), n));
    //         double dem = sqrt((DIRECT_MULTIDIM_ELEM(V(),n) * DIRECT_MULTIDIM_ELEM(V(), n)) + (DIRECT_MULTIDIM_ELEM(avgVolume(),n) * DIRECT_MULTIDIM_ELEM(avgVolume(), n)));
    //         double div = num / dem;

    //         // Only for debug lets see how is the map from which the mask is extracted
    //         // DIRECT_MULTIDIM_ELEM(V(),n) = div;

    //         if (div > significance_thr)
    //         {
    //             DIRECT_MULTIDIM_ELEM(coincidentMask,n) = 1;
    //         }            
    //     }
    // }
    
    // Compare intesities between coincident and different regions
    double coincident_avg;
    double different_avg;
    double unused_std;

    V().computeAvgStdev_within_binary_mask(coincidentMask, coincident_avg, unused_std);
    V().computeAvgStdev_within_binary_mask(differentMask, different_avg, unused_std);

    partialOccupancyFactor = different_avg / coincident_avg;

    std::cout << "coincident_avg ---------------------> " << coincident_avg << std::endl;
    std::cout << "different_avg ---------------------> " << different_avg << std::endl;
    std::cout << "partialOccupancyFactor ---------------------> " << partialOccupancyFactor << std::endl;

    // Dumpen density in different mask
    // FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    // {
    //     if (DIRECT_MULTIDIM_ELEM(differentMask,n) > 0)
    //     {
    //         DIRECT_MULTIDIM_ELEM(V(),n) *=  partialOccupancyFactor;
    //     }        
    // }

    // Subtract weighted average map 
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        DIRECT_MULTIDIM_ELEM(V(),n) =  DIRECT_MULTIDIM_ELEM(V(),n) - (DIRECT_MULTIDIM_ELEM(avgVolume(),n) * (1-partialOccupancyFactor));
    }
}

double ProgStatisticalMap::t_cdf(double t, int nu) {
    // Adapted from: ACM Algorithm 395 (Hill, 1962)
    // Two-tailed probability
    double a = t / std::sqrt(nu);
    double b = 1.0 + (a * a);
    double y = std::pow(b, -0.5 * (nu + 1));
    
    double sum = 0.0;
    if (nu % 2 == 0) {
        for (int i = 1; i <= nu / 2 - 1; ++i)
            sum += std::tgamma(nu / 2.0) / (std::tgamma(i + 1.0) * std::tgamma(nu / 2.0 - i)) * std::pow(a * a / b, i);
        return 0.5 + a * y * sum;
    } else {
        return 0.5 + std::asin(a / std::sqrt(b)) / M_PI;
    }
}

// Returns two-sided p-value
double ProgStatisticalMap::t_p_value(double t_stat, int nu) {
    double cdf = t_cdf(t_stat, nu);
    return 2 * std::min(cdf, 1.0 - cdf);
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

    if (protein_radius > 0) // Only if mas radius is provided
        createRadiusMask();
}

void ProgStatisticalMap::createRadiusMask()
{
    double radiusInPx = protein_radius / sampling_rate;
    proteinRadiusMask.initZeros(Zdim, Ydim, Xdim);

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

    #ifdef DEBUG_OUTPUT_FILES
    Image<int> saveImage;
    std::string debugFileFn = fn_oroot + "proteinRadiusMask.mrc";
    saveImage() = proteinRadiusMask;
    saveImage.write(debugFileFn);
    #endif   
} 


double ProgStatisticalMap::median(std::vector<double> v) {
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


double ProgStatisticalMap::normal_cdf(double z) {
    return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

double ProgStatisticalMap::percentile(MultidimArray<double>& data, double p)
{
    // MultidimArray<double> data_sorted;
    // data.sort(data_sorted);

    // std::cout << "------------------------" << std::endl;
    // std::cout << "NZYXSIZE(data_sorted)   "  << NZYXSIZE(data_sorted) << std::endl;
    // std::cout << "NZYXSIZE(data)   "  << NZYXSIZE(data) << std::endl;

    // double pos = (p / 100.0) * (NZYXSIZE(data_sorted) - 1);
    // size_t idx = static_cast<size_t>(std::floor(pos));
    // double frac = pos - idx;

    // std::cout << "idx   "  <<  idx << std::endl;
    // std::cout << "pos   "  <<  pos << std::endl;
    // std::cout << "frac   "  << frac << std::endl;

    // // Linear interpolation
    // // double percentile = DIRECT_MULTIDIM_ELEM(data_sorted, idx) * (1.0 - frac) + DIRECT_MULTIDIM_ELEM(data_sorted, idx+1) * frac;
    // double percentile = DIRECT_MULTIDIM_ELEM(data_sorted, NZYXSIZE(data_sorted)-1);

    // #ifdef DEBUG_PERCENTILE
    // std::cout << "Calulated percentile: " << percentile << std::endl;
    // #endif

    // std::cout << "------------------------" << std::endl;

    // return percentile;

    MultidimArray<double> data_sorted;
    data.sort(data_sorted);

    std::cout << "------------------------" << std::endl;
    std::cout << "NZYXSIZE(data_sorted)   "  << NZYXSIZE(data_sorted) << std::endl;
    std::cout << "NZYXSIZE(data)   "  << NZYXSIZE(data) << std::endl;

    double p_over3 = 0;
    double p_over0 = 0;

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(data_sorted)
    {
        if (DIRECT_MULTIDIM_ELEM(data_sorted, n) > 0)
        {
            p_over0++;

            if (DIRECT_MULTIDIM_ELEM(data_sorted, n) > 3)
            {
                p_over3++;
            }        
        }
    }

    double calculated_percentile = (p_over3/p_over0) * NZYXSIZE(data_sorted);

    std::cout << "p_over0 " << p_over0 << std::endl;
    std::cout << "p_over3 " << p_over3 << std::endl;

    #ifdef DEBUG_PERCENTILE
    std::cout << "Calulated percentile: " << calculated_percentile << std::endl;
    #endif

    std::cout << "------------------------" << std::endl;

    return calculated_percentile;
}