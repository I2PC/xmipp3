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
 #include <iostream>
 #include <string>
 #include <chrono>



// I/O methods ===================================================================
void ProgStatisticalMap::readParams()
{
    fn_mapPool = getParam("-i");
    fn_mapPool_statistical = getParam("--input_mapPool");
    fn_oroot = getParam("--oroot");
    sampling_rate = getDoubleParam("--sampling_rate");
}

void ProgStatisticalMap::show() const
{
    if (!verbose)
        return;
	std::cout
	<< "Input metadata with map pool for analysis:\t" << fn_mapPool << std::endl
	<< "Input metadata with map pool for statistical map calculation:\t" << fn_mapPool_statistical << std::endl
	<< "Output location for statistical volumes:\t" << fn_oroot << std::endl;
}

void ProgStatisticalMap::defineParams()
{
	//Usage
    addUsageLine("This algorithm computes a statistical map that characterize the input map pool for posterior comparison \
                  to new map pool to characterize the likelyness of its densities.");

    //Parameters
    addParamsLine("-i <i=\"\">                              : Input metadata containing volumes to analyze against the calculated statical map.");
    addParamsLine("--input_mapPool <input_mapPool=\"\">     : Input metadata containing map pool for statistical map calculation.");
    addParamsLine("--oroot <oroot=\"\">                     : Location for saving output.");
    addParamsLine("--sampling_rate <sampling_rate=1.0>      : Sampling rate of the input of maps.");
}

void ProgStatisticalMap::writeStatisticalMap() 
{
    avgVolume.write(fn_out_avg_map);
    stdVolume.write(fn_out_std_map);
    #ifdef DEBUG_WRITE_OUTPUT
    std::cout << "Statistical map saved at: " << fn_out_avg_map << " and " << fn_out_std_map<<std::endl;
    #endif
}

void ProgStatisticalMap::writeZscoresMap(FileName fnIn) 
{
    // Compose filename
    size_t lastSlashPos = fnIn.find_last_of("/\\");
    size_t lastDotPos = fnIn.find_last_of('.');

    FileName newFileName = fnIn.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1) + "_Zscores.mrc";
    FileName fnOut = fn_oroot + (fn_oroot.back() == '/' || fn_oroot.back() == '\\' ? "" : "/") + newFileName;

    // Check if file already existes (the same pool map might contain to identical filenames
    int counter = 1;
    while (std::ifstream(fnOut)) 
    {
        fnOut = fn_oroot + (fn_oroot.back() == '/' || fn_oroot.back() == '\\' ? "" : "/") + fnIn.substr(fnIn.find_last_of("/\\") + 1, fnIn.find_last_of('.') - fnIn.find_last_of("/\\") - 1) + "_Zscores_" + std::to_string(counter++) + fnIn.substr(fnIn.find_last_of('.'));
    }

    //Write output weighted volume
    V_Zscores.write(fnOut);
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


// Main method ===================================================================
void ProgStatisticalMap::run()
{
	auto t1 = std::chrono::high_resolution_clock::now();

    generateSideInfo();

    calculateFSCoh();

    // Calculate statistical map
    mapPoolMD.read(fn_mapPool_statistical);
    Ndim = mapPoolMD.size();

    for (const auto& row : mapPoolMD)
	{
        row.getValue(MDL_IMAGE, fn_V);

        #ifdef DEBUG_STAT_MAP
        std::cout << "Processing volume " << fn_V << " from statistical map pool..." << std::endl;
        #endif

        V.read(fn_V); 

        if (!dimInitialized)
        {
            // Read dim
            Xdim = XSIZE(V());
            Ydim = YSIZE(V());
            Zdim = ZSIZE(V());

            #ifdef DEBUG_DIM
            std::cout 
            << "Xdim: " << Xdim << std::endl
            << "Ydim: " << Ydim << std::endl
            << "Zdim: " << Zdim << std::endl
            << "Ndim: " << Ndim << std::endl;
            #endif

            avgVolume().initZeros(Zdim, Ydim, Xdim);
            stdVolume().initZeros(Zdim, Ydim, Xdim);
            avgDiffVolume().initZeros(Zdim, Ydim, Xdim);
            V_Zscores().initZeros(Zdim, Ydim, Xdim);

            dimInitialized = true;
        }

        preprocessMap();
        processStaticalMap();
    }

    computeStatisticalMaps();
    calculateAvgDiffMap();

    #ifdef DEBUG_STAT_MAP
    std::cout << "Statistical map succesfully calculated!" << std::endl;
    #endif
    
    writeStatisticalMap();

    // Compare input maps against statistical map
    mapPoolMD.read(fn_mapPool);

    for (const auto& row : mapPoolMD)
	{
        row.getValue(MDL_IMAGE, fn_V);

        #ifdef DEBUG_WEIGHT_MAP
        std::cout << "Anayzing volume " << fn_V << " against statistical map..." << std::endl;
        #endif

        V.read(fn_V);

        preprocessMap();

        calculateZscoreMap();
        writeZscoresMap(fn_V);

        weightMap();
        writeWeightedMap(fn_V);
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
	std::cout << "----- Calculate FOCoh" << std::endl;
	#endif

	fscoh.run();

	#ifdef VERBOSE_OUTPUT
	std::cout << "----- FOCoh caluclated successfully!" << std::endl;
	#endif
}

void ProgStatisticalMap::preprocessMap()
{
    std::cout << "    Preprocessing input map..." << std::endl;

    // Normalize map: mean=0, std=1
    double avg;
    double std;
    V().computeAvgStdev(avg, std);

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        DIRECT_MULTIDIM_ELEM(V(), n) = (DIRECT_MULTIDIM_ELEM(V(), n) - avg) / std;
    }

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

void ProgStatisticalMap::calculateZscoreMap()
{
    std::cout << "    Calculating Zscore map..." << std::endl;

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        // Classic Z-score
        double zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / DIRECT_MULTIDIM_ELEM(stdVolume(),n);

        // Average-normalized Z-score
        // double zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) * DIRECT_MULTIDIM_ELEM(avgVolume(),n) / DIRECT_MULTIDIM_ELEM(stdVolume(),n);

        // Add constant to denominator
        // double zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / sqrt(DIRECT_MULTIDIM_ELEM(stdVolume(),n) + 0.5);

        // Multiply by the 
        // double zscore  = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) * DIRECT_MULTIDIM_ELEM(avgDiffVolume(),n) / sqrt(DIRECT_MULTIDIM_ELEM(stdVolume(),n));

        // Take only positive Z-score (densities that appear in the test map that are not present in the pool)
        if (zscore > 0)
        {
            DIRECT_MULTIDIM_ELEM(V_Zscores(),n) = zscore;
        }

        // DIRECT_MULTIDIM_ELEM(V_Zscores(),n) = zscore;
    }
    
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
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        DIRECT_MULTIDIM_ELEM(V(),n) *= DIRECT_MULTIDIM_ELEM(V_Zscores(),n);
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
    fn_out_avg_map = fn_oroot + "statsMap_avg.mrc";
    fn_out_std_map = fn_oroot + "statsMap_std.mrc";
}
