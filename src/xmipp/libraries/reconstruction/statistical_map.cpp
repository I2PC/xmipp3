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
 #include <cmath>
#include <numeric>



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
    #ifdef VERBOSE_OUTPUT
    std::cout << "\n\n---Analyzing input map pool for statistical characterization---" << std::endl;
    #endif
    
    mapPoolMD.read(fn_mapPool_statistical);
    Ndim = mapPoolMD.size();

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

            // For Dixon
            // stdVolume().initConstant(DBL_MAX);

            dimInitialized = true;
        }

        preprocessMap(fn_V);
        // processStaticalMapDixon();
        processStaticalMap();
    }

    computeStatisticalMaps();
    // calculateAvgDiffMap();

    #ifdef DEBUG_STAT_MAP
    std::cout << "Statistical map succesfully calculated!" << std::endl;
    #endif
    
    writeStatisticalMap();

    // Calculate Z-score maps from statistical map pool for histogram equalization
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
        calculateZscoreMap();
        writeZscoresMap(fn_V);

        // double p = percentile(V_Zscores(), percentileThr);
        // histogramEqualizationParameters.push_back(p);        

        double min;
        double max;
        V_Zscores().computeDoubleMinMax(min, max);

        #ifdef DEBUG_PERCENTILE
        std::cout << "Max value in Z-score map: " << max << std::endl;
        #endif

        histogramEqualizationParameters.push_back(max);        
    }

    // Calculate average transformation
    double sum = std::accumulate(histogramEqualizationParameters.begin(), histogramEqualizationParameters.end(), 0.0);
    equalizationParam =  sum / histogramEqualizationParameters.size();

    #ifdef DEBUG_PERCENTILE
    std::cout << "Equalization parameter: " << equalizationParam << std::endl;
    #endif

    // Compare input maps against statistical map
    #ifdef VERBOSE_OUTPUT
    std::cout << "\n\n---Comparing input map pool agains statistical map---" << std::endl;
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
        calculateZscoreMap();
        // calculateDixonMap();
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

    #ifdef DEBUG_OUTPUT_FILES
    FileName fnOut = fn_oroot + (fn_oroot.back() == '/' || fn_oroot.back() == '\\' ? "" : "/") + fnIn.substr(fnIn.find_last_of("/\\") + 1, fnIn.find_last_of('.') - fnIn.find_last_of("/\\") - 1) + "_preprocess.mrc";
    V.write(fnOut);
    #endif
}

void ProgStatisticalMap::processStaticalMapDixon()
{ 
    std::cout << "    Processing input map for statistical map calculation..." << std::endl;

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        // Reuse avg and std maps for min and max (memory efficient)
        double value = DIRECT_MULTIDIM_ELEM(V(),n);

        if(value > DIRECT_MULTIDIM_ELEM(avgVolume(),n))
        {
            DIRECT_MULTIDIM_ELEM(avgVolume(),n) = value;   // max
        }
        if(value < DIRECT_MULTIDIM_ELEM(stdVolume(),n))
        {
            DIRECT_MULTIDIM_ELEM(stdVolume(),n) = value;   // min
        }
    }
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

void ProgStatisticalMap::calculateDixonMap()
{
    std::cout << "    Calculating Dixon map..." << std::endl;
    double dixonThreshold = 0.521;

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        // Compute Dixon
        if (DIRECT_MULTIDIM_ELEM(V(),n) > DIRECT_MULTIDIM_ELEM(avgVolume(),n))
        {
            DIRECT_MULTIDIM_ELEM(V_Zscores(),n) = (DIRECT_MULTIDIM_ELEM(V(),n) - DIRECT_MULTIDIM_ELEM(avgVolume(),n)) / (DIRECT_MULTIDIM_ELEM(avgVolume(),n) - DIRECT_MULTIDIM_ELEM(stdVolume(),n));

            if (DIRECT_MULTIDIM_ELEM(V_Zscores(),n) > dixonThreshold)
            {
               DIRECT_MULTIDIM_ELEM(V(),n) = 1;
            }
            
        }
        else
        {
            DIRECT_MULTIDIM_ELEM(V_Zscores(),n) = 0;
            DIRECT_MULTIDIM_ELEM(V(),n) = 0;
        }
    }
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
        if (zscore > 0)
        {
            DIRECT_MULTIDIM_ELEM(V_Zscores(),n) = zscore;
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
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        DIRECT_MULTIDIM_ELEM(V(),n) =  DIRECT_MULTIDIM_ELEM(V_Zscores(),n) / equalizationParam;
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