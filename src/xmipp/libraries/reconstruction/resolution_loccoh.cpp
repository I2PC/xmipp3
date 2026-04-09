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

 #include "resolution_loccoh.h"
 #include "core/metadata_extension.h"
 #include "core/multidim_array.h"
 #include "core/xmipp_image_base.h"
 #include <iostream>
 #include <string>
 #include <chrono>



// I/O methods ===================================================================
void ProgLocCoh::readParams()
{
    fn_mapPool = getParam("-i");
    fn_oroot = getParam("--oroot");

	if (!fn_oroot.empty() && fn_oroot.back() != '/')
		fn_oroot += '/';
}

void ProgLocCoh::defineParams()
{
	//Usage
    addUsageLine("This algorithm calculate the Local Coherence from a input map pool.");

    //Parameters
    addParamsLine("-i <i=\"\">              : Input metadata containing the map pool.");
    addParamsLine("--oroot <oroot=\"\">     : Location for saving output.");
}

void ProgLocCoh::show() const
{
    if (!verbose)
        return;
	std::cout
	<< "Input metadata containing map pool:\t" << fn_mapPool << std::endl
	<< "Output location for LocCoh:\t"         << fn_oroot   << std::endl;
}


// Main method ===================================================================
void ProgLocCoh::run()
{
	auto t1 = std::chrono::high_resolution_clock::now();

    mapPoolMD.read(fn_mapPool);
    Ndim = mapPoolMD.size();

    // Calculate statistical map
    localCoherence(mapPoolMD);

    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

 	std::cout << "Execution time: " << ms_int.count() << " ms" << std::endl;
}


// Core methods ===================================================================
void ProgLocCoh::localCoherence(MetaDataVec mapPoolMD)
{
    std::cout << "Calculating Fourier Shell Coherence..." << std::endl;

    MultidimArray<std::complex<double>> V_ft;

    for (const auto& row : mapPoolMD)
	{
        row.getValue(MDL_IMAGE, fn_V);

        #ifdef VERBOSE_OUTPUT
        std::cout << "  Processing volume " << fn_V << " for LocCoh calculation" << std::endl;
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

            LocCohMap.initZeros(Zdim, Ydim, Xdim);
            sum_map.initZeros(Zdim, Ydim, Xdim);
            sum_map2.initZeros(Zdim, Ydim, Xdim);

            dimInitialized = true;
        }

        normalizeMap();

        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
        {
            double value = DIRECT_MULTIDIM_ELEM(V, n);

            DIRECT_MULTIDIM_ELEM(sum_map,  n) += value;
            DIRECT_MULTIDIM_ELEM(sum_map2, n) += value * value;           
        }
    }

	std::cout << "  Calculating LocCohMap map... " << std::endl;

    // Local coherence per voxel
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(LocCohMap)
	{
        if(DIRECT_MULTIDIM_ELEM(sum_map2,n) > 0)
        {
            DIRECT_MULTIDIM_ELEM(LocCohMap, n) = (DIRECT_MULTIDIM_ELEM(sum_map,n) * DIRECT_MULTIDIM_ELEM(sum_map,n)) / (Ndim * DIRECT_MULTIDIM_ELEM(sum_map2,n));
        }
	}

    // // Local coherence per voxel with local neighborhood (3x3)
    // int winSize = 3;
    // int halfWin = winSize / 2;

    // for (int k = halfWin; k < Zdim - halfWin; k++)
	// {
	//     for (int i = halfWin; i < Ydim - halfWin; i++)
	// 	{
	// 		for (int j = halfWin; j < Xdim - halfWin; j++)
	// 		{	
    //             // Iterate over local neighborhood
    //             double localSum = 0.0;
    //             double localSum2 = 0.0;
    //             double localCoherence = 0.0;

    //             for (int z = -halfWin; z <= halfWin; z++)                
    //             {
    //                 for (int y = -halfWin; y <= halfWin; y++)                    
    //                 {
    //                     for (int x = -halfWin; x <= halfWin; x++)
    //                     {
    //                         localSum  += DIRECT_A3D_ELEM(sum_map, k+z, i+y, j+x);
    //                         localSum2 += DIRECT_A3D_ELEM(sum_map2, k+z, i+y, j+x);
    //                         // localCoherence += (DIRECT_A3D_ELEM(sum_map, k+z, i+y, j+x) * DIRECT_A3D_ELEM(sum_map, k+z, i+y, j+x)) / (Ndim * DIRECT_A3D_ELEM(sum_map2, k+z, i+y, j+x));
    //                     }
    //                 }
    //             }

    //             if (localSum2 > 0.00001)
    //             {
    //                 DIRECT_A3D_ELEM(LocCohMap, k, i, j) = (localSum * localSum) / (winSize * winSize * winSize * localSum2 * Ndim);
    //             }
                
    //             DIRECT_A3D_ELEM(LocCohMap, k, i, j) = (localSum * localSum) / (winSize * winSize * winSize * localSum2 * Ndim);                // DIRECT_A3D_ELEM(LocCohMap, k, i, j) = localCoherence / (winSize * winSize * winSize);
	// 		}
	// 	}
	// } 

    Image<double> saveImage;

    // Save local coherence map
    saveImage() = LocCohMap;
    saveImage.write(fn_oroot + "LocCoh.mrc");

    #ifdef DEBUG_OUTPUT_FILES
    // Save sum_map
    saveImage() = sum_map;
    saveImage.write(fn_oroot + "sum_map.mrc");

    // Save sum_map2
    saveImage() = sum_map2;
    saveImage.write(fn_oroot + "sum_map2.mrc");
    #endif

	std::cout << "  Local coherence saved at: " << fn_oroot << "LocCoh.mrc" << std::endl;
}

// Utils methods ===================================================================
void ProgLocCoh::normalizeMap()
{
    double avg;
    double std;
    V().computeAvgStdev(avg, std);

    MultidimArray<int> positiveMask;
    positiveMask.initZeros(Zdim, Ydim, Xdim);

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        if (DIRECT_MULTIDIM_ELEM(V(), n) > 1.64 * std)
        {
            DIRECT_MULTIDIM_ELEM(positiveMask, n) = 1;
        }
    }

    // // Normalize map on positive densities dividing by std
    // double foo;
    // V().computeAvgStdev_within_binary_mask(positiveMask, foo, std);
    // std::cout << "    Normalizing map by stdev value of positive densities: " << std << std::endl;

    // FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    // {
    //     DIRECT_MULTIDIM_ELEM(V(), n) = (DIRECT_MULTIDIM_ELEM(V(), n)) / std;
    // }

    // Normalize map on positive densities dividing by bouding in (min, max)
    double min = MAXDOUBLE;
    double max = 0;

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        if (DIRECT_MULTIDIM_ELEM(positiveMask, n) > 0)
        {
            double val = DIRECT_MULTIDIM_ELEM(V(), n);
            
            if (val < min)
            {
                min = val;
            }
            if (val > max)
            {
                max = val;
            }
        }
    }

    std::cout << "    Normalizing map by min and max value of positive densities: " << min << " - " << max << std::endl;

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
    {
        double val = (DIRECT_MULTIDIM_ELEM(V(), n) - min) / (max - min);

        if (val < 0)
        {
            DIRECT_MULTIDIM_ELEM(V(), n) = 0;
        }
        else
        {
            DIRECT_MULTIDIM_ELEM(V(), n) = val;
        }
    }

    #ifdef DEBUG_OUTPUT_FILES
    // Build base name
    const size_t s = fn_V.find_last_of("/\\");
    const size_t d = fn_V.find_last_of('.');
    const size_t start = (s == FileName::npos ? 0 : s + 1);
    const size_t end = (d == FileName::npos || d <= s ? fn_V.size() : d);
    const FileName base = fn_V.substr(start, end - start);

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
    std::cout << "  Normalized map saved at: " << fnOut << std::endl;
    #endif
}
