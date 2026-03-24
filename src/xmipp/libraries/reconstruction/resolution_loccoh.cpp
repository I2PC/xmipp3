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
    addUsageLine("This algorithm calculate the Fourier Shell Coherence from a input map pool.");

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

	// Calculate LocCoh threshold
	calculateLocCohThreshold();

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
		normalizeMap(V());

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

            dimInitialized = true;
        }

        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(V())
        {
            DIRECT_MULTIDIM_ELEM(sum_map,  n) += DIRECT_MULTIDIM_ELEM(V,n);
            DIRECT_MULTIDIM_ELEM(sum_map2, n) += DIRECT_MULTIDIM_ELEM(V,n) * DIRECT_MULTIDIM_ELEM(V,n);
        }
    }

	std::cout << "  Calculating LocCohMap map... " << std::endl;

    // Local coherence per voxel
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(LocCohMap)
	{
        DIRECT_MULTIDIM_ELEM(LocCohMap, n) = DIRECT_MULTIDIM_ELEM(sum_map,n) * DIRECT_MULTIDIM_ELEM(sum_map,n) / (Ndim * DIRECT_MULTIDIM_ELEM(sum_map2,n));
	}

    #ifdef DEBUG_OUTPUT_FILES
	Image<double> saveImage;
    std::string debugFileFn = fn_oroot + "LocCoh.mrc";

	saveImage() = LocCohMap;
	saveImage.write(debugFileFn);
    #endif

	std::cout << "  Local coherence saved at: " << LocCohMap << std::endl;
}

void ProgLocCoh::calculateLocCohThreshold()
{
    // Calculate lcoal coherence threhold
	double LocCoh_thr = (Ndim + 3.0)/(4.0 * Ndim);
    
    std::cout << "  Local coherence thresholded at: " << LocCoh_thr << std::endl;
}

// Utils methods ===================================================================
void ProgLocCoh::normalizeMap(MultidimArray<double> &vol)
{
    // Compute avg and std
    double avg;
    double std;
    V().computeAvgStdev(avg, std);

    // Normalize map
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(vol)
    {
        DIRECT_MULTIDIM_ELEM(vol, n) = (DIRECT_MULTIDIM_ELEM(vol, n) - avg) / std;
    }
}
