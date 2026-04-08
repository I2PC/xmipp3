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

 #include "resolution_fscoh.h"
 #include "core/metadata_extension.h"
 #include "core/multidim_array.h"
 #include "core/xmipp_image_base.h"
 #include <iostream>
 #include <string>
 #include <chrono>



// I/O methods ===================================================================
void ProgClassifyMapCluster::readParams()
{
    fn_mapPool = getParam("-i");
    fn_oroot = getParam("--oroot");
    sampling_rate = getDoubleParam("--sampling_rate");

	if (!fn_oroot.empty() && fn_oroot.back() != '/')
		fn_oroot += '/';
}

void ProgClassifyMapCluster::defineParams()
{
	//Usage
    addUsageLine("This algorithm calculate the Fourier Shell Coherence from a input map pool.");

    //Parameters
    addParamsLine("-i <i=\"\">                              : Input metadata containing the map pool.");
    addParamsLine("--oroot <oroot=\"\">                     : Location for saving output.");
    addParamsLine("--sampling_rate <sampling_rate=1.0>      : Sampling rate of the input of maps.");
}

void ProgClassifyMapCluster::show() const
{
    if (!verbose)
        return;
	std::cout
	<< "Input metadata containing map pool:\t" << fn_mapPool << std::endl
	<< "Output location for FSCoh:\t" << fn_oroot << std::endl;
}


// Main method ===================================================================
void ProgClassifyMapCluster::run()
{
	auto t1 = std::chrono::high_resolution_clock::now();

    mapPoolMD.read(fn_mapPool);
    Ndim = mapPoolMD.size();

	for (const auto& row : mapPoolMD)
	{
        row.getValue(MDL_IMAGE, fn_V);

        #ifdef DEBUG_STAT_MAP
        std::cout << "Processing volume " << fn_V << " from statistical map pool..." << std::endl;
        #endif

        V.clear();
		V_tf.clean();  // ** ojo que esto no se este cargando el vecto y solo lo ponga a ceros

        V.read(fn_V);
		ft.FourierTransform(V(), V_ft, false);
		normalizeFTMap(V_ft);

        if (!dimInitialized)
        {
            // Generate side info
            generateSideInfo();

            #ifdef DEBUG_DIM
            std::cout 
            << "Xdim: " << Xdim << std::endl
            << "Ydim: " << Ydim << std::endl
            << "Zdim: " << Zdim << std::endl
			<< "Xdim_ft: " << Xdim_ft << std::endl
            << "Ydim_ft: " << Ydim_ft << std::endl
            << "Zdim_ft: " << Zdim_ft << std::endl
            << "Ndim: " << Ndim << std::endl;
            #endif

			referenceMapPool_ft().initZeros(Ndim, Zdim_ft, Ydim_ft, Xdim_ft);
			distanceMatrix.initZeros(Ndim, Ndim);

            dimInitialized = true;
        }

        // Load preprocess map in reference map pool
        for(size_t k = 0; k < Zdim_ft; k++)
	    {
            for(size_t j = 0; j <Xdim_ft; j++)
            {
                for(size_t i = 0; i < Ydim_ft; i++)
                {
                    DIRECT_NZYX_ELEM(referenceMapPool_ft(), volCounter, k, i, j) = DIRECT_ZYX_ELEM(V_ft(), k, i, j);
                }
            }
        }

        volCounter++;
    }

    // Calculate pairwise FSC-based distances
	double distance = 0;

	for(size_t i1 = 0; i1 < Ndim; i1++)
	{
		for(size_t i2 = 0; i2 < Ndim; i2++)
		{
			if (i1==i2)
			{
				DIRECT_YX_ELEM(distanceMatrix, i1, i2) = 1;
				DIRECT_YX_ELEM(distanceMatrix, i2, i1) = 1;
			}
			else
			{
				calculateDistanceFSC(distance, i1, i2);
				DIRECT_YX_ELEM(distanceMatrix, i1, i2) = distance;
				DIRECT_YX_ELEM(distanceMatrix, i2, i1) = distance;
			}
		}
	}

	// Cluster maps
	// ***TODO!!!!!!!!!!!

    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

 	std::cout << "Execution time: " << ms_int.count() << " ms" << std::endl;
}


// Core methods ===================================================================
void ProgClassifyMapCluster::calculateDistanceFSC(double distance, int i1, int i2)
{
    std::cout << "Calculating FSC distance for indexes " << i1 << " and " << i2 << "..." << std::endl;

	for(size_t k = 0; k < Zdim_ft; k++)
	{
		for(size_t j = 0; j <Xdim_ft; j++)
		{
			for(size_t i = 0; i < Ydim_ft; i++)
			{
				DIRECT_NZYX_ELEM(referenceMapPool_ft(), volCounter, k, i, j) = DIRECT_ZYX_ELEM(V_ft(), k, i, j);

				int freqIdx = (int)DIRECT_ZYX_ELEM(freqMap, k, i, j);

				// Consider only up to Nyquist (remove corners from analysis)
				if (freqIdx < NZYXSIZE(FSC))
				{
					std::complex<double> i1_value = DIRECT_NZYX_ELEM(referenceMapPool_ft(), i1, k, i, j);
					std::complex<double> i2_value = DIRECT_NZYX_ELEM(referenceMapPool_ft(), i2, k, i, j);
					DIRECT_MULTIDIM_ELEM(FSC_num, freqIdx) += std::abs(i1_value * i2_value);
					DIRECT_MULTIDIM_ELEM(FSC_den, freqIdx) += sqrt(std::norm(i1_value) + std::norm(i2_value));
				}
			}
		}
	}

	distance = 0;

	MetaDataVec md;
	size_t id;

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FSC)
	{
        double value;
		value = DIRECT_MULTIDIM_ELEM(FSC_num, n) / DIRECT_MULTIDIM_ELEM(FSC_den,n);
		
		// Because it is a distance the higher the FSC the lower de distance (1-value)
		distance += 1 - value;

		// Save FSC for debugging
		DIRECT_MULTIDIM_ELEM(FSC, n) = value;

		id = md.addObject();

		md.setValue(MDL_RESOLUTION, value, id);
		md.setValue(MDL_RESOLUTION_FREQ, (1.0 * n / (2 * NZYXSIZE(FSC))), id);
	}

	std::string outputMD = fn_oroot + "FSC_" + str(i1) + "_vs_" + str(i2) + ".xmd";
	md.write(outputMD);

	std::cout << "  FSC saved at: " << outputMD << std::endl;
	std::cout << "  Caluculated distance for maps " << str(i1) << " and " << str(i2) << ": " << distance << std::endl;
}


// Utils methods ===================================================================
void ProgClassifyMapCluster::generateSideInfo()
{
    Xdim = XSIZE(V());
    Ydim = YSIZE(V());
    Zdim = ZSIZE(V());

    Xdim_ft = XSIZE(V_ft());
    Ydim_ft = YSIZE(V_ft());
    Zdim_ft = ZSIZE(V_ft());

    fn_out_avg_map = fn_oroot + "statsMap_avg.mrc";
    fn_out_std_map = fn_oroot + "statsMap_std.mrc";
    fn_out_median_map = fn_oroot + "statsMap_median.mrc";
    fn_out_mad_map = fn_oroot + "statsMap_mad.mrc";

    composefreqMap();
}

void ProgClassifyMapCluster::composefreqMap()
{
	// Calculate FT
	MultidimArray<std::complex<double>> V_ft; // Volume FT

	ft.FourierTransform(V(), V_ft, false);

	// FT dimensions
	int Xdim_ft = XSIZE(V_ft);
	int Ydim_ft = YSIZE(V_ft);
	int Zdim_ft = ZSIZE(V_ft);
	int Ndim_ft = NSIZE(V_ft);

    // Use this dimension to initialize mFSC auxiliary maps
    FSC.initZeros(std::min(Xdim_ft, std::min(Ydim_ft, Zdim_ft)));
    FSC_num.initZeros(std::min(Xdim_ft, std::min(Ydim_ft, Zdim_ft)));
    FSC_den.initZeros(std::min(Xdim_ft, std::min(Ydim_ft, Zdim_ft)));

	if (Zdim_ft == 1)
	{
		Zdim_ft = Ndim_ft;
	}

	int maxRadius = std::min(Xdim_ft, std::min(Ydim_ft, Zdim_ft));	// Restric analysis to Nyquist

	#ifdef DEBUG_FREQUENCY_MAP
	std::cout << "FFT map dimensions: " << std::endl;  
	std::cout << "FT xSize " << Xdim_ft << std::endl;
	std::cout << "FT ySize " << Ydim_ft << std::endl;
	std::cout << "FT zSize " << Zdim_ft << std::endl;
	std::cout << "FT nSize " << Ndim_ft << std::endl;
	std::cout << "maxRadius " << maxRadius << std::endl;
	#endif

	// Construct frequency map and initialize the frequency vectors
	Matrix1D<double> freq_fourier_x;
	Matrix1D<double> freq_fourier_y;
	Matrix1D<double> freq_fourier_z;

	freq_fourier_x.initZeros(Xdim_ft);
	freq_fourier_y.initZeros(Ydim_ft);
	freq_fourier_z.initZeros(Zdim_ft);

	double u;	// u is the frequency

	// Defining frequency components. First element should be 0, it is set as the smallest number to avoid singularities
	VEC_ELEM(freq_fourier_z,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<Zdim_ft; ++k){
		FFT_IDX2DIGFREQ(k,Zdim, u);
		VEC_ELEM(freq_fourier_z, k) = u;
	}

	VEC_ELEM(freq_fourier_y,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<Ydim_ft; ++k){
		FFT_IDX2DIGFREQ(k,Ydim, u);
		VEC_ELEM(freq_fourier_y, k) = u;
	}

	VEC_ELEM(freq_fourier_x,0) = std::numeric_limits<double>::min();
	for(size_t k=1; k<Xdim_ft; ++k){
		FFT_IDX2DIGFREQ(k,Xdim, u);
		VEC_ELEM(freq_fourier_x, k) = u;
	}

	//Initializing map with frequencies
	freqMap.resizeNoCopy(V_ft);

	// Directional frequencies along each direction
	double uz;
	double uy;
	double ux;
	double uz2;
	double uz2y2;
	long n=0;
	int idx = 0;

	for(size_t k=0; k<Zdim_ft; ++k)
	{
		uz = VEC_ELEM(freq_fourier_z, k);
		uz2 = uz*uz;
		
		for(size_t i=0; i<Ydim_ft; ++i)
		{
			uy = VEC_ELEM(freq_fourier_y, i);
			uz2y2 = uz2 + uy*uy;

			for(size_t j=0; j<Xdim_ft; ++j)
			{
				ux = VEC_ELEM(freq_fourier_x, j);
				ux = sqrt(uz2y2 + ux*ux);

				idx = (int) round(ux * Xdim);
				DIRECT_MULTIDIM_ELEM(freqMap,n) = idx;

				++n;
			}
		}
	}

    #ifdef DEBUG_OUTPUT_FILES
	Image<double> saveImage;
	std::string debugFileFn = fn_oroot + "FSCoh_freqMap.mrc";
	saveImage() = freqMap;
	saveImage.write(debugFileFn);
    #endif
}


void ProgClassifyMapCluster::normalizeFTMap(MultidimArray<std::complex<double>> &volFT)
{
    // Compute avg and std
    std::complex<double> sum;
    double sum2;
	int numElems; 

	sum = (0,0);	// also mean
	sum2 = 0;		// also std
	numElems = 0;

	// Compute sum and sum^2 
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(volFT)
	{
		int freqIdx = static_cast<int>(DIRECT_MULTIDIM_ELEM(freqMap, n));

       	if (freqIdx < NZYXSIZE(FSCoh))
		{
            sum      += DIRECT_MULTIDIM_ELEM(volFT,n);
            sum2     += std::norm(DIRECT_MULTIDIM_ELEM(volFT,n));
            numElems += 1;
        }
	}

	sum /= numElems;
	sum2 = sqrt(sum2 / static_cast<double>(numElems) - std::norm(sum));

    // Normalize map
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(volFT)
    {
        DIRECT_MULTIDIM_ELEM(volFT, n) = (DIRECT_MULTIDIM_ELEM(volFT, n) - sum) / sum2;
    }
}
