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

 #include "classify_map_cluster.h"
 #include "core/metadata_extension.h"
 #include "core/multidim_array.h"
 #include "core/xmipp_image_base.h"
 #include <iostream>
 #include <string>
 #include <chrono>
 #include <iomanip>



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
    addUsageLine("This algorithm cluster maps based on FSC distance.");

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
	<< "Output location for FSC:\t" << fn_oroot << std::endl;
}


// Main method ===================================================================
void ProgClassifyMapCluster::run()
{
	auto t1 = std::chrono::high_resolution_clock::now();

    mapPoolMD.read(fn_mapPool);
    Ndim = mapPoolMD.size();

	size_t volCounter = 0;

	for (const auto& row : mapPoolMD)
	{
        row.getValue(MDL_IMAGE, fn_V);

        #ifdef VERBOSE_OUTPUT
        std::cout << "Processing volume " << fn_V << " from statistical map pool..." << std::endl;
        #endif

        V.clear();
		V_ft.clear();

        V.read(fn_V);
		ft.FourierTransform(V(), V_ft, false);

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

		// Normalize map in Fourier space 
		normalizeFTMap(V_ft);

        // Load preprocess map in reference map pool
        for(size_t k = 0; k < Zdim_ft; k++)
	    {
            for(size_t j = 0; j <Xdim_ft; j++)
            {
                for(size_t i = 0; i < Ydim_ft; i++)
                {
                    DIRECT_NZYX_ELEM(referenceMapPool_ft(), volCounter, k, i, j) = DIRECT_ZYX_ELEM(V_ft, k, i, j);
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
				MAT_ELEM(distanceMatrix, i1, i2) = 0.0;
				MAT_ELEM(distanceMatrix, i2, i1) = 0.0;
			}
			else
			{
				calculateDistanceFSC(distance, i1, i2);
				MAT_ELEM(distanceMatrix, i1, i2) = distance;
				MAT_ELEM(distanceMatrix, i2, i1) = distance;
			}
		}
	}

	#ifdef VERBOSE_OUTPUT
	std::cout << "--Distance matrix" << std::endl;

	for(size_t i = 0; i < Ndim; i++)
	{
		for(size_t j = 0; j <Ndim; j++)
		{	
			if (j == 0) std::cout << "\n";
			std::cout << std::fixed << std::setprecision(2) << MAT_ELEM(distanceMatrix, i, j) << "\t";

		}
	}
	std::cout << std::endl;
	std::cout << std::endl;
	#endif

	// MDS of distance matrix
	Matrix2D<double> B;
	Matrix1D<double> eigenvals;
    Matrix2D<double> eigenvecs;
	
	classicalMDS(distanceMatrix, B, eigenvals, eigenvecs);

	// Build embedding
	int p = 0; // Number of dimensions to keep in the embedding

	// Use expalined variance as criteria to select number of dimensions in the embedding
	double explainedVariance_threshold = 0.9;
	double totalVariance = 0.0;

	for(size_t i = 0; i < Ndim; i++)
	{
		totalVariance += eigenvals[i];
	}

	double cumulativeVariance = 0.0;

	for(size_t i = 0; i < Ndim; i++)	
	{
		cumulativeVariance = cumulativeVariance + eigenvals[i];
		double explainedVariance = cumulativeVariance / totalVariance;

		std::cout << "Cumulative variance explained by " << (i+1) << " dimensions: " << explainedVariance << std::endl;

		if (explainedVariance < explainedVariance_threshold)
		{
			p++;
		}
		else
		{
			break;
		}
	}

	std::cout << "Number of dimensions in the embedding: " << p << std::endl;
	std::cout << std::endl;

	Matrix2D<double> embedding;
	embedding.initZeros(Ndim, p);

	for (size_t i = 0; i < Ndim; ++i)
	{
		for (size_t d = 0; d < p; ++d)
		{
			double lambda = eigenvals(d);
			if (lambda > 0)
				MAT_ELEM(embedding, i, d) = std::sqrt(lambda) * MAT_ELEM(eigenvecs, i, d);
			else
				MAT_ELEM(embedding, i, d) = 0.0;
		}
	}

	// Apply k-means to embedding
	int k = 2;
	int maxIter = 100;
	Matrix1D<int> labels;
	labels.initZeros(Ndim);

	kmeans(embedding, k, maxIter, labels);

	#ifdef DEBUG_MDS
	std::cout << "--Labelling" << std::endl;

	for(size_t i = 0; i < Ndim; i++)
	{
		std::cout << std::fixed << labels[i] << "\t";
	}
	std::cout << std::endl;
	std::cout << std::endl;
	#endif

    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

 	std::cout << "Execution time: " << ms_int.count() << " ms" << std::endl;
}

// Core methods ===================================================================
void ProgClassifyMapCluster::calculateDistanceFSC(double &distance, int i1, int i2)
{
    std::cout << "Calculating FSC distance for indexes " << i1 << " and " << i2 << "..." << std::endl;

	FSC_num.initZeros(std::min(Xdim_ft, std::min(Ydim_ft, Zdim_ft)));
    FSC_den1.initZeros(std::min(Xdim_ft, std::min(Ydim_ft, Zdim_ft)));
    FSC_den2.initZeros(std::min(Xdim_ft, std::min(Ydim_ft, Zdim_ft)));

	for(size_t k = 0; k < Zdim_ft; k++)
	{
		for(size_t j = 0; j < Xdim_ft; j++)
		{
			for(size_t i = 0; i < Ydim_ft; i++)
			{
				int freqIdx = (int)DIRECT_ZYX_ELEM(freqMap, k, i, j);

				// Consider only up to Nyquist (remove corners from analysis)
				if (freqIdx < NZYXSIZE(FSC))
				{
					std::complex<double> i1_value = DIRECT_NZYX_ELEM(referenceMapPool_ft(), i1, k, i, j);
					std::complex<double> i2_value = DIRECT_NZYX_ELEM(referenceMapPool_ft(), i2, k, i, j);
					DIRECT_MULTIDIM_ELEM(FSC_num, freqIdx) += std::abs(i1_value * i2_value);
					DIRECT_MULTIDIM_ELEM(FSC_den1, freqIdx) += std::norm(i1_value);
					DIRECT_MULTIDIM_ELEM(FSC_den2, freqIdx) += std::norm(i2_value);
				}
			}
		}
	}

	distance = 0;

	MetaDataVec md;
	size_t id;

    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(FSC)
	{
        double frc;
		frc = DIRECT_MULTIDIM_ELEM(FSC_num, n) / sqrt(DIRECT_MULTIDIM_ELEM(FSC_den1,n) * DIRECT_MULTIDIM_ELEM(FSC_den2, n));
		
		// Because it is a distance the higher the FSC the lower de distance (1-frc)
		distance += frc;

		// Save FSC for debugging
		DIRECT_MULTIDIM_ELEM(FSC, n) = frc;

		id = md.addObject();

		md.setValue(MDL_RESOLUTION_FRC, frc, id);
		md.setValue(MDL_RESOLUTION_FREQ, (1.0 * n / (2 * NZYXSIZE(FSC))), id);
	}

	distance = 1 - distance / NZYXSIZE(FSC);

	std::string outputMD = fn_oroot + "FSC_" + std::to_string(i1) + "_vs_" + std::to_string(i2) + ".xmd";
	md.write(outputMD);

	std::cout << "  FSC saved at: " << outputMD << std::endl;
	std::cout << "  Caluculated distance for maps " << std::to_string(i1) << " and " << std::to_string(i2) << ": " << distance << std::endl;
}


void ProgClassifyMapCluster::classicalMDS(
    Matrix2D<double>& D,     	   // NxN distance matrix
    Matrix2D<double>& B,           // NxN transformed matrix (optional but useful)
    Matrix1D<double>& eigenvals,   // size N
    Matrix2D<double>& eigenvecs    // NxN (columns = eigenvectors)
)
{
    // ------------------------------------------------------------
    // Step 1: compute D^2 (element-wise square)
    // ------------------------------------------------------------

	Matrix2D<double> D2;
	D2.initZeros(Ndim, Ndim);

	for(size_t i=0; i<Ndim; ++i)
	{
		for(size_t j=0; j<Ndim; ++j)
		{
			MAT_ELEM(D2, i, j) = MAT_ELEM(D, i, j) * MAT_ELEM(D, i, j);
		}
	}

	#ifdef DEBUG_MDS
	std::cout << "--D2 matrix" << std::endl;

	for(size_t i = 0; i < Ndim; i++)
	{
		for(size_t j = 0; j <Ndim; j++)
		{	
			if (j == 0) std::cout << "\n";
			std::cout << std::fixed << std::setprecision(2) << MAT_ELEM(D2, i, j) << "\t";

		}
	}
	std::cout << std::endl;
	std::cout << std::endl;
	#endif

    // ------------------------------------------------------------
    // Step 2: build centering matrix J = I - (1/N)
    // ------------------------------------------------------------

	Matrix2D<double> J;
	J.initZeros(Ndim, Ndim);

	Matrix2D<double> I;
	I.initIdentity(Ndim);

	Matrix2D<double> avgMat;
	avgMat.initConstant(Ndim, Ndim, 1.0/Ndim);
	
	J = I - avgMat;

	#ifdef DEBUG_MDS
	std::cout << "--J matrix" << std::endl;

	for(size_t i = 0; i < Ndim; i++)
	{
		for(size_t j = 0; j <Ndim; j++)
		{	
			if (j == 0) std::cout << "\n";
			std::cout << std::fixed << std::setprecision(2) << MAT_ELEM(J, i, j) << "\t";

		}
	}
	std::cout << std::endl;
	std::cout << std::endl;
	#endif

    // ------------------------------------------------------------
    // Step 3: compute B = -0.5 * J * D2 * J
    // ------------------------------------------------------------

    B = -0.5 * J * D2 * J;

	#ifdef DEBUG_MDS
	std::cout << "--B matrix" << std::endl;

	for(size_t i = 0; i < Ndim; i++)
	{
		for(size_t j = 0; j <Ndim; j++)
		{	
			if (j == 0) std::cout << "\n";
			std::cout << std::fixed << std::setprecision(2) << MAT_ELEM(B, i, j) << "\t";

		}
	}
	std::cout << std::endl;
	std::cout << std::endl;
	#endif

    // ------------------------------------------------------------
    // Step 4: enforce symmetry (numerical safety)
    // ------------------------------------------------------------

	for(size_t i=0; i<Ndim; ++i)
	{
		for(size_t j=0; j<i; ++j)
		{
			double avg = (MAT_ELEM(B, i, j) + MAT_ELEM(B, j, i)) / 2.0;
			MAT_ELEM(B, i, j) = avg;
			MAT_ELEM(B, j, i) = avg;
		}
	}

	#ifdef DEBUG_MDS
	std::cout << "--B matrix symmetry enforced" << std::endl;

	for(size_t i = 0; i < Ndim; i++)
	{
		for(size_t j = 0; j <Ndim; j++)
		{	
			if (j == 0) std::cout << "\n";
			std::cout << std::fixed << std::setprecision(2) << MAT_ELEM(B, i, j) << "\t";

		}
	}
	std::cout << std::endl;
	std::cout << std::endl;
	#endif

    // ------------------------------------------------------------
    // Step 5: eigen decomposition
    // ------------------------------------------------------------

	// This implementation alrady return eigenvalues in descending order and eigenvectors as columns
	// There is no need to sort them
	firstEigs(B, Ndim, eigenvals, eigenvecs);

	#ifdef DEBUG_MDS
	std::cout << "--Eigenvalues" << std::endl;

	for(size_t i = 0; i < Ndim; i++)
	{
		std::cout << std::fixed << std::setprecision(4) << eigenvals[i] << "\t";
	}
	std::cout << std::endl;
	std::cout << std::endl;
	#endif

	#ifdef DEBUG_MDS
	std::cout << "--Eigenvectors matrix" << std::endl;

	for(size_t i = 0; i < Ndim; i++)
	{
		for(size_t j = 0; j <Ndim; j++)
		{	
			if (j == 0) std::cout << "\n";
			std::cout << std::fixed << std::setprecision(2) << MAT_ELEM(eigenvecs, i, j) << "\t";

		}
	}
	std::cout << std::endl;
	std::cout << std::endl;
	#endif
}

void ProgClassifyMapCluster::kmeans(
    Matrix2D<double>& X,  		// N x p embedding
    int k,                      // number of clusters
    int maxIter,				// maximum number of iterations
    Matrix1D<int>& labels       // output classes
)
{
    const size_t N = MAT_YSIZE(X);
    const size_t p = MAT_XSIZE(X);

    // Initialization
    labels.initZeros(N);

    Matrix2D<double> centroids;
    centroids.initZeros(k, p);

    Matrix1D<int> counts;
    counts.initZeros(k);

    // Simple init: first k points
    for (int c = 0; c < k; ++c)
    {
        for (size_t d = 0; d < p; ++d)
            MAT_ELEM(centroids, c, d) = MAT_ELEM(X, c, d);
    }

    // Main loop
    for (int iter = 0; iter < maxIter; ++iter)
    {
        bool changed = false;

        // --- Assignment step
        for (size_t i = 0; i < N; ++i)
        {
            int bestCluster = -1;
            double bestDist = std::numeric_limits<double>::max();

            for (int c = 0; c < k; ++c)
            {
                double dist = 0.0;

                for (size_t d = 0; d < p; ++d)
                {
                    double diff = MAT_ELEM(X, i, d) - MAT_ELEM(centroids, c, d);
                    dist += diff * diff;
                }

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestCluster = c;
                }
            }

            if (labels(i) != bestCluster)
            {
                labels(i) = bestCluster;
                changed = true;
            }
        }

        // Stop if no change
        if (!changed && iter > 0)
            break;

        // --- Update step
        centroids.initZeros(k, p);
        for (int c = 0; c < k; ++c)
            counts(c) = 0;

        // Accumulate
        for (size_t i = 0; i < N; ++i)
        {
            int c = labels(i);
            counts(c)++;

            for (size_t d = 0; d < p; ++d)
                MAT_ELEM(centroids, c, d) += MAT_ELEM(X, i, d);
        }

        // Normalize
        for (int c = 0; c < k; ++c)
        {
            if (counts(c) == 0)
            {
                // Reinitialize empty cluster with random point
                int idx = rand() % N;
                for (size_t d = 0; d < p; ++d)
                    MAT_ELEM(centroids, c, d) = MAT_ELEM(X, idx, d);
            }
            else
            {
                for (size_t d = 0; d < p; ++d)
                    MAT_ELEM(centroids, c, d) /= counts(c);
            }
        }
    }
}

// Utils methods ===================================================================
void ProgClassifyMapCluster::generateSideInfo()
{
    Xdim = XSIZE(V());
    Ydim = YSIZE(V());
    Zdim = ZSIZE(V());

    Xdim_ft = XSIZE(V_ft);
    Ydim_ft = YSIZE(V_ft);
    Zdim_ft = ZSIZE(V_ft);

    composefreqMap();
}

void ProgClassifyMapCluster::composefreqMap()
{
	ft.FourierTransform(V(), V_ft, false);

	// FT dimensions
	Xdim_ft = XSIZE(V_ft);
	Ydim_ft = YSIZE(V_ft);
	Zdim_ft = ZSIZE(V_ft);
	Ndim_ft = NSIZE(V_ft);

    // Use this dimension to initialize mFSC auxiliary maps
    FSC.initZeros(std::min(Xdim_ft, std::min(Ydim_ft, Zdim_ft)));
    FSC_num.initZeros(std::min(Xdim_ft, std::min(Ydim_ft, Zdim_ft)));
    FSC_den1.initZeros(std::min(Xdim_ft, std::min(Ydim_ft, Zdim_ft)));
    FSC_den2.initZeros(std::min(Xdim_ft, std::min(Ydim_ft, Zdim_ft)));

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
	std::string debugFileFn = fn_oroot + "FSC_freqMap.mrc";
	saveImage() = freqMap;
	saveImage.write(debugFileFn);
    #endif
}


void ProgClassifyMapCluster::normalizeFTMap(MultidimArray<std::complex<double>> &volFT)
{
    // Compute avg and std
    std::complex<double> sum(0.0, 0.0);		// Also mean
    double sum2 = 0.0;						// Also std
	int numElems = 0; 

	// Compute sum and sum^2 
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(volFT)
	{
		int freqIdx = static_cast<int>(DIRECT_MULTIDIM_ELEM(freqMap, n));

       	if (freqIdx < NZYXSIZE(FSC))
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
