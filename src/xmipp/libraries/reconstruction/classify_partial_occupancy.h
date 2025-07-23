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

#ifndef _PROG_CLASSIFY_PARTIAL_OCCUPANCY
#define _PROG_CLASSIFY_PARTIAL_OCCUPANCY

#include "core/metadata_vec.h"
#include "core/xmipp_program.h"
#include "core/xmipp_image.h"
#include "data/fourier_filter.h"
#include "data/fourier_projection.h"
#include "data/unitCell.h"
#include "core/xmipp_metadata_program.h"

#define DEBUG_COMPUTE_STATISTICS
#define VERBOSE_OUTPUT
// #define DEBUG_FREQUENCY_PROFILE
// #define DEBUG_NOISE_CALCULATION
// #define DEBUG_REGIONS_COMPARISON
// #define DEBUG_RADIAL_AVERAGE
#define DEBUG_METRICS
// #define DEBUG_WRITE_PARICLE
#define DEBUG_OUTPUT_FILES

/**@defgroup ProgClassifyPartialOccupancy Subtract projections
   @ingroup ReconsLibrary */
//@{
/** Subtract projections from particles */

class ProgClassifyPartialOccupancy: public XmippMetadataProgram
 {
 public:
    // Input params
    FileName fnVolR; // Input reference volume
	FileName fnImgI; // Particle filename
    FileName fnMaskRoi; // Input 3D mask for region of interest to keep or subtract
    FileName fnMaskProtein; // Input 3D mask for the specimen
    FileName fnNoiseEst; // Input path to previously calculated noise estimation
    int numParticlesNoiseEst; // Number of particles to compute noise estimation

    // Volume dimensions
    size_t Xdim;
    size_t Ydim;
    size_t Zdim;
    size_t Ndim;

	double padFourier; 
    int maxwiIdx;
    bool realSpaceProjector;
	MultidimArray<int> wi;

    // Data variables
 	Image<double> V; // volume
 	Image<double> vMaskRoi; // ROI mask 3D
 	Image<double> vMaskP; // Protein mask 3D

 	Image<double> M; // mask projected and smooth
 	Image<double> M_P; // mask protein projected and smooth
 	Image<double> I; // particle
    Image<double> IsubP; // projection-subtracted particle

 	Projection P; // projection
 	Projection PmaskProtein; // mask projection for the protein
 	Projection PmaskRoi; // mask projection for region to keep

    const MultidimArray<double> *ctfImage = nullptr; // needed for FourierProjector
	FourierTransformer transformerP; // Fourier transformer for projection
    FourierTransformer transformerI; // Fourier transformer for particle
    FourierTransformer transformerIsubP; // Fourier transformer for particle with subtracted projection
    
    MultidimArray< std::complex<double> > IFourier; // FT(particle)
	MultidimArray< std::complex<double> > PFourier; // FT(projection)
    MultidimArray< std::complex<double> > PFourier0; // FT(projection) estimation of order 0
	MultidimArray< std::complex<double> > PFourier1; // FT(projection) estimation of order 1
    MultidimArray< std::complex<double> > IiMFourier;
	MultidimArray< std::complex<double> > PiMFourier;

    FourierTransformer transformerIiM;
	FourierTransformer transformerPiM;

    CTFDescription ctf;
	FourierFilter FilterCTF;
	Image<double> padp; // padded image when applying CTF
	Image<double> PmaskI; // inverted projected mask
	Image<double> ImgiM; // auxiliary image for computing estimation images
	MultidimArray< std::complex<double> > ImgiMFourier; // FT(ImgiM)

    // Particle metadata
    MetaDataVec mdParticles;
    MDRowVec row;
    Matrix1D<double> roffset; // particle shifts
    struct Angles // particle angles for projection
    {
    	double rot;
    	double tilt;
    	double psi;
    };
    struct Angles part_angles;

    struct AdjustParams // particle adjustment parameters
    {
    	double b;
    	double b0;
    	double b1;
    };
    struct AdjustParams adjustParams; 


    int rank; // for MPI version
    FourierProjector *projector;

    // Variables for noise estimation
    bool computeNoiseEstimation;
    size_t numberParticlesForBoundaryDetermination = 50;
    size_t cropSize = 11;
    MultidimArray< std::complex<double> > noiseSpectrum;
    Image<double> powerNoise;
    
    // Variables for frequency profiling
    double thrModuleFT; // Defined as x% of the value of the frequency with the maximum module
    double maxModuleFT; // Just keep track of the maximum frequency magnitude for normalization
    std::vector<double> radialAvgVolFT; // *** hay un overwritting de esta variable!
    MultidimArray<double> particleFreqMap;

    // Variables for unit cell calculation
    bool symmetryProvided;
 	Image<double> V_unitcell; // volume unit cell for symmetric specimens
    String uc_sym;
    int uc_rmin = 0;
    int uc_rmax;
    int uc_expandFactor = 0;
    int uc_offset = 0;
    double uc_sampling = 1.0;
    double uc_x_origin;
    double uc_y_origin;
    double uc_z_origin;


public:

    // ---------------------- IN/OUT METHODS -----------------------------
    // Define parameters
    void defineParams() override;
    // Read argument from command line
    void readParams() override;
    // Show
    void show() const override;
    /// Read and write methods
    void readParticle(const MDRow &rowIn);
    void writeParticle(MDRow &rowOut, double, double, double);

    // ----------------------- MAIN METHODS ------------------------------
    void preProcess() override;
    void processParticle(const MDRow &rowIn, int sizeImg);
    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut) override;
    void finishProcessing();

    // ----------------------- CORE METHODS ------------------------------
    void unitCellExtraction();
    void frequencyCharacterization();
    void noiseEstimation();
    void compareRegions(double &ll_I, double &ll_IsubP, const FileName &fnImgOut);
    void logLikelihood(double &ll_I_it,
                       double &ll_IsubP_it, 
                       MultidimArray<std::complex<double>> fftI,
                       MultidimArray<std::complex<double>> fftIsubP);
    void radialLogLikelihood(double &ll_I_it, 
		     		         double &ll_IsubP_it, 
							 MultidimArray<std::complex<double>> fftI,
							 MultidimArray<std::complex<double>> fftIsubP);
    void entropy(double &ll_I_it,
                 double &ll_IsubP_it, 
                 MultidimArray<std::complex<double>> fftI,
                 MultidimArray<std::complex<double>> fftIsubP);
    void kullbackLeibler(double &ll_I_it,
                         double &ll_IsubP_it, 
                         MultidimArray<std::complex<double>> fftI,
                         MultidimArray<std::complex<double>> fftIsubP);
    void crossEntropy(double &entropy_I_it, 
					  double &entropy_IsubP_it, 
					  MultidimArray<std::complex<double>> fftI,
					  MultidimArray<std::complex<double>> fftIsubP);

    // ---------------------- UTILS METHODS ------------------------------
    Image<double> binarizeMask(Projection &) const;
    Image<double> invertMask(const Image<double> &);
    void calculateBoundingBox(MultidimArray<double> PmaskRoiLabel, 
                              std::vector<int> &minX, 
                              std::vector<int> &minY, 
                              std::vector<int> &maxX, 
                              std::vector<int> &maxY, 
                              int numLig);
    void calculateRadialAverage(const MultidimArray<std::complex<double>> &particleFT, 
								std::vector<double> &radialAvgVolFT,
                                bool normalize);
    void calculatePowerNoiseRadialAverage(std::vector<double> &radialAvgFT,
										  bool normalize);                                

    // ----------------------- CLASS METHODS ------------------------------
    // Empty constructor
    ProgClassifyPartialOccupancy();

    // Destructor
    ~ProgClassifyPartialOccupancy();

    // ---------------------- UNUSED METHODS ------------------------------
    void computeParticleStats(MultidimArray<double> &mI,
                              double &avg, 
                              double &std, 
                              double &zScore, 
                              double &energy);
 };
 //@}
#endif
