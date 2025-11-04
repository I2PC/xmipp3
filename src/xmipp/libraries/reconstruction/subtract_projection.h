/***************************************************************************
 *
 * Authors:    Estrella Fernandez Gimenez (me.fernandez@cnb.csic.es)
 *             Federico P. de Isidro-Gomez (federico.pdeisidro@astx.com)
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

 #ifndef _PROG_SUBTRACT_PROJECTION
 #define _PROG_SUBTRACT_PROJECTION

 #include "core/metadata_vec.h"
 #include "core/xmipp_program.h"
 #include "core/xmipp_image.h"
 #include "data/fourier_filter.h"
 #include "data/fourier_projection.h"
 #include "core/xmipp_metadata_program.h"

//  #define DEBUG
//  #define DEBUG_OUTPUT_FILES
 #define DEBUG_NOISE_ESTIMATION


/**@defgroup ProgSubtractProjection Subtract projections
   @ingroup ReconsLibrary */
//@{
/** Subtract projections from particles */

class ProgSubtractProjection: public XmippMetadataProgram
 {
 public:
    // Input params
    FileName fnVolR; // Input reference volume
    FileName fnParticles; // Input metadata
    FileName fnMaskVol; // Input 3D mask of the reference volume
	FileName fnImgI; // Particle filename
    FileName fnMaskRoi; // Input 3D mask for region of interest to keep or subtract
    FileName fnProj; // Path to save intermediate files

	double sampling; 
	double padFourier; 
	double maxResol;
    double cirmaskrad; // Radius of the circular mask
	int sigma;
    int maxwiIdx;
    bool nonNegative;
    bool boost;
    bool subtract;
    bool realSpaceProjector;
    bool maskVolProvided;
    bool ignoreCTF;
	MultidimArray<int> wi;

    // Input volume dimensions
    size_t Xdim;
	size_t Ydim;
	size_t Zdim;
	size_t Ndim;

    // Variables for noise estimation
    bool noiseEstimationBool;
    MultidimArray< double > powerNoise;
    int cropRadius = 5; // Crop size to estimate noise power
    int max_noiseEst;
    int min_noiseEst;

    // Data variables
 	Image<double> V; // volume
 	Image<double> vM; // mask 3D
    Image<double> ivM; // invert mask 3D

 	Image<double> M; // mask projected and smooth
 	Image<double> I; // particle
    Image<double> Pctf; // projection with CTF applied
    Image<double> iM; // inverse mask of the region to keep
    Image<double> Mfinal; // final dilated mask
    Image<double> Idiff; // final subtracted image
	Image<double> maskVol; // mask for reference volume (circular if not provided)	

 	Projection P; // projection
 	Projection Pmask; // mask projection for the protein
 	Image<double> PmaskImg; // mask projection for the protein as Image
 	Projection PmaskRoi; // mask projection for region to keep

    const MultidimArray<double> *ctfImage = nullptr; // needed for FourierProjector
	FourierTransformer transformerP; // Fourier transformer for projection
    FourierTransformer transformerI; // Fourier transformer for particle
    
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

    bool disable;
    /// Read and write methods
    void readParticle(const MDRow &rowIn);
    void writeParticle(MDRow &rowOut, FileName, Image<double> &, double, double, double, double);
    /// Processing methods
    void createMask(const FileName &, Image<double> &, Image<double> &);
    Image<double> binarizeMask(Projection &) const;
    Image<double> invertMask(const Image<double> &);
    Image<double> applyCTF(const MDRow &, Projection &);
    void processParticle(const MDRow &rowIn, int sizeImg);
    MultidimArray< std::complex<double> > computeEstimationImage(const MultidimArray<double> &, 
    const MultidimArray<double> *, FourierTransformer &);
    double evaluateFitting(const MultidimArray< std::complex<double> > &, const MultidimArray< std::complex<double> > &) const;
    Matrix1D<double> checkBestModel(MultidimArray< std::complex<double> > &, const MultidimArray< std::complex<double> > &, 
    const MultidimArray< std::complex<double> > &, const MultidimArray< std::complex<double> > &) const;
    void computeParticleStats(Image<double> &Idiff, Image<double> &iM, FileName fnImgOut, double &avg, double &std, double &zScore);
    void generateNoiseEstimationSideInfo();
    void noiseEstimation();

    int rank; // for MPI version
    FourierProjector *projector;

    // Empty constructor
    ProgSubtractProjection();

    // Destructor
    ~ProgSubtractProjection();

    // Read argument from command line
    void readParams() override;

    // Show
    void show() const override;

    // Define parameters
    void defineParams() override;
    void preProcess() override;
    void processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut) override;
    void finishProcessing();
 };
 //@}
#endif

