/***************************************************************************
 *
 * Authors:    Federico P. de Isidro Gomez			  fp.deisidro@cnb.csic.es
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

#ifndef __TOMO_DETECT_LANDMARKS
#define __TOMO_DETECT_LANDMARKS

#include <core/xmipp_program.h>
#include <iostream>
#include <data/point3D.h>
#include <data/point2D.h>
#include <data/filters.h>
#include <core/xmipp_fft.h>
#include <core/xmipp_fftw.h>
#include <core/xmipp_image_generic.h>
#include <core/xmipp_filename.h>
#include <core/multidim_array.h>

   
#define VERBOSE_OUTPUT

// #define DEBUG_DIM
// #define DEBUG_DOWNSAMPLE
// #define DEBUG_INTERPOLATION_EDGES
// #define DEBUG_SOBEL
// #define DEBUG_CLOSING
// #define DEBUG_FILTERLABEL
// #define DEBUG_HCC
// #define DEBUG_DIRECTIONAL_FOURIER
// #define DEBUG_REFERENCE
// #define DEBUG_CENTER_COORDINATES
// #define DEBUG_OUTPUT_FILES

class ProgTomoDetectLandmarks : public XmippProgram
{

public:
    /** Filenames */
    FileName fnVol;
    FileName fnOut;
    FileName fnTiltAngles;
    FileName fnInputCoord;

    /** Tilt series images */
    MultidimArray<double> tiltSeries;

    /** Input info */
    double fiducialSize;
    double samplingRate;
    int numberFTdirOfDirections;

    /** Input tilt-series dimensions */
    size_t xSize;
	size_t ySize;
	size_t zSize;
    size_t nSize;
    size_t normDim;

    /** Input tilt-series dimensions after downsampling */
    size_t xSize_d;
	size_t ySize_d;
	size_t zSize_d;
    size_t nSize_d;
    size_t normDim_d;

    /** Target fiducial size and downsampling factor */
    double targetFS;
    double ds_factor;
    double thrSD;

    /** Vector for peaked coordinates components */
    std::vector<Point3D<double>> coordinates3D;

    // Vector of points saving the interpolation limits for each tilt-image
    std::vector<std::vector<Point2D<int>>> interpolationLimitsVector;
    std::vector<std::vector<Point2D<int>>> interpolationLimitsVector_ds;

    // Background value to fill interpolation edges
    double backgroundValue;

    // Margin thickness to remove interpolation edges
    int marginThickness;

    /** Fiducial size in pixels */
    float fiducialSizePx;

    // Define the Sobel kernels
    std::vector<std::vector<double>> sobelX = {{-1, 0, 1},
                                            {-2, 0, 2},
                                            {-1, 0, 1}};

    std::vector<std::vector<double>> sobelY = {{-1, -2, -1},
                                            { 0,  0,  0},
                                            { 1,  2,  1}};

    // Landmark reference for enhancement
    MultidimArray<double> landmarkReference;
    MultidimArray<double> landmarkReference_Gaussian;

    // Centralized Fourier transofrmer
    FourierTransformer transformer;

public:

    // --------------------------- INFO functions ----------------------------

    /**
     * Read input program parameters.
    */
    void readParams();

    /**
     * Define input program parameters.
    */
    void defineParams();

    /**
     * Calculate global parameters used through the program.
    */
    void generateSideInfo();


    // --------------------------- HEAD functions ----------------------------

    /**
     * Dowsample input tilt-image to resize landmarks to target size.
    */
    void downsample(MultidimArray<double> &tiltImage, MultidimArray<double> &tiltImage_ds);

    /**
     * Detect interpolation edges before preprocessing.
    */
    void detectInterpolationEdges(MultidimArray<double> &tiltImage);

    /**
     * Apply Sobel filter to input tilt image.
    */
    void sobelFiler(MultidimArray<double> &tiltImage);

    /**
     * Convolve tilt-image with filtered reference landmark (correlation in Fourier
     * space) to enhance landamrks.
    */
    void enhanceLandmarks(MultidimArray<double> &tiltImage);

    /**
     * Peak high contrast coordinates in a volume. Detect coordinates with an outlier value, 
     * generate a binary map to posterior label it, and filter the labeled regions depending on
     * size and shape.
    */
    void getHighContrastCoordinates(MultidimArray<double> tiltSeriesFiltered);

    /**
     * Center the high-contrast coordinates selected according to the density.
    */
    void centerCoordinates(MultidimArray<double> tiltSeries);


    // ---------------------------- I/O functions -----------------------------

    /**
     * Write output coordinates metadata.
    */
    void writeOutputCoordinates();


    // ------------------------------ MAIN ------------------------------------

    /**
     * Run main program.
    */
    void run();

    // --------------------------- UTILS functions ----------------------------

    /**
     * Filter labeled regions. Util method to remove those labeles regions based on their size 
     * (minimum number of points that should contain) and shape (present a globular structure,
     * as expected from a gold bead).
    */
    bool filterLabeledRegions(std::vector<int> coordinatesPerLabelX, std::vector<int> coordinatesPerLabelY, double centroX, double centroY);

    /**
     * Create landmark template on the fly to posteriorly to convolve with each tilt-image to
     * enhance landmarks.
    */
    void createLandmarkTemplate();
    void createLandmarkTemplate_Gaussian();

    /**
     * Maximum pooling operation.
    */
    void maxPooling(MultidimArray<double> &image, size_t windowSize, std::vector<Point2D<int>> interLim);

    /**
     * Fourier directional bandpass filter for landmark enhancement.
    */
    void filterFourierDirections(MultidimArray<double> &imag, size_t k);
    void directionalFilterFourier(MultidimArray<double> &image, double xdir, double ydir);

    /**
     * Compute average and stardard deviation in tilt-image band.
    */
    void computeAvgAndStdevFromMiltidimArray(MultidimArray<double> &tiltImage, double &avg, double &stddev, std::vector<Point2D<int>> interLim, int xMin, int xMax, bool onlyPositive);
};

#endif
