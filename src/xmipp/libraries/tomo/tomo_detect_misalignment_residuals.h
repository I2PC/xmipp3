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

#ifndef __IMAGE_PEAK_HIGH_CONTRAST
#define __IMAGE_PEAK_HIGH_CONTRAST

#include <iostream>
#include <core/xmipp_program.h>
#include <core/xmipp_image.h>
#include <data/point3D.h>
#include <data/point2D.h>

#define VERBOSE_OUTPUT
// #define GENERATE_RESIDUAL_STATISTICS
// #define DEBUG_DIM
// #define DEBUG_RESID
// #define DEBUG_RESIDUAL_ANALYSIS
// #define DEBUG_RESIDUAL_STATISTICS_FILE

class ProgTomoDetectMisalignmentResiduals : public XmippProgram
{

public:
    /** Filenames */
    FileName fnResidualInfo;
    FileName fnOut;

    /** Input info */
    double fiducialSize;    // Fiducial size in Angstroms
    float fiducialSizePx;   // Fiducial size in pixels
    double samplingRate;

    /** Thresholds */
    float thrRatioMahalanobis;   // Maximum ratio of elements (chains on individual residuals) with Mahalnobis distance > 1

    /** Coordinate model structure */
    struct resMod {
        Point3D<double> landmarkCoord;          // Coordinate detected in each tilt-image
        Point3D<double> coordinate3d;           // 3D coordinate whose porjection is the closest
        Point2D<double> residuals;              // Residual vector from detected to projected
        size_t id;                              // ID common for all the CM belonging to the same coordinate 3D
        double mahalanobisDistance;             // Mahalanobis distance to the expected residual distribution
    };
    
    /** Number of tilt-images */
    int nSize;

    /** Execution modes */
    bool removeOutliers;
    bool voteCriteria;


    /** Array of coordinate model structures */
    std::vector<resMod> vResMod;

    /** Vector containig the input 3D coorinates used for alignment */
    size_t numberOfInputCoords;

    /** Alignment report. True = aligned / False = misaligned */
    bool globalAlignment = true;
    std::vector<bool> localAlignment;

    /** Avg and STD of Mahalanobis distance for each tilt image */
    std::vector<double> avgMahalanobisDistanceV;
    std::vector<double> stdMahalanobisDistanceV;

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

    // --------------------------- HEAD functions ----------------------------
    /**
     * Generate side info usefull for the rest of protocols
    */
    void generateSideInfo();

    /**
     * Detect msialignment from a set of residual vectors
    */
    void detectMisalignmentFromResidualsMahalanobis();

    /**
     * Method to exexcute and digest statistical info from a set of residuals
     * executing python script. 
    */
    void generateResidualStatiscticsFile();

    // --------------------------- I/O functions ----------------------------
    /**
     * Write obtained coordinates in output file.
    */
   void readInputResiduals();

    /**
     * Write alignment report.
    */
   void writeOutputAlignmentReport();

    /**
     * Write residuals weighted by Mahalanobis distance.
    */
   void writeWeightedResiduals();

    // --------------------------- UTILS functions ----------------------------
    /**
     * Get all coordinate models associated to the same fiducial.
    */
    void getResModByFiducial(size_t fiducialNumber, std::vector<resMod> &vResMod_fiducial);

    /**
     * Get all coordinate models associated to the same image.
    */
    void getResModByImage(size_t tiltImageNumber, std::vector<resMod> &vResMod_image);


    // --------------------------- MAIN ----------------------------------
    /**
     * Main.
    */
    void run();
};

#endif
