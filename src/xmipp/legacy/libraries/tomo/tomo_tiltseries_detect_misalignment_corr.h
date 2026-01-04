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

#ifndef __TOMO_TS_DETECT_MISALI_CORR
#define __TOMO_TS_DETECT_MISALI_CORR

#include <iostream>
#include <core/xmipp_program.h>
#include <core/xmipp_image.h>
#include <core/metadata_vec.h>
#include "core/xmipp_image_generic.h"
#include <math.h>
#include <limits>
#include <string>
#include <stdio.h>
#include <fstream>

#define VERBOSE_OUTPUT

// #define DEBUG_DIM
// #define DEBUG_TI_CORR
// #define DEBUG_OUTPUT_FILES

class ProgTomoTSDetectMisalignmentCorr : public XmippProgram
{

public:
    /** Params specific for landmark detectior */
    double targetFS;

    /** Filenames */
    FileName fnIn;
    FileName fnOut;
    FileName fnTiltAngles;
    FileName fnInputCoord;

    /** Input info */
    double fiducialSize;    // Fiducial size in Angstroms
    float fiducialSizePx;   // Fiducial size in pixels
    double samplingRate;

    /** Input tilt-series dimensions */
    size_t xSize;
	size_t ySize;
	size_t zSize;
    size_t nSize;
    size_t normDim;

    /** Vector containig the tilt angles from the series */
    std::vector<double> tiltAngles;

    /** Alignment report. True = aligned / False = misaligned */
    std::vector<bool> localAlignment;

    /** Vector containig the tilt angles from the series */
    std::vector<Matrix2D<double>> relativeShifts;

    /** Input info */
    double shiftTol;


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

    // --------------------------- I/O functions ----------------------------
    /**
     * Write output metadata with shifts
    */
    void writeOutputShifts();

    /**
     * Write output alignent report
    */
    void writeOutputAlignmentReport();

    // --------------------------- UTILS functions ----------------------------
    /**
     * Apply cosine stretching to tilt image
    */
    void cosineStretching(MultidimArray<double> &ti, double ti_angle_high, double ti_angle_low);

    /**
     * Compose matrix for cosine stretching
    */
    Matrix2D<double> getCosineStretchingMatrix(double ti_angle_high, double ti_angle_low);

    /**
     * Calculate shift for maximum correlation between images
    */
    Matrix2D<double> maxCorrelationShift(MultidimArray<double> &ti1, MultidimArray<double> &ti2);

    /**
     * Assess local alginemnt quality
    */
    void detectSubtleMisalingment(MultidimArray<double> &ts);

    /**
     * Correct alignment with calculated shifts
    */
    void refineAlignment(MultidimArray<double> &ts);
    
    /**
     * LPF tilt image
    */
    void lowpassFilter(MultidimArray<double> &tiltImage);

    /**
     * Remove outlier values in tilt image
    */
    // void removeOutliers(MultidimArray<double> &ti);

    // --------------------------- MAIN ----------------------------------
    /**
     * Main
    */
    void run();
};

#endif
