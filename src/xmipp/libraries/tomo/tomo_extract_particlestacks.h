/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
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

#ifndef _PROG_TOMO_EXTRACT_PARTICLESTACKS
#define _PROG_TOMO_EXTRACT_PARTICLESTACKS

#include <iostream>
#include <core/xmipp_program.h>
#include <core/xmipp_image.h>
#include <core/metadata_extension.h>
#include <limits>
#include <complex>
#include <string>


class ProgTomoExtractParticleStacks : public XmippProgram
{
public:
	 /** Filenames */
	FileName fnOut;
    FileName fnTs;
    FileName fnCoor;

    size_t Xdim;
    size_t Ydim;
    size_t Zdim;
    size_t Xts, Yts;
    double tilt, rot, tx, ty;

    bool invertContrast, normalize, swapXY, setCTF, defocusPositive;
    std::vector<double> tsTiltAngles, tsRotAngles, tsShiftX, tsShiftY, tsDefU, tsDefV, tsDefAng, tsDose;
    std::vector<MultidimArray<double> > tsImages;
    std::vector<bool> tsFlip;

    double scaleFactor, sampling;

	/** Is the volume previously masked?*/
	int  boxsize; 
    int nthrs;

public:

    void defineParams();

    void createCircle(MultidimArray<double> &maskNormalize);
    
    void readTiltSeriesInfo();

    void getCoordinateOnTiltSeries(const double xcoor, const double ycoor, const double zcoor,
    							   const double rot, const double tilt, const double tx, const double ty,
								   int &x_2d, int &y_2d);

    void extractTiltSeriesParticle(double &xcoor, double &ycoor, double &zcoor, double &signValue,
									const MultidimArray<double> &maskNormalize,
									std::vector<MultidimArray<double>> &imgVec,
									std::string &tsid, size_t &subtomoId, MetaDataVec &mdTSParticle);

    void normalizeTiltParticle(const MultidimArray<double> &maskNormalize, MultidimArray<double> &singleImage);

    void readParams();

    void run();
};
//@}
#endif
