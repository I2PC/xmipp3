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

#include "tomo_extract_particlestacks.h"
#include <core/bilib/kernel.h>
#include <numeric>

void ProgTomoExtractParticleStacks::readParams()
{
	fnTs = getParam("--tiltseries");
	fnCoor = getParam("--coordinates");
	boxsize = getIntParam("--boxsize");
	sampling = getDoubleParam("--sampling");
	defocusPositive = checkParam("--defocusPositive");
	invertContrast = checkParam("--invertContrast");
	// scaleFactor = getDoubleParam("--downsample");
	normalize = checkParam("--normalize");
    swapXY = checkParam("--swapXY");
    setCTF = checkParam("--setCTF");
	fnOut = getParam("-o");
	nthrs = getIntParam("--threads");
}


void ProgTomoExtractParticleStacks::defineParams()
{
	addUsageLine("This function takes a tomogram an extract a set of subtomogram from it. The coordinates of the subtomograms are speciffied in the metadata given by coordinates.");
	addParamsLine("  --tiltseries <xmd_file=\"\">       : Metadata (.xmd file) with the tilt series");
	addParamsLine("  --coordinates <xmd_file=\"\">      : Metadata (.xmd file) with the coordidanates to be extracted from the tomogram");
	addParamsLine("  --boxsize <boxsize=100>            : Particle box size in voxels.");
	addParamsLine("  --sampling <s=1>                   : Sampling rate in (A).");
	addParamsLine("  [--defocusPositive]                : This flag must be put if the defocus increases or decreases along the z-axis. This is requires to set the local CTF");
	addParamsLine("  [--invertContrast]                 : Put this flag if the particles to be extracted are 3D particles (subtvolumes)");
    addParamsLine("  [--swapXY]                         : Put this flag if the tomogram and the tilt series have the same dimensions but the X and Y coordinates are swaped");
    addParamsLine("  [--setCTF]                         : Put this flag if the tilt series metadata has CTF parameters. The CTF per particle will be calculated and set in the final set of particles");
	addParamsLine("  [--normalize]                      : Put this flag to normalize the background of the particle (zero mean and unit std)");
	// addParamsLine("  [--downsample <scaleFactor=0.5>]   : Scale factor of the extracted subtomograms");
	addParamsLine("  -o <mrc_file=\"\">                 : path to the output directory. ");
	addParamsLine("  [--threads <s=4>]                  : Number of threads");
}

void ProgTomoExtractParticleStacks::createCircle(MultidimArray<double> &maskNormalize)
{
	int halfboxsize = 0.5* boxsize;
	if (normalize)
	{
		
		maskNormalize.initZeros(1, 1, boxsize, boxsize);

		for (int i=0; i<boxsize; i++)
		{
			int i2 = i-halfboxsize;
			int i2k2 = i2*i2;
			for (int j=0; j<boxsize; j++)
			{
				int j2 = (j- halfboxsize);
				if (sqrt(i2k2 + j2*j2)>halfboxsize)
					A2D_ELEM(maskNormalize, i, j) = 1;
			}
		}
	}
}

void ProgTomoExtractParticleStacks::getCoordinateOnTiltSeries(const double xcoor, const double ycoor, const double zcoor,
															  const double rot, const double tilt, const double tx, const double ty,
															  int &x_2d, int &y_2d)
{
	double ct = cos(tilt);
	double st = sin(tilt);

	double cr = cos(rot);
	double sr = sin(rot);

	/*
	First the picked coordinate, r, is projected on the aligned tilt series
	r' = Pr  Where r is the coordinates to be projected by the matrix P
	[x']   [ct   0   st][x]
	[y'] = [0    1    0][y]
	[z']   [0    0    0][z]
	Next is to undo the transformation, This is Aligned->Unaligned
	If T is the transformation matrix unaligned->aligned, we need T^-{1}
	Let us define a rotation matrix
	R=[cos(rot) -sin(rot)]
	  [ sin(rot) cos(rot)];

	The inverse of T is given by
	T^{-1} = [R' -R'*t;
	         0  0 1]);
	         
	 Where t are the shifts of T
	*/

	x_2d = xcoor * ct + zcoor* st;
	y_2d = ycoor;


	x_2d -= tx;
	y_2d -= ty;
    double x_2d_prime =   cr*x_2d  + sr*y_2d;
	double y_2d_prime =  -sr*x_2d  + cr*y_2d;
    //double x_2d_prime =  x_2d;
	//double y_2d_prime =  y_2d;

        
    if (swapXY)
	{
    	x_2d = x_2d_prime+0.5*Yts;
    	y_2d = y_2d_prime+0.5*Xts;
	}
	else
	{
		x_2d = x_2d_prime+0.5*Xts;
		y_2d = y_2d_prime+0.5*Yts;
	}
}

void ProgTomoExtractParticleStacks::readTiltSeriesInfo()
{
	std::cout << "reading tilt series " << std::endl;
	//TODO: Ensure there is only a single TSId
	MetaDataVec mdts;
	mdts.read(fnTs);

	Image<double> tiltImg;
	auto &ptrtiltImg = tiltImg();

	// The tilt series is stored as a stack of images;
	MultidimArray<double> tsImg;
	std::vector<FileName> tsNames(0);

	FileName fnImg;
	bool flip;

	size_t Nimages = 0;
	double defU=0, defV=0, defAng=0, dose = 0;

	for (const auto& row : mdts)
	{
		row.getValue(MDL_IMAGE, fnImg);
		row.getValue(MDL_ANGLE_TILT, tilt);
		row.getValue(MDL_ANGLE_ROT, rot);
		row.getValue(MDL_FLIP, flip);
		row.getValue(MDL_SHIFT_X, tx);
		row.getValue(MDL_SHIFT_Y, ty);

		if (setCTF)
		{
			row.getValue(MDL_CTF_DEFOCUSU, defU);
			row.getValue(MDL_CTF_DEFOCUSV, defV);
			row.getValue(MDL_CTF_DEFOCUS_ANGLE, defAng);
			//row.getValue(MDL_DOSE, dose);
			tsDefU.push_back(defU);
			tsDefV.push_back(defV);
			tsDefAng.push_back(defAng);

			//tsDose.push_back(dose);
		}
		std::string tsid;
		row.getValue(MDL_TSID, tsid);

		tiltImg.read(fnImg);

		tsImages.push_back(ptrtiltImg);

		tsTiltAngles.push_back(tilt);
		tsRotAngles.push_back(rot);
		tsShiftX.push_back(tx);
		tsShiftY.push_back(ty);
		tsFlip.push_back(flip);
		tsNames.push_back(fnImg);
		Nimages +=1;
	}

	Xts = XSIZE(ptrtiltImg);
	Yts = YSIZE(ptrtiltImg);
}

void ProgTomoExtractParticleStacks::normalizeTiltParticle(const MultidimArray<double> &maskNormalize,
														  MultidimArray<double> &singleImage)
{
	double sumVal = 0, sumVal2 = 0, counter = 0;

	long n = 0;
	for (int i=0; i<boxsize; i++)
	{
		for (int j=0; j<boxsize; j++)
		{
			if (DIRECT_MULTIDIM_ELEM(maskNormalize, n)>0)
			{
				double val = A2D_ELEM(singleImage, i, j);
				sumVal += val;
				sumVal2 += val*val;
				counter = counter + 1;
			}
			n++;
		}
	}

	double mean, sigma2;
	mean = sumVal/counter;
	sigma2 = sqrt(sumVal2/counter - mean*mean)+1e-38;  //To avoid singularities

	for (int i=0; i<boxsize; i++)
	{
		for (int j=0; j<boxsize; j++)
		{
			A2D_ELEM(singleImage, i, j) -= mean;
			A2D_ELEM(singleImage, i, j) /= sigma2;
		}
	}
}


void ProgTomoExtractParticleStacks::extractTiltSeriesParticle(double &xcoor, double &ycoor, double &zcoor, double &signValue,
														const MultidimArray<double> &maskNormalize,
														std::vector<MultidimArray<double>> &imgVec,
														std::string &tsid, size_t &subtomoId, MetaDataVec &mdTSParticle)
{
	MultidimArray<double> singleImage;
	singleImage.initZeros(1,1,boxsize, boxsize);

	int halfboxsize = floor(0.5*boxsize);

	FileName fnMrc;
	fnMrc = tsid + formatString("-%i.mrc", subtomoId);

	size_t tiltId = 1;

	for (size_t idx = 0; idx<tsImages.size(); idx++)
	{
		auto tsImg = tsImages[idx];
		tilt = tsTiltAngles[idx]*PI/180;
		rot = tsRotAngles[idx]*PI/180;
		tx = tsShiftX[idx];
		ty = tsShiftY[idx];
		auto flip = tsFlip[idx];
		int x_2d, y_2d;

		getCoordinateOnTiltSeries(xcoor, ycoor, zcoor, rot, tilt, tx, ty, x_2d, y_2d);

		//std::cout << idx << "     " << tilt*180.0/PI << "     " << rot*180.0/PI << "     " << tx << "     " << ty << "     " << xcoor << "     " << ycoor <<
		//		"     " << zcoor << "     " << x_2d << "     " << y_2d << std::endl;

		int xlim = x_2d + halfboxsize;
		int ylim = y_2d + halfboxsize;

		auto xinit = x_2d - halfboxsize;
		auto yinit = y_2d - halfboxsize;

		if ((xlim>Xts) || (ylim>Yts) || (xinit<0) || (yinit<0))
		{
			continue;
		}

		auto beginElem = MULTIDIM_ARRAY(tsImages[idx]);
		auto lastElem = beginElem + NZYXSIZE(tsImages[idx]);

		const auto has_nan = std::find_if(beginElem, lastElem,
			[] (double x) -> bool
			{
				return std::isnan(x);
			}
		) != lastElem;

		if(has_nan)
			continue;

		for (int i=0; i<boxsize; i++)
		{
			int ii = i+yinit;
			for (int j=0; j<boxsize; j++)
			{
				int jj = j+xinit;
				A2D_ELEM(singleImage, i, j) = signValue*A2D_ELEM(tsImages[idx], ii, jj);
			}
		}

		if (normalize)
			normalizeTiltParticle(maskNormalize, singleImage);


		imgVec.push_back(singleImage);

		MDRowVec rowParticleStack;
		rowParticleStack.setValue(MDL_TSID, tsid);
		rowParticleStack.setValue(MDL_TILTPARTICLEID, tiltId);
		rowParticleStack.setValue(MDL_SUBTOMOID, subtomoId);
		FileName idxstr;
		idxstr = formatString("%i@",tiltId);
		rowParticleStack.setValue(MDL_IMAGE, idxstr+fnMrc);
		rowParticleStack.setValue(MDL_ANGLE_TILT, tsTiltAngles[idx]);
		rowParticleStack.setValue(MDL_ANGLE_ROT, tsRotAngles[idx]);
		rowParticleStack.setValue(MDL_SHIFT_X, tsShiftX[idx]);
		rowParticleStack.setValue(MDL_SHIFT_Y, tsShiftY[idx]);
		//rowParticleStack.setValue(MDL_DOSE, tsDose[idx]);

		double defU=0, defV=0, defAng=0;
		if (setCTF)
		{
			double handness = defocusPositive ? 1.0 : -1.0;

			double Df = (xcoor * cos(tilt) + zcoor* sin(tilt))*sampling*sin(tilt);

			defU = tsDefU[idx]  + handness*Df;
			defV = tsDefV[idx]  + handness*Df;
		}
		rowParticleStack.setValue(MDL_CTF_DEFOCUSU, defU);
		rowParticleStack.setValue(MDL_CTF_DEFOCUSV, defV);
		rowParticleStack.setValue(MDL_CTF_DEFOCUS_ANGLE, tsRotAngles[idx]);
		rowParticleStack.setValue(MDL_XCOOR, x_2d);
		rowParticleStack.setValue(MDL_YCOOR, y_2d);

		mdTSParticle.addRow(rowParticleStack);
		tiltId++;
	}
}


void ProgTomoExtractParticleStacks::run()
{
	std::cout << "Starting ... "<< std::endl;

	// Reading the input data (coordinates and tilt series)
	MetaDataVec mdcoords, mdTSParticle;
	mdcoords.read(fnCoor);

	readTiltSeriesInfo();

	double xcoor, ycoor, zcoor;

	double signValue = invertContrast ? -1 : 1;

	Image<double> finalStack;
	auto &particlestack = finalStack();

	MultidimArray<double> maskNormalize;
	if (normalize)
		createCircle(maskNormalize);

	FileName fnXmd;
	std::string tsid;

	MultidimArray<double> singleImage;
	singleImage.initZeros(1, 1, boxsize, boxsize);

	for (const auto& row : mdcoords)
	{
		row.getValue(MDL_X, xcoor);
		row.getValue(MDL_Y, ycoor);
		row.getValue(MDL_Z, zcoor);
		size_t subtomoId;
		row.getValue(MDL_SUBTOMOID, subtomoId);
		row.getValue(MDL_TSID, tsid);
		
		std::vector<MultidimArray<double>> imgVec;

		extractTiltSeriesParticle(xcoor, ycoor, zcoor, signValue,
				                  maskNormalize, imgVec, tsid, subtomoId, mdTSParticle);

		auto Nimages = imgVec.size();
		if (Nimages == 0)
		{
			std::cout << "coordinate is out of the image" << std::endl;
			continue;
		}

		particlestack.initZeros(Nimages, 1, boxsize, boxsize);

		for (int k=0; k<Nimages; k++)
		{
			singleImage = imgVec[k];
			for (int i=0; i<boxsize; i++)
			{
				for (int j=0; j<boxsize; j++)
				{
					NZYX_ELEM(particlestack, k, 0, i, j) = A2D_ELEM(singleImage, i, j);
				}
			}
		}

		FileName fnMrc, fnXmd;
		fnXmd = tsid + formatString(".xmd");
		mdTSParticle.write(fnOut+"/"+fnXmd);
		fnMrc = tsid + formatString("-%i.mrc", subtomoId);
		finalStack.write(fnOut+"/"+fnMrc);
	}
	std::cout << "-----------DONE-----------" << std::endl;
}

