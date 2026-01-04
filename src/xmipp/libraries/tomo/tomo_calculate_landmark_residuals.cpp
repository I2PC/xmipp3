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

#include "tomo_calculate_landmark_residuals.h"
#include <chrono>



// --------------------------- INFO functions ----------------------------
void ProgTomoCalculateLandmarkResiduals::readParams()
{
	fnVol = getParam("-i");
	fnTiltAngles = getParam("--tlt");
	fnOut = getParam("-o");
	samplingRate = getDoubleParam("--samplingRate");
	fiducialSize = getDoubleParam("--fiducialSize");
	thrSDHCC = getDoubleParam("--thrSDHCC");
 	fnInputCoord = getParam("--inputCoord");
    numberFTdirOfDirections = getIntParam("--numberFTdirOfDirections");
	targetFS = getDoubleParam("--targetLMsize");
}


void ProgTomoCalculateLandmarkResiduals::defineParams()
{
	addUsageLine("This program calculate a set of residual vectors from a detected landmark and its proected 3D coordinate.");
	addParamsLine("  -i <mrcs_file=\"\">                   						: Input tilt-series as metadata or image file.");
	addParamsLine("  --tlt <xmd_file=\"\">      								: Input file containning the tilt angles of the tilt-series in .xmd format.");
	addParamsLine("  --inputCoord <output=\"\">									: Input coordinates of the 3D landmarks. Origin at top left coordinate (X and Y always positive) and centered at the middle of the volume (Z positive and negative).");
	addParamsLine("  [-o <output=\"./alignemntReport.xmd\">]       				: Output file containing the alignemnt report.");
	addParamsLine("  [--samplingRate <samplingRate=1>]							: Sampling rate of the input tomogram (A/px).");
	addParamsLine("  [--fiducialSize <fiducialSize=100>]						: Fiducial size in Angstroms (A).");
	addParamsLine("  [--thrSDHCC <thrSDHCC=5>]      							: Threshold number of SD a coordinate value must be over the mean to consider that it belongs to a high contrast feature.");
	addParamsLine("  [--numberFTdirOfDirections <numberFTdirOfDirections=8>]	: Number of directions to analyze in the Fourier directional filter.");
	addParamsLine("  [--targetLMsize <targetLMsize=8>]		    				: Target size of landmark when downsampling (px).");
}


void ProgTomoCalculateLandmarkResiduals::generateSideInfo()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Generating side info..." << std::endl;
	#endif

	// Read tilt angles file
	MetaDataVec inputTiltAnglesMd;
	inputTiltAnglesMd.read(fnTiltAngles);

	size_t objIdTlt;

	double tiltAngle;

	for(size_t objIdTlt : inputTiltAnglesMd.ids())
	{
		inputTiltAnglesMd.getValue(MDL_ANGLE_TILT, tiltAngle, objIdTlt);
		tiltAngles.push_back(tiltAngle);
	}

	#ifdef VERBOSE_OUTPUT
	std::cout << "Input tilt angles read from: " << fnTiltAngles << std::endl;
	#endif


	// Initialize local alignment vector (depends on the number of acquisition angles)
	localAlignment.resize(nSize, true);

	// Update thresholds depending on input tilt-series sampling rate
	fiducialSizePx = fiducialSize / samplingRate; 

	#ifdef VERBOSE_OUTPUT
	std::cout << "Thresholds:" << std::endl;
	std::cout << "fiducialSizePx: "<< fiducialSizePx << std::endl;
	#endif

	#ifdef VERBOSE_OUTPUT
	std::cout << "Side info generated succesfully!" << std::endl;
	#endif

	// Read input coordinates file
	MetaDataVec inputCoordMd;
	inputCoordMd.read(fnInputCoord);

	size_t objIdCoord;

	int goldBeadX;
	int goldBeadY;
	int goldBeadZ;

	for(size_t objIdCoord : inputCoordMd.ids())
	{
		inputCoordMd.getValue(MDL_XCOOR, goldBeadX, objIdCoord);
		inputCoordMd.getValue(MDL_YCOOR, goldBeadY, objIdCoord);
		inputCoordMd.getValue(MDL_ZCOOR, goldBeadZ, objIdCoord);

		Point3D<int> coord3D(goldBeadX, goldBeadY, goldBeadZ);
		inputCoords.push_back(coord3D);
	}

	#ifdef VERBOSE_OUTPUT
	std::cout << "Input coordinates read from: " << fnInputCoord << std::endl;
	#endif

	numberOfInputCoords = inputCoords.size();

	#ifdef VERBOSE_OUTPUT
	std::cout << "Number of input coordinates: " << numberOfInputCoords << std::endl;
	#endif

	// Initialize landmark detector
	size_t lastIndex = fnOut.find_last_of("\\/");
	std::string lmCoordsFn = fnOut.substr(0, lastIndex);
	lmCoordsFn = lmCoordsFn + "/landmarkCoordinates.xmd";

	lmDetector.fnVol = fnVol;
	lmDetector.fnOut = lmCoordsFn;
	lmDetector.samplingRate = samplingRate;
	lmDetector.fiducialSize = fiducialSize;
	lmDetector.targetFS = targetFS;
	lmDetector.thrSD = thrSDHCC;
	lmDetector.numberFTdirOfDirections = numberFTdirOfDirections;

	#ifdef VERBOSE_OUTPUT
	std::cout << "----- Run landmark detector" << std::endl;
	#endif

	lmDetector.run();

	#ifdef VERBOSE_OUTPUT
	std::cout << "----- Landmark detector execution finished!" << std::endl;
	#endif
}


// --------------------------- HEAD functions ----------------------------
void ProgTomoCalculateLandmarkResiduals::calculateResidualVectors()
{
	// ***TODO: homogeneizar PointXD y MatrixXD
	#ifdef VERBOSE_OUTPUT
	std::cout << "Calculating residual vectors" << std::endl;
	#endif

	size_t objId;
	size_t minIndex;
	double tiltAngle;
	double distance;
	double minDistance;

	int goldBeadX;
	int goldBeadY;
	int goldBeadZ;

	Matrix2D<double> projectionMatrix;
	Matrix1D<double> goldBead3d;
	Matrix1D<double> projectedGoldBead;

	std::vector<Point2D<double>> coordinatesInSlice;

	goldBead3d.initZeros(3);

	// Iterate through every tilt-image
	for(size_t n = 0; n < tiltAngles.size(); n++)
	{	
		#ifdef DEBUG_RESID
		std::cout << "Analyzing coorinates in image "<< n <<std::endl;
		#endif

		tiltAngle = tiltAngles[n];

		# ifdef DEBUG
		std::cout << "Calculating residual vectors at slice " << n << " with tilt angle " << tiltAngle << "º" << std::endl;
		#endif

		coordinatesInSlice = getCoordinatesInSlice(n);

		projectionMatrix = getProjectionMatrix(tiltAngle);

		#ifdef DEBUG_RESID
		std::cout << "Projection matrix------------------------------------"<<std::endl;
		std::cout << MAT_ELEM(projectionMatrix, 0, 0) << " " << MAT_ELEM(projectionMatrix, 0, 1) << " " << MAT_ELEM(projectionMatrix, 0, 2) << std::endl;
		std::cout << MAT_ELEM(projectionMatrix, 1, 0) << " " << MAT_ELEM(projectionMatrix, 1, 1) << " " << MAT_ELEM(projectionMatrix, 1, 2) << std::endl;
		std::cout << MAT_ELEM(projectionMatrix, 2, 0) << " " << MAT_ELEM(projectionMatrix, 2, 1) << " " << MAT_ELEM(projectionMatrix, 2, 2) << std::endl;
		std::cout << "------------------------------------"<<std::endl;

		std::cout << "---------------- Coordinates in slice " << n << "------------------------------------" << std::endl;
		for(size_t i = 0; i < coordinatesInSlice.size(); i++)
		{
			std::cout << coordinatesInSlice[i].x << ", " << coordinatesInSlice[i].y << std::endl;
		}
		
		std::cout << "=============================================================================================================" << std::endl;
		#endif
		

		if (coordinatesInSlice.size() != 0)
		{
			size_t coordinate3dId = 0;

			// Iterate through every input 3d gold bead coordinate and project it onto the tilt image
			for(int j = 0; j < numberOfInputCoords; j++)
			{
				minDistance = MAXDOUBLE;

				goldBeadX = inputCoords[j].x;
				goldBeadY = inputCoords[j].y;
				goldBeadZ = inputCoords[j].z;

				#ifdef DEBUG_RESID
				std::cout << "=============================================================================================================" << std::endl;
				std::cout << "Analyzing image " << n << std::endl;
				std::cout << "goldBeadX " << goldBeadX << std::endl;
				std::cout << "goldBeadY " << goldBeadY << std::endl;
				std::cout << "goldBeadZ " << goldBeadZ << std::endl;
				#endif

				// Update coordinates wiht origin as the center of the tomogram (needed for rotation matrix multiplicaiton)
				XX(goldBead3d) = (double) (goldBeadX - (double)xSize/2);
				YY(goldBead3d) = (double) goldBeadY; // Since we are rotating respect to Y axis, no conversion is needed
				ZZ(goldBead3d) = (double) (goldBeadZ);

				projectedGoldBead = projectionMatrix * goldBead3d;

				XX(projectedGoldBead) += (double)xSize/2;
				// YY(projectedGoldBead) += 0; // Since we are rotating respect to Y axis, no conersion is needed

				// Check that the coordinate is not proyected out of the interpolation edges for this tilt-image
				bool coordInIC = checkProjectedCoordinateInInterpolationEdges(projectedGoldBead, n);
				
				if (coordInIC)
				{
					#ifdef DEBUG_RESID
					std::cout << "XX(goldBead3d) " << XX(goldBead3d) << std::endl;
					std::cout << "YY(goldBead3d) " << YY(goldBead3d) << std::endl;
					std::cout << "ZZ(goldBead3d) " << ZZ(goldBead3d) << std::endl;

					std::cout << "tiltAngles[n] " << tiltAngles[n] << std::endl;
					std::cout << "XX(projectedGoldBead) " << XX(projectedGoldBead) << std::endl;
					std::cout << "YY(projectedGoldBead) " << YY(projectedGoldBead) << std::endl;
					std::cout << "ZZ(projectedGoldBead) " << ZZ(projectedGoldBead) << std::endl;
					#endif

					// Iterate though every coordinate in the tilt-image and calculate the minimum distance
					for(size_t i = 0; i < coordinatesInSlice.size(); i++)
					{
						distance = (XX(projectedGoldBead) - coordinatesInSlice[i].x)*(XX(projectedGoldBead) - coordinatesInSlice[i].x) + (YY(projectedGoldBead) - coordinatesInSlice[i].y)*(YY(projectedGoldBead) - coordinatesInSlice[i].y);

						if(distance < minDistance)
						{
							minDistance = distance;
							minIndex = i;
						}
					}

					// Back to Xmipp origin (top-left corner)
					XX(goldBead3d) = (double) goldBeadX;

					Point3D<double> cis(coordinatesInSlice[minIndex].x, coordinatesInSlice[minIndex].y, n);
					Point3D<double> c3d(XX(goldBead3d), YY(goldBead3d), ZZ(goldBead3d));
					Point2D<double> res(XX(projectedGoldBead)-coordinatesInSlice[minIndex].x, YY(projectedGoldBead) - coordinatesInSlice[minIndex].y); 

					#ifdef DEBUG_RESID
					std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
					std::cout << "XX(projectedGoldBead) " << XX(projectedGoldBead) << std::endl;
					std::cout << "YY(projectedGoldBead) " << YY(projectedGoldBead) << std::endl;
					std::cout << "ZZ(projectedGoldBead) " << ZZ(projectedGoldBead) << std::endl;

					std::cout << "XX(goldBead3d) " << XX(goldBead3d) << std::endl;
					std::cout << "YY(goldBead3d) " << YY(goldBead3d) << std::endl;
					std::cout << "ZZ(goldBead3d) " << ZZ(goldBead3d) << std::endl;

					std::cout << "coordinatesInSlice[minIndex].x " << coordinatesInSlice[minIndex].x << std::endl;
					std::cout << "coordinatesInSlice[minIndex].y " << coordinatesInSlice[minIndex].y << std::endl;

					std::cout << "coordinatesInSlice[minIndex].x - XX(projectedGoldBead) " << coordinatesInSlice[minIndex].x - XX(projectedGoldBead) << std::endl;
					std::cout << "coordinatesInSlice[minIndex].y - YY(projectedGoldBead) " << coordinatesInSlice[minIndex].y - YY(projectedGoldBead) << std::endl;

					std::cout << "minDistance " << minDistance << std::endl;
					std::cout << "sqrt(minDistance) " << sqrt(minDistance) << std::endl;
					std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
					#endif


					CM cm {cis, c3d, res, coordinate3dId};
					vCM.push_back(cm);

					coordinate3dId += 1;
				}
			}
		}else
		{
			std::cout << "WARNING: No coorinate peaked in slice " << n << ". IMPOSIBLE TO STUDY MISALIGNMENT IN THIS SLICE." << std::endl;
		}
	}

	#ifdef VERBOSE_OUTPUT
	std::cout << "Residual vectors calculated: " << vCM.size() << std::endl;
	#endif
}


void ProgTomoCalculateLandmarkResiduals::pruneResidualVectors()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Pruning resudial vectors..." << std::endl;
	#endif

	MultidimArray<double> fiducial_origin;
	MultidimArray<double> fiducial_end;

	std::vector<CM> vCM_pruned;

	std::vector<Point3D<double>> newCoordinates3D;

	int deletedCoordinates = 0;

	int boxSize = 2 * int(fiducialSizePx);
	int halfBoxSize = boxSize / 2;

	int numberOfCM = vCM.size();

	for(size_t n = 0; n < numberOfCM ; n++)
	{
		CM cm = vCM[n];

		Point3D<double> detectedCoordinate = cm.detectedCoordinate;
        Point2D<double> residuals = cm.residuals;

		int fiducialOriginX = detectedCoordinate.x - halfBoxSize;
		int fiducialOriginY = detectedCoordinate.y - halfBoxSize;
		int fiducialEndX = fiducialOriginX + residuals.x;
		int fiducialEndY = fiducialOriginY + residuals.y;
		int tiltImage = detectedCoordinate.z;

		#ifdef DEBUG_PRUNE_RESIDUALS
		std::cout << "--- Analyzing residual number " << n << std::endl;
		std::cout << "tiltImage " << tiltImage << std::endl;
		std::cout << "fiducialOriginX " << fiducialOriginX << std::endl;
		std::cout << "fiducialOriginY " << fiducialOriginY << std::endl;
		std::cout << "fiducialEndX " << fiducialEndX << std::endl;
		std::cout << "fiducialEndY " << fiducialEndY << std::endl;
		std::cout << "residuals.x " << residuals.x << std::endl;
		std::cout << "residuals.y " << residuals.y << std::endl;
		std::cout << "residual module ^2 " << (residuals.x*residuals.x + residuals.y*residuals.y) << std::endl;
		#endif

		// Check that residual do not point to the fiducial itself
		// (only run test for non-matching vectors)
		if ((residuals.x*residuals.x + residuals.y*residuals.y) > (fiducialSizePx * fiducialSizePx / 4))
		{
			// Construct cropped fiducial at origin and end of the resildual vector
			fiducial_origin.initZeros(boxSize, boxSize);
			fiducial_end.initZeros(boxSize, boxSize);

			for(int j = 0; j < boxSize; j++) // xDim
			{
				for(int i = 0; i < boxSize; i++) // yDim
				{
					// Origin
					if ((fiducialOriginY + i) < 0 || (fiducialOriginY + i) >= ySize ||
						(fiducialOriginX + j) < 0 || (fiducialOriginX + j) >= xSize)
					{
						DIRECT_A2D_ELEM(fiducial_origin, i, j) = 0;
					}
					else
					{
						DIRECT_A2D_ELEM(fiducial_origin, i, j) = DIRECT_A3D_ELEM(lmDetector.tiltSeries,
																				 tiltImage,
																				 fiducialOriginY + i, 
																				 fiducialOriginX + j);
					}

					// End
					if ((fiducialEndY + i) < 0 || (fiducialEndY + i) >= ySize ||
						(fiducialEndX + j) < 0 || (fiducialEndX + j) >= xSize)
					{
						DIRECT_A2D_ELEM(fiducial_end, i, j) = 0;
					}
					else
					{
						DIRECT_A2D_ELEM(fiducial_end, i, j) = DIRECT_A3D_ELEM(lmDetector.tiltSeries,
																			  tiltImage,
																			  fiducialEndY + i, 
																			  fiducialEndX + j);
					}
				}
			}

			fiducial_origin.statisticsAdjust(0.0, 1.0);
			fiducial_end.statisticsAdjust(0.0, 1.0);

			#ifdef DEBUG_PRUNE_RESIDUALS
			Image<double> fiducial;

			std::cout << "Origin fiducial dimensions (" << XSIZE(fiducial_origin) << ", " << YSIZE(fiducial_origin) << ")" << std::endl;
			fiducial() = fiducial_origin;
			size_t lastindex = fnOut.find_last_of(".");
			std::string rawname = fnOut.substr(0, lastindex);
			std::string outputFileNameSubtomo;
			outputFileNameSubtomo = rawname + "_" + std::to_string(n) + "_fid_origin.mrc";
			fiducial.write(outputFileNameSubtomo);

			std::cout << "End fiducial dimensions (" << XSIZE(fiducial_end) << ", " << YSIZE(fiducial_end) << ")" << std::endl;
			fiducial() = fiducial_end;
			outputFileNameSubtomo = rawname + "_" + std::to_string(n) + "_fid_end.mrc";
			fiducial.write(outputFileNameSubtomo);
			#endif

			double dotProduct = 0;

			// Calculate scalar product
			for(int j = 0; j < boxSize; j++) // xDim
			{
				for(int i = 0; i < boxSize; i++) // yDim
				{
					dotProduct += DIRECT_A2D_ELEM(fiducial_origin, i, j) * DIRECT_A2D_ELEM(fiducial_end, i, j);
				}
			}

			dotProduct = dotProduct /(boxSize * boxSize);

			#ifdef DEBUG_PRUNE_RESIDUALS
			std::cout << "Dot product: " << dotProduct << std::endl;
			#endif

			if (dotProduct > 0.3)
			{
				vCM.erase(vCM.begin()+n);
				numberOfCM--;
				n--;
				deletedCoordinates++;

				#ifdef DEBUG_PRUNE_RESIDUALS
				std::cout << "Residual " << n << " removed. Origin-end correlation: " << dotProduct << std::endl;
				#endif
			}
		}
	}

	#ifdef DEBUG_PRUNE_RESIDUALS
	std::cout << "Number of residuals removed by origin-end correlation: " << deletedCoordinates << std::endl;
	#endif

	#ifdef VERBOSE_OUTPUT
	std::cout << "Pruning resudial vectors finished succesfully!" << std::endl;
	#endif
}


// --------------------------- I/O functions ----------------------------
void ProgTomoCalculateLandmarkResiduals::writeOutputVCM()
{
	MetaDataVec md;
	size_t id;

	for(size_t i = 0; i < vCM.size(); i++)
	{
		id = md.addObject();
		md.setValue(MDL_X, vCM[i].detectedCoordinate.x, id);
		md.setValue(MDL_Y, vCM[i].detectedCoordinate.y, id);
		md.setValue(MDL_Z, vCM[i].detectedCoordinate.z, id);
		md.setValue(MDL_XCOOR, (int)vCM[i].coordinate3d.x, id);
		md.setValue(MDL_YCOOR, (int)vCM[i].coordinate3d.y, id);
		md.setValue(MDL_ZCOOR, (int)vCM[i].coordinate3d.z, id);
		md.setValue(MDL_SHIFT_X, vCM[i].residuals.x, id);
		md.setValue(MDL_SHIFT_Y, vCM[i].residuals.y, id);
		md.setValue(MDL_FRAME_ID, vCM[i].id, id);

	}

	md.write(fnOut);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Vector coordinates model metadata saved at: " << fnOut << std::endl;
	#endif
}


// --------------------------- MAIN ----------------------------------
void ProgTomoCalculateLandmarkResiduals::run()
{
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;	

	auto t1 = high_resolution_clock::now();

	std::cout << "Starting..." << std::endl;

	size_t Xdim, Ydim;

	MetaDataVec tiltseriesmd;
    ImageGeneric tiltSeriesImages;

    if (fnVol.isMetaData())
    {
        tiltseriesmd.read(fnVol);
    }
    else
    {
        tiltSeriesImages.read(fnVol, HEADER);

        size_t Zdim, Ndim;
        tiltSeriesImages.getDimensions(Xdim, Ydim, Zdim, Ndim);

        if (fnVol.getExtension() == "mrc" and Ndim == 1)
            Ndim = Zdim;

        size_t id;
        FileName fn;
        for (size_t i = 0; i < Ndim; i++) 
        {
            id = tiltseriesmd.addObject();
            fn.compose(i + FIRST_IMAGE, fnVol);
            tiltseriesmd.setValue(MDL_IMAGE, fn, id);
        }
    }

	tiltSeriesImages.getDimensions(xSize, ySize, zSize, nSize);

	generateSideInfo();
	coordinates3D = lmDetector.coordinates3D;

	#ifdef DEBUG_OUTPUT_FILES
	MultidimArray<int> proyectedCoordinates;
	proyectedCoordinates.initZeros(ySize, xSize);

	for(size_t n; n < coordinates3D.size(); n++)
	{
		fillImageLandmark(proyectedCoordinates, (int)coordinates3D[n].x, (int)coordinates3D[n].y, 1);
	}

	size_t lastindexBis = fnOut.find_last_of("\\/");
	std::string rawnameBis = fnOut.substr(0, lastindexBis);
	std::string outputFileNameFilteredVolumeBis;
    outputFileNameFilteredVolumeBis = rawnameBis + "/ts_proyected.mrc";

	Image<int> saveImageBis;
	saveImageBis() = proyectedCoordinates;
	saveImageBis.write(outputFileNameFilteredVolumeBis);
	#endif

	calculateResidualVectors();
	pruneResidualVectors();

	writeOutputVCM();

	adjustCoordinatesCosineStreching();

	auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1); 	// Getting number of milliseconds as an integer
 	std::cout << "Execution time: " << ms_int.count() << "ms\n";
}


// --------------------------- UTILS functions ----------------------------
void ProgTomoCalculateLandmarkResiduals::fillImageLandmark(MultidimArray<int> &proyectedImage, int x, int y, int value)
{
	int fiducialSizePxRatio = 6;	// Relative size of the region to be filled
	int distance = fiducialSizePx/fiducialSizePxRatio;

	for (int i = -distance; i <= distance; i++)
	{
		for (int j = -distance; j <= distance; j++)
		{
			if (j + y > 0 && i + x  > 0 && j + y < ySize && i + x < xSize && i*i+j*j <= distance*distance)
			{
				DIRECT_A2D_ELEM(proyectedImage, (j + y), (i + x)) = value;
			}
		}
	}
}


void ProgTomoCalculateLandmarkResiduals::adjustCoordinatesCosineStreching()
{
	MultidimArray<int> csProyectedCoordinates;
	csProyectedCoordinates.initZeros(ySize, xSize);

	Point3D<double> dc;
	Point3D<double> c3d;
	int xTA = (int)(xSize/2);

	int goldBeadX;
	int goldBeadY;
	int goldBeadZ;

	for(int j = 0; j < numberOfInputCoords; j++)
	{
		goldBeadX = inputCoords[j].x;
		goldBeadY = inputCoords[j].y;
		goldBeadZ = inputCoords[j].z;

		#ifdef DEBUG_COORDS_CS
		std::cout << "Analyzing residuals corresponding to coordinate 3D " << goldBeadX << ", " << goldBeadY << ", " << goldBeadZ << std::endl;
		#endif

	    std::vector<CM> vCMc;
		getCMFromCoordinate(goldBeadX, goldBeadY, goldBeadZ, vCMc);

		std::vector<Point2D<double>> residuals;
		for (size_t i = 0; i < vCMc.size(); i++)
		{
			residuals.push_back(vCMc[i].residuals);
		}

		#ifdef DEBUG_COORDS_CS
		std::cout << " vCMc.size()" << vCMc.size() << std::endl;
		#endif

		// These are the proyected 2D points. The Z component is the id for each 3D coordinate (cluster projections).
		std::vector<Point3D<double>> proyCoords;

		for (size_t i = 0; i < vCMc.size(); i++)
		{
			CM cm = vCMc[i];
			double tiltAngle = tiltAngles[(int)cm.detectedCoordinate.z]* PI/180.0;

			#ifdef DEBUG_COORDS_CS
			std::cout << "------------------------------------------------------------------------------------------------" << std::endl;

			std::cout << "xTA=" << xTA << std::endl;
			
			std::cout << "cm.detectedCoordinate.x=" << cm.detectedCoordinate.x << std::endl;
			std::cout << "cm.detectedCoordinate.y=" << cm.detectedCoordinate.y << std::endl;
			std::cout << "cm.detectedCoordinate.z=" << cm.detectedCoordinate.z << std::endl; 

			std::cout << "cm.coordinate3d.x=" << cm.coordinate3d.x << std::endl;
			std::cout << "cm.coordinate3d.y=" << cm.coordinate3d.y << std::endl;
			std::cout << "cm.coordinate3d.z=" << cm.coordinate3d.z << std::endl; 
			
			std::cout << "(int)cm.detectedCoordinate.x=" << (int)cm.detectedCoordinate.x << std::endl;
			std::cout << "(int)cm.detectedCoordinate.y=" << (int)cm.detectedCoordinate.y << std::endl;
			std::cout << "(int)cm.detectedCoordinate.z=" << (int)cm.detectedCoordinate.z << std::endl;

			std::cout << "(int)cm.coordinate3d.x=" << (int)cm.coordinate3d.x << std::endl;
			std::cout << "(int)cm.coordinate3d.y=" << (int)cm.coordinate3d.y << std::endl;
			std::cout << "(int)cm.coordinate3d.z=" << (int)cm.coordinate3d.z << std::endl; 

			std::cout << "tiltAngle=" << tiltAngle << std::endl;
			std::cout << "cos(tiltAngle)=" << cos(tiltAngle) << std::endl;
			std::cout << "sin(tiltAngle)=" << sin(tiltAngle) << std::endl;

			std::cout << "Xo=" << (int) (((cm.detectedCoordinate.x-xTA)-((cm.coordinate3d.z)*sin(tiltAngle))/cos(tiltAngle))+xTA) << std::endl;
			#endif

			Point3D<double> proyCoord(((((cm.detectedCoordinate.x-xTA)-((cm.coordinate3d.z)*sin(tiltAngle)))/cos(tiltAngle))+xTA), 
									  cm.detectedCoordinate.y,
									  i);
			proyCoords.push_back(proyCoord);

			#ifdef DEBUG_OUTPUT_FILES
			fillImageLandmark(csProyectedCoordinates,
							  (int) ((((cm.detectedCoordinate.x-xTA)-((cm.coordinate3d.z)*sin(tiltAngle)))/cos(tiltAngle))+xTA),
							  (int)cm.detectedCoordinate.y,
							  i);
			#endif
		}
	}

	#ifdef DEBUG_OUTPUT_FILES
	size_t li = fnOut.find_last_of("\\/");
	std::string rn = fnOut.substr(0, li);
	std::string ofn;
    ofn = rn + "/ts_proyected_cs.mrc";

	Image<int> si;
	si() = csProyectedCoordinates;
	si.write(ofn);
	#endif
}


Matrix2D<double> ProgTomoCalculateLandmarkResiduals::getProjectionMatrix(double tiltAngle)
{
	double cosTiltAngle = cos(tiltAngle * PI/180.0);
	double sinTiltAngle = sin(tiltAngle * PI/180.0);

	Matrix2D<double> projectionMatrix(3,3);

	MAT_ELEM(projectionMatrix, 0, 0) = cosTiltAngle;
	// MAT_ELEM(projectionMatrix, 0, 1) = 0;
	MAT_ELEM(projectionMatrix, 0, 2) = sinTiltAngle;
	// MAT_ELEM(projectionMatrix, 1, 0) = 0;
	MAT_ELEM(projectionMatrix, 1, 1) = 1;
	// MAT_ELEM(projectionMatrix, 1, 2) = 0;
	MAT_ELEM(projectionMatrix, 2, 0) = -sinTiltAngle;
	// MAT_ELEM(projectionMatrix, 2, 1) = 0;
	MAT_ELEM(projectionMatrix, 2, 2) = cosTiltAngle;

	return projectionMatrix;
}


std::vector<Point2D<double>> ProgTomoCalculateLandmarkResiduals::getCoordinatesInSlice(size_t slice)
{
	std::vector<Point2D<double>> coordinatesInSlice;
	Point2D<double> coordinate(0,0);

	for(size_t n = 0; n < coordinates3D.size(); n++)
	{
		if(slice == coordinates3D[n].z)
		{
			coordinate.x = coordinates3D[n].x;
			coordinate.y = coordinates3D[n].y;
			coordinatesInSlice.push_back(coordinate);
		}
	}

	return coordinatesInSlice;
}


void ProgTomoCalculateLandmarkResiduals::getCMFromCoordinate(int x, int y, int z, std::vector<CM> &vCMc)
{
	for (size_t i = 0; i < vCM.size(); i++)
	{
		auto cm = vCM[i];
		
		if (cm.coordinate3d.x==x && cm.coordinate3d.y==y && cm.coordinate3d.z==z)
		{
			#ifdef DEBUG_COORDS_CS
			std::cout << "ADDED!!!!!! " <<i<<std::endl;
			#endif

			vCMc.push_back(cm);
		}
	}
	
	#ifdef DEBUG_COORDS_CS
	std::cout << "vCMc.size()" << vCMc.size() << std::endl;
	#endif
}


bool ProgTomoCalculateLandmarkResiduals::checkProjectedCoordinateInInterpolationEdges(Matrix1D<double> projectedCoordinate, size_t slice)
{
	std::vector<Point2D<int>> interpolationLimits = lmDetector.interpolationLimitsVector[slice];

	int x = (int)(XX(projectedCoordinate));
	int y = (int)(YY(projectedCoordinate));

	if (x >= interpolationLimits[y].x && x <= interpolationLimits[y].y)
	{
		return true;
	}
	else
	{
		return false;
	}
}
