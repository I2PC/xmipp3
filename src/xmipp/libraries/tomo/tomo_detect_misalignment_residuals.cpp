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

#include "tomo_detect_misalignment_residuals.h"
#include <chrono>
#include <core/metadata_vec.h>
#include <fstream>


// --------------------------- INFO functions ----------------------------
void ProgTomoDetectMisalignmentResiduals::readParams()
{
	fnResidualInfo = getParam("--inputResInfo");
    fnOut = getParam("-o");
	samplingRate = getDoubleParam("--samplingRate");
	fiducialSize = getDoubleParam("--fiducialSize");
	nSize = getIntParam("--numberTiltImages");
	removeOutliers = checkParam("--removeOutliers");
	voteCriteria = checkParam("--voteCriteria");
	thrRatioMahalanobis = getDoubleParam("--thrRatioMahalanobis");
}


void ProgTomoDetectMisalignmentResiduals::defineParams()
{
	addUsageLine("This program detect the misaligned images in a tilt-series based on a set of residual vectors.");
	addParamsLine("  --inputResInfo <input=\"\">							: Input file containing residual information of the detected landmarks.");
	addParamsLine("  [-o <output=\"./alignemntReport.xmd\">]       			: Output file containing the alignemnt report.");
	addParamsLine("  [--samplingRate <samplingRate=1>]						: Sampling rate of the input tomogram (A/px).");
	addParamsLine("  [--fiducialSize <fiducialSize=100>]					: Fiducial size in Angstroms (A).");
	addParamsLine("  [--numberTiltImages <numberTiltImages=60>]				: Number of tilt-images. Needed in case some image is missing form residual information.");
	addParamsLine("  [--removeOutliers]										: Remove outliers before calculate mahalanobis distance.");
	addParamsLine("  [--voteCriteria]										: Use a votting system (instead of the average) to detect local misalignment.");
	addParamsLine("  [--thrRatioMahalanobis <thrRatioMahalanobis=0.8>]		: Maximum ratio of residuals with Mahalanobis distance over 1 to consider chain/image as misaligned.");
}


void ProgTomoDetectMisalignmentResiduals::generateSideInfo()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Generating side info..." << std::endl;
	#endif

	// Update thresholds depending on input tilt-series sampling rate
	fiducialSizePx = fiducialSize / samplingRate; 

	#ifdef VERBOSE_OUTPUT
	std::cout << "fiducialSizePx: "<< fiducialSizePx << std::endl;
	#endif

	// Read input residuals file
	readInputResiduals();

	// Get number of inpur coordinates from input residuals
	numberOfInputCoords = 0;

	for (size_t i = 0; i < vResMod.size(); i++)
	{
		if (numberOfInputCoords < vResMod[i].id)
		{
			numberOfInputCoords = vResMod[i].id;
		}
	}

	numberOfInputCoords += 1;

	#ifdef VERBOSE_OUTPUT
	std::cout << "Number of input coordinates: " << numberOfInputCoords << std::endl;
	std::cout << "Number of tilt-images: " << nSize << std::endl;
	#endif

	// Initialize local alignment vector (depends on the number of acquisition angles)
	localAlignment.resize(nSize, true);
	avgMahalanobisDistanceV.resize(nSize, 0.0);
	stdMahalanobisDistanceV.resize(nSize, 0.0);

	#ifdef VERBOSE_OUTPUT
	std::cout << "Side info generated succesfully!" << std::endl;
	#endif
}


// --------------------------- HEAD functions ----------------------------
void ProgTomoDetectMisalignmentResiduals::detectMisalignmentFromResidualsMahalanobis()
{
	double sigma = fiducialSizePx / 3;	// Sigma for 99% of the points inside the fiducial radius
	double sigma2 = sigma * sigma;

	double mahaDist;
	double sumMahaDist = 0;
	double sumMahaDist2 = 0;

	double avgMahaDist;
	double stdMahaDist;

	size_t numberResMod = vResMod.size();

	// Calculate Mahalanobis distance for each residual
	// Set robust threshold so only those residuals < avg + 3*std are considered
	for(size_t i = 0; i < numberResMod; i++)
	{
		mahaDist = sqrt((vResMod[i].residuals.x*vResMod[i].residuals.x)/sigma2 + 
		                (vResMod[i].residuals.y*vResMod[i].residuals.y)/sigma2);
		
		vResMod[i].mahalanobisDistance = mahaDist;
		sumMahaDist += mahaDist;
		sumMahaDist2 += mahaDist * mahaDist; 
	}

	avgMahaDist = sumMahaDist / numberResMod;
	stdMahaDist = sqrt(sumMahaDist2 / numberResMod - avgMahaDist * avgMahaDist);
	double mahaThr = avgMahaDist + 3 * stdMahaDist;

	// Global alignment analysis
	std::cout << "---------------- Global misalignemnt analysis" << std::endl;

	double rationMisalignedChains = 0;	// 0 -> aligned -- 1 -> misaligned 

	for (size_t n = 0; n < numberOfInputCoords; n++)
	{
		std::vector<resMod> resMod_fid;
		getResModByFiducial(n, resMod_fid);

		size_t numberResMod = resMod_fid.size();

		sumMahaDist = 0;
		sumMahaDist2 = 0;

		for (size_t i = 0; i < numberResMod; i++)
		{
			if (removeOutliers && resMod_fid[i].mahalanobisDistance < mahaThr)
			{		
				sumMahaDist += resMod_fid[i].mahalanobisDistance;
				sumMahaDist2 += resMod_fid[i].mahalanobisDistance * resMod_fid[i].mahalanobisDistance;
			}
		}

		avgMahaDist = sumMahaDist / numberResMod;

		if (avgMahaDist > 1)
		{
			rationMisalignedChains += 1;
		}
		
		#ifdef DEBUG_RESIDUAL_ANALYSIS
		std::cout << "Average mahalanobis distance for 3D cooridinate " << n << ": " << avgMahaDist << std::endl;
		#endif
	}

	if (rationMisalignedChains / numberOfInputCoords > thrRatioMahalanobis)
	{
		globalAlignment = false;
	}

	std::cout << "------> Global alignment score: " << rationMisalignedChains / numberOfInputCoords << std::endl;
	
	// Local alignment analysis
	std::cout << "---------------- Local misaligment analysis" << std::endl;

	// Use votting as criteria for local misalignment detection
	if (voteCriteria)
	{
		for (size_t n = 0; n < nSize; n++)
		{
			std::vector<resMod> resMod_image;
			getResModByImage(n, resMod_image);

			size_t numberResMod = resMod_image.size();

			if (numberResMod > 1)
			{
				double rationMisalignedResid = 0.0;

				for (size_t i = 0; i < numberResMod; i++)
				{
					if (resMod_image[i].mahalanobisDistance > 1)
					{
						rationMisalignedResid += 1;
					}
				}

				rationMisalignedResid /= numberResMod;

				if (rationMisalignedResid > thrRatioMahalanobis)
				{
					localAlignment[n] = false;
					std::cout << "------> Local misalignment detected at image: " << n << " with ratio " << rationMisalignedResid << std::endl;
				}
			}

			else
			{
				std::cout << "ERROR: impossible to study misalignment in tilt-image " << n << ". Number of residuals for this image: " << numberResMod << std::endl;
				localAlignment[n] = false;
			}
		}
	}
	else // Use average as criteria for local misalignment detection
	{
		for (size_t n = 0; n < nSize; n++)
		{
			std::vector<resMod> resMod_image;
			getResModByImage(n, resMod_image);

			size_t numberResMod = resMod_image.size();

			if (numberResMod > 1)
			{
				sumMahaDist = 0;
				sumMahaDist2 = 0;

				for (size_t i = 0; i < numberResMod; i++)
				{
					double distance = resMod_image[i].mahalanobisDistance;
					
					if (removeOutliers &&  distance < mahaThr)
					{
						sumMahaDist += distance;
						sumMahaDist2 += distance * distance;
					}
				}

				avgMahaDist = sumMahaDist / numberResMod;
				double stdMahaDist = sqrt(sumMahaDist2 / numberResMod - avgMahaDist * avgMahaDist);

				avgMahalanobisDistanceV[n] = avgMahaDist;
				stdMahalanobisDistanceV[n] = stdMahaDist;

				if (avgMahaDist > 1)
				{
					localAlignment[n] = false;
					std::cout << "------> Local misalignment detected at image: " << n << std::endl;
				}
				
				#ifdef DEBUG_RESIDUAL_ANALYSIS
				std::cout << "Statistics of mahalanobis distances for tilt-image " << n << std::endl;
				std::cout << "Number of residual models: " << numberResMod << std::endl;
				std::cout << "Average mahalanobis distance: " << avgMahaDist << std::endl;
				std::cout << "STD mahalanobis distance: " << stdMahaDist << std::endl;
				#endif
			}

			else
			{
				std::cout << "ERROR: impossible to study misalignment in tilt-image " << n << ". Number of residuals for this image: " << numberResMod << std::endl;
				localAlignment[n] = false;
			}
		}
	}
}


void ProgTomoDetectMisalignmentResiduals::generateResidualStatiscticsFile()
{
	// Run XmippScript for statistical residual analysis
	std::cout << "Running residual statistical analysis..." << std::endl;

	// Compose commnad with realtive path to script
    char path[PATH_MAX], scriptPath[2048];
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    path[len] = '\0';  // Null-terminate

    char *pos = strstr(path, "/dist/");
    if (pos) *pos = '\0';  // Cut at "/dist/"

    snprintf(scriptPath, sizeof(scriptPath), "%s/src/xmipp/applications/scripts/tomo_misalignment_resid_statistics/batch_tomo_misalignment_resid_statistics.py", path);

	size_t lastindex = fnOut.find_last_of("\\/");
	std::string rawname = fnOut.substr(0, lastindex);

	std::string fnStats;
	fnStats = rawname + "/residualStatistics.xmd";
 
	std::string cmd;

	#ifdef DEBUG_RESIDUAL_STATISTICS_FILE
		cmd = "python3 " + std::string(scriptPath) + " -i " + fnResidualInfo + " -o " + fnStats + " --debug";
	#else
		cmd = "python3 " + std::string(scriptPath) + " -i " + fnResidualInfo + " -o " + fnStats;
	#endif

    std::cout << "Running command: " << cmd << std::endl;
	int systemOut = system(cmd.c_str());
	
	std::string fnStats_lm;
	std::string fnStats_image;

	fnStats_lm = rawname + "/residualStatistics_resid.xmd";
	fnStats_image = rawname + "/residualStatistics_image.xmd";

	std::string statistic;
	std::string statisticName;
	std::string residualNumber;
	int residualNumber_int;
	double value;

	// -- Save residual information --
	MetaDataVec residualStatsMd;
	residualStatsMd.read(fnStats_lm);

	// Vector containing stats: avg, std, chArea, chPerim, pvBinX, pvBinY, pvF, pvADF, ImageOutOfRange, LongestMisaliChain
	std::vector<double> residualStatsLine (10);
	std::vector<std::vector<double>> residualStatsTable;

	for (size_t i = 0; i < numberOfInputCoords; i++)
	{
		residualStatsTable.push_back(residualStatsLine);
	}

	// Save statistics in table
	for(size_t objId : residualStatsMd.ids())
	{
		residualStatsMd.getValue(MDL_IMAGE, statistic, objId);
		residualStatsMd.getValue(MDL_MIN, value, objId);

		statisticName = statistic.substr(statistic.find("_")+1);
		residualNumber = statistic.substr(0, statistic.find("_"));
		residualNumber_int = std::stoi(residualNumber);

		#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
		std::cout << "Reading object " << objId << " from metadata" << std::endl;
		std::cout << "statistic " << statistic << std::endl;
		std::cout << "value " << value << std::endl;
		std::cout << "statisticName " << statisticName << std::endl;
		std::cout << "residualNumber " << residualNumber << std::endl;
		std::cout << "residualNumber_int " << residualNumber_int << std::endl;
		#endif

		if (strcmp(statisticName.c_str(), "chArea") == 0)
		{
			residualStatsTable[residualNumber_int][2] = value;
		}

		else if (strcmp(statisticName.c_str(), "chPerim") == 0)
		{
			residualStatsTable[residualNumber_int][3] = value;
		}
		
		else if (strcmp(statisticName.c_str(), "pvBinX") == 0)
		{
			residualStatsTable[residualNumber_int][4] = value;
		}

		else if (strcmp(statisticName.c_str(), "pvBinY") == 0)
		{
			residualStatsTable[residualNumber_int][5] = value;
		}

		else if (strcmp(statisticName.c_str(), "pvF") == 0)
		{
			residualStatsTable[residualNumber_int][6] = value;
		}

		else if (strcmp(statisticName.c_str(), "pvADF") == 0)
		{
			residualStatsTable[residualNumber_int][7] = value;
		}
	}

	// Complete residual info
	double mod2Thr = (fiducialSizePx * thrRatioMahalanobis) * (fiducialSizePx * thrRatioMahalanobis);

	for (size_t n = 0; n < numberOfInputCoords; n++)
	{
		std::vector<resMod> resMod_fid;
		getResModByFiducial(n, resMod_fid);

		size_t numberResMod = resMod_fid.size();

		double avg;
		double std;
		size_t imagesOutOfRange = 0;
		double imagesOutOfRangePercentage;
		size_t longestMisaliChain = 0;
		double longestMisaliChainPercentage;
		size_t misaliChain = 0;

		double sumResid = 0;
		double sumResid2 = 0;

		for (size_t i = 0; i < numberResMod; i++)
		{
			double sum2 = resMod_fid[i].residuals.x*resMod_fid[i].residuals.x + resMod_fid[i].residuals.y*resMod_fid[i].residuals.y;
			sumResid2 += sum2;
			sumResid += sqrt(sum2);

			if (sum2 > mod2Thr)
			{
				imagesOutOfRange += 1;
				misaliChain += 1;
			}

			else if (misaliChain > longestMisaliChain)
			{
				longestMisaliChain = misaliChain;
				misaliChain = 0;
			}
		}

		imagesOutOfRangePercentage = double(imagesOutOfRange) / nSize;
		longestMisaliChainPercentage = double(longestMisaliChain) / nSize;
		avg = sumResid / numberResMod;
		std = sqrt(sumResid2 / numberResMod - avg * avg);

		#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
		std::cout << "n " << n << std::endl;
		std::cout << "numberResMod " << numberResMod << std::endl;
		std::cout << "sumResid " << sumResid << std::endl;
		std::cout << "sumResid2 " << sumResid2 << std::endl;
		std::cout << "imagesOutOfRange " << imagesOutOfRange << std::endl;
		std::cout << "longestMisaliChain " << longestMisaliChain << std::endl;
		std::cout << "imagesOutOfRangePercentage " << imagesOutOfRangePercentage << std::endl;
		std::cout << "longestMisaliChainPercentage " << longestMisaliChainPercentage << std::endl;		std::cout << "avg " << avg << std::endl;
		std::cout << "std " << std << std::endl;
		#endif

		residualStatsTable[n][0] = avg;
		residualStatsTable[n][1] = std;
		residualStatsTable[n][8] = imagesOutOfRangePercentage;
		residualStatsTable[n][9] = longestMisaliChainPercentage;
	}

	#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
	std::cout << " ----------------------------------------------- residualStatsTable" << std::endl;
	for (size_t n = 0; n < numberOfInputCoords; n++)
	{
		for (size_t i = 0; i < 10; i++)
		{
			std::cout << residualStatsTable[n][i] << " , ";
		}
		std::cout << "\n" ;
	}
	#endif

	// Save image information
	residualStatsMd.read(fnStats_image);

	// Vector containing stats: avg, std, chArea, chPerim, pvBinX, pvBinY, pvF, ResidOutOfRange
	std::vector<double> imageStatsLine (8);
	std::vector<std::vector<double>> imageStatsTable;

	for (size_t i = 0; i < nSize; i++)
	{
		imageStatsTable.push_back(imageStatsLine);
	}
	
	// Save statistics in table
	for(size_t objId : residualStatsMd.ids())
	{
		residualStatsMd.getValue(MDL_IMAGE, statistic, objId);
		residualStatsMd.getValue(MDL_MIN, value, objId);

		statisticName = statistic.substr(statistic.find("_")+1);
		residualNumber = statistic.substr(0, statistic.find("_"));
		residualNumber_int = std::stoi(residualNumber);

		if (strcmp(statisticName.c_str(), "chArea") == 0)
		{
			imageStatsTable[residualNumber_int][2] = value;
		}

		else if (strcmp(statisticName.c_str(), "chPerim") == 0)
		{
			imageStatsTable[residualNumber_int][3] = value;
		}
		
		else if (strcmp(statisticName.c_str(), "pvBinX") == 0)
		{
			imageStatsTable[residualNumber_int][4] = value;
		}

		else if (strcmp(statisticName.c_str(), "pvBinY") == 0)
		{
			imageStatsTable[residualNumber_int][5] = value;
		}

		else if (strcmp(statisticName.c_str(), "pvF") == 0)
		{
			imageStatsTable[residualNumber_int][6] = value;
		}
	}

	// Complete image info
	for (size_t n = 0; n < nSize; n++)
	{
		std::vector<resMod> resMod_image;
		getResModByImage(n, resMod_image);

		size_t numberResMod = resMod_image.size();

		if (numberResMod > 0)
		{
			double avg;
			double std;
			size_t residOutOfRange = 0;
			double residOutOfRangePercentage = 0;

			double sumResid = 0;
			double sumResid2 = 0;

			for (size_t i = 0; i < numberResMod; i++)
			{
				double sum = resMod_image[i].residuals.x*resMod_image[i].residuals.x + resMod_image[i].residuals.y*resMod_image[i].residuals.y;
				sumResid2 += sum;
				sumResid += sqrt(sum);

				if (sumResid > mod2Thr)
				{
					residOutOfRange += 1;
				}
			}

			residOutOfRangePercentage = double(residOutOfRange) / nSize;
			
			#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
			std::cout << "n " << n << std::endl;
			std::cout << "numberResMod " << numberResMod << std::endl;
			std::cout << "sumResid " << sumResid << std::endl;
			std::cout << "sumResid2 " << sumResid2 << std::endl;
			std::cout << "residOutOfRange " << residOutOfRange << std::endl;
			std::cout << "residOutOfRangePercentage " << residOutOfRangePercentage << std::endl;
			#endif

			avg = sumResid / numberResMod;
			std = sqrt(sumResid2 / numberResMod - avg * avg);

			#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
			std::cout << "avg " << avg << std::endl;
			std::cout << "std " << std << std::endl;
			#endif

			imageStatsTable[n][0] = avg;
			imageStatsTable[n][1] = std;
			imageStatsTable[n][7] = residOutOfRangePercentage;
		}
	}

	#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
	std::cout << " ----------------------------------------------- imageStatsTable" << std::endl;
	for (size_t n = 0; n < nSize; n++)
	{
		for (size_t i = 0; i < 8; i++)
		{
			std::cout << imageStatsTable[n][i] << " , ";
		}
		std::cout << "\n" ;
	}
	#endif	

	// Write residual statistics file both by chain and image
	std::string statsFileName_chain;
	std::string statsFileName_image;

	size_t li = fnOut.find_last_of("\\/");
	std::string fileBaseName = fnOut.substr(0, li);
	li = fileBaseName.find_last_of("\\/");
	fileBaseName = fileBaseName.substr(0, li);

	statsFileName_chain = fileBaseName + "/residStats_chain.txt";
	statsFileName_image = fileBaseName + "/residStats_image.txt";

	#ifdef DEBUG_RESIDUAL_STATISTICS_FILE	
	std::cout << "fileBaseName " << fileBaseName << std::endl;
	std::cout << "statsFileName_chain " << statsFileName_chain << std::endl;
	std::cout << "statsFileName_image " << statsFileName_image << std::endl;
	#endif
	
	std::ofstream myfile;

	// Write residual statistics by chain
	myfile.open (statsFileName_chain, std::ios_base::app);
	for (size_t n = 0; n < residualStatsTable.size(); n++)
	{
		myfile << residualStatsTable[n][0];
		myfile << ", ";
		myfile << residualStatsTable[n][1];
		myfile << ", ";
		myfile << residualStatsTable[n][2];
		myfile << ", ";
		myfile << residualStatsTable[n][3];
		myfile << ", ";
		myfile << residualStatsTable[n][4];
		myfile << ", ";
		myfile << residualStatsTable[n][5];
		myfile << ", ";
		myfile << residualStatsTable[n][6];
		myfile << ", ";
		myfile << residualStatsTable[n][7];
		myfile << ", ";
		myfile << residualStatsTable[n][8];
		myfile << ", ";
		myfile << residualStatsTable[n][9];
		myfile << "\n";
	}
	myfile.close();

	// Write residual statistics by image
	myfile.open (statsFileName_image, std::ios_base::app);
	for (size_t n = 0; n < imageStatsTable.size(); n++)
	{
		myfile << imageStatsTable[n][0];
		myfile << ", ";
		myfile << imageStatsTable[n][1];
		myfile << ", ";
		myfile << imageStatsTable[n][2];
		myfile << ", ";
		myfile << imageStatsTable[n][3];
		myfile << ", ";
		myfile << imageStatsTable[n][4];
		myfile << ", ";
		myfile << imageStatsTable[n][5];
		myfile << ", ";
		myfile << imageStatsTable[n][6];
		myfile << ", ";
		myfile << imageStatsTable[n][7];	
		myfile << "\n";
	}
	myfile.close();
}


// --------------------------- I/O functions ----------------------------
void ProgTomoDetectMisalignmentResiduals::readInputResiduals()
{
	#ifdef VERBOSE_OUTPUT
	std::cout << "Reading input residuals from " << fnResidualInfo  << std::endl;
	#endif

	MetaDataVec md;
	md.read(fnResidualInfo);

	size_t id;

	double lmX;
	double lmY;
	double lmZ;

	int coord3dx;
	int coord3dy;
	int coord3dz;

	double resX;
	double resY;

	size_t idLM;

	for(size_t id : md.ids())
	{
		md.getValue(MDL_X, lmX, id);
		md.getValue(MDL_Y, lmY, id);
		md.getValue(MDL_Z, lmZ, id);
		md.getValue(MDL_XCOOR, coord3dx, id);
		md.getValue(MDL_YCOOR, coord3dy, id);
		md.getValue(MDL_ZCOOR, coord3dz, id);
		md.getValue(MDL_SHIFT_X, resX, id);
		md.getValue(MDL_SHIFT_Y, resY, id);
		md.getValue(MDL_FRAME_ID, idLM, id);

		Point3D<double> lm(lmX, lmY, lmZ);
		Point3D<double> coord3d(coord3dx, coord3dy, coord3dz);
		Point2D<double> res(resX, resY);
		double mahalanobisDistance = 0;

		resMod rm {lm, coord3d, res, idLM, mahalanobisDistance};
		vResMod.push_back(rm);
	}

	#ifdef VERBOSE_OUTPUT
	std::cout << "Input residuals vectors read successfully!" << std::endl;
	#endif
}


void ProgTomoDetectMisalignmentResiduals::writeWeightedResiduals()
{
	auto outputResFilename = fnResidualInfo;
    size_t lastDotPos = outputResFilename.find_last_of('.');

    if (lastDotPos != std::string::npos) {
        outputResFilename.erase(lastDotPos);
    }

	outputResFilename += "_weighted.xmd";


	#ifdef VERBOSE_OUTPUT
	std::cout << "Writting weighted output residuals at " << outputResFilename  << std::endl;
	#endif

	MetaDataVec md;
	size_t id;

	for(size_t i = 0; i < vResMod.size(); i++)
	{
		id = md.addObject();
		md.setValue(MDL_X, vResMod[i].landmarkCoord.x, id);
		md.setValue(MDL_Y, vResMod[i].landmarkCoord.y, id);
		md.setValue(MDL_Z, vResMod[i].landmarkCoord.z, id);
		md.setValue(MDL_XCOOR, (int)vResMod[i].coordinate3d.x, id);
		md.setValue(MDL_YCOOR, (int)vResMod[i].coordinate3d.y, id);
		md.setValue(MDL_ZCOOR, (int)vResMod[i].coordinate3d.z, id);
		md.setValue(MDL_SHIFT_X, vResMod[i].residuals.x, id);
		md.setValue(MDL_SHIFT_Y, vResMod[i].residuals.y, id);
		md.setValue(MDL_FRAME_ID, vResMod[i].id, id);
		md.setValue(MDL_COST, vResMod[i].mahalanobisDistance, id);
	}

	md.write(outputResFilename);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Weighted residuals metadata saved successfully" << std::endl;
	#endif
}


void ProgTomoDetectMisalignmentResiduals::writeOutputAlignmentReport()
{
	MetaDataVec md;
	size_t id;


	if(!globalAlignment)
	{
		for(size_t i = 0; i < nSize; i++)
		{
			id = md.addObject();

			// Tilt-image			
			md.setValue(MDL_IDX, i, id);

			// Alignment
			md.setValue(MDL_ENABLED, -1, id);

			// Avg and STD of Mahalanobis distance
			md.setValue(MDL_COST, avgMahalanobisDistanceV[i], id);
			md.setValue(MDL_COST_PERCENTILE, stdMahalanobisDistanceV[i], id);
		}
	}
	else
	{
		for(size_t i = 0; i < localAlignment.size(); i++)
		{
			id = md.addObject();

			// Tilt-image			
			md.setValue(MDL_IDX, i, id);

			// Alignment
			if(localAlignment[i])
			{
				md.setValue(MDL_ENABLED, 1, id);
			}
			else
			{
				md.setValue(MDL_ENABLED, -1, id);
			}

			// Avg and STD of Mahalanobis distance
			md.setValue(MDL_COST, avgMahalanobisDistanceV[i], id);
			md.setValue(MDL_COST_PERCENTILE, stdMahalanobisDistanceV[i], id);		}
	}

	md.write(fnOut);
	
	#ifdef VERBOSE_OUTPUT
	std::cout << "Alignment report saved at: " << fnOut << std::endl;
	#endif
}


// --------------------------- MAIN ----------------------------------
void ProgTomoDetectMisalignmentResiduals::run()
{
	using std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;	

	auto t1 = steady_clock::now();

	std::cout << "Starting..." << std::endl;

	size_t Xdim, Ydim;

	generateSideInfo();

	detectMisalignmentFromResidualsMahalanobis();
		
	#ifdef GENERATE_RESIDUAL_STATISTICS
	generateResidualStatiscticsFile();
	#endif

	writeOutputAlignmentReport();
	writeWeightedResiduals();

	auto t2 = steady_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1); 	// Getting number of milliseconds as an integer
 	std::cout << "Execution time: " << ms_int.count() << "ms\n";
}


// --------------------------- UTILS functions ----------------------------
void ProgTomoDetectMisalignmentResiduals::getResModByFiducial(size_t fiducialNumber, std::vector<resMod> &vResMod_fiducial)
{
	for (size_t i = 0; i < vResMod.size(); i++)
	{
		if (vResMod[i].id == fiducialNumber)
		{
			vResMod_fiducial.push_back(vResMod[i]);
		}
	}
}


void ProgTomoDetectMisalignmentResiduals::getResModByImage(size_t tiltImageNumber, std::vector<resMod> &vResMod_image)
{
	for (size_t i = 0; i < vResMod.size(); i++)
	{
		if ((size_t)(vResMod[i].landmarkCoord.z) == tiltImageNumber)
		{
			vResMod_image.push_back(vResMod[i]);
		}
	}
}
