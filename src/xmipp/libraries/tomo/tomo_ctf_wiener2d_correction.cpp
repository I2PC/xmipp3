/***************************************************************************
 *
 * Authors:     Jose Luis Vilas (jlvilas@cnb.csic.es)
 *
 * Spanish Research Council for Biotechnology, Madrid, Spain
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

#include "tomo_ctf_wiener2d_correction.h"
#include <sys/stat.h>
#include "data/ctf.h"
#include <core/metadata_extension.h>
#include "reconstruction/ctf_correct_wiener2d.h"
#include "core/transformations.h"
#include <limits>
#include <type_traits>


void ProgCTFWiener2DCorrection::defineParams()
{
	addUsageLine("This method performs a CTF correction of a tilt series");
	addParamsLine("   -i <input_file>                    : Metadata with the tilt series");
	addParamsLine("   -o <output_volume>          	 	 : Output volume.");
	addParamsLine("   --sampling <sampling=1>            : Pixel size (Angstrom)");
	addParamsLine("   [--wiener_constant <wc=0.1>        : Wiener constant");
	addParamsLine("   [--threads <nthreads=1>]           : (Optional) Number of threads to be used");

	addExampleLine("Obtain the average of a set of subtomograms from a metadata file", false);
	addExampleLine("xmipp_tomo_average_subtomograms -i inputSubtomos.xmd -o average.mrc");
}

void ProgCTFWiener2DCorrection::readParams()
{
	fnIn = getParam("-i");
	fnOut = getParam("-o");
	sampling = getDoubleParam("--sampling");
	wc = getDoubleParam("--wiener_constant");
	sigmaDf = getDoubleParam("--sigmaDf");
	nthreads = getIntParam("--threads");
}

void ProgCTFWiener2DCorrection::gaussianMask(MultidimArray<double> &cumMask, 
											MultidimArray<double> &tiMask,  
											MultidimArray<double> &ptrImg, int x0, int stripeSize)
{
	auto xdim = XSIZE(ptrImg);
	auto ydim = XSIZE(ptrImg);

	// To set sigma: We assume that the gaussian g(stripeSize) = 0.1
	// 0.1 = exp(-(stripeSize)^2/(2*sigma2)),  - log 10 = - (stripeSize)^2/(2*sigma2)
	// therefore sigma2 = stripeSize^2/(2*log 10)
	double sigma2 = (double) stripeSize*stripeSize/(log(100.0));

	long n = 0;
	for (size_t i = 0; i<ydim; i++)
	{
		for (size_t j = 0; j<xdim; j++)
		{
			double p = (j-x0)^2;
			double g;
			g = exp(-(p*p)/(2*sigma2));
			
			DIRECT_MULTIDIM_ELEM(ptrImg, n) *= g;
			DIRECT_MULTIDIM_ELEM(cumMask, n) += g;
			n++;
		}
	}
}


void ProgCTFWiener2DCorrection::run()
{
	std::cout << "Starting ... " << std::endl;

	MetaDataVec mdTs, mdOut;
	FileName fnImg;

	double expDfU, expDfV, expDF;
	double tilt;

	Image<double> tiImg;
	auto &ti = tiImg();

	int stripeWidth = round(sigmaDf/sampling);
	int halfWidth = round(stripeWidth*0.5);

	mdTs.read(fnIn);

	size_t idx=0;
	for (const auto& row : mdTs)
	{
		row.getValue(MDL_IMAGE, fnImg);
		row.getValue(MDL_ANGLE_TILT, tilt);
		row.getValue(MDL_CTF_DEFOCUSU, expDfU);
		row.getValue(MDL_CTF_DEFOCUSV, expDfV);
		
		expDF = 0.5*(expDfU+expDfV);

		tiImg.read(fnImg);
		Image<double> imgCor;
		auto &ptrImgCor = imgCor();

		ptrImgCor.initZeros(ti);

		size_t Xdim, Ydim, Zdim, Ndim;
		tiImg.getDimensions(Xdim, Ydim, Zdim, Ndim);

		// The size (in px) of the stripes will be the accuracy over the sampling
		int stripeSize = floor(sigmaDf/sampling);
		int halfStripe = stripeSize/2;
		int nstripes = 2*(Xdim - halfStripe)/stripeSize + 1;
		auto imgCenter = Xdim/2;

		// The indices of the beginning of the stripes are goign to be computed.
		// There is always a stripe at the center of the image.
		std::vector<int> idxVec;
		for (int s=0; s<nstripes; s +=stripeSize)
		{
			auto pos_p = imgCenter + halfStripe*s;
			auto pos_m = imgCenter - halfStripe*s;
			if (pos_p>Xdim && pos_m<0)
			{
				idxVec.push_back(pos_p);
				idxVec.push_back(pos_m);
				// Commented for possible debugging
				// std::cout << "imgCenter + halfStripe*s = " << imgCenter + halfStripe*s << std::endl;
				// std::cout << "imgCenter - halfStripe*s = " << imgCenter - halfStripe*s << std::endl;
			}
		}

		CTFDescription ctf;
		MultidimArray<double> cumMask;
		cumMask.initZeros(ptrImgCor);

		ctf.readFromMdRow(row);
		ctf.phase_shift = (ctf.phase_shift*PI)/180;

		for (size_t s=0; s<idxVec.size(); s++)
		{
			int x0 = idxVec[s];
			double df = sampling*x0*tan(tilt*PI/180);
			ctf.DeltafU = expDfU + df;
			ctf.DeltafV = expDfV + df;

			auto ptrImg = tiImg();
			Wiener2D WF;
			WF.pad = 1.0;
			WF.correct_envelope = false;
			WF.sampling_rate = sampling;
			WF.wiener_constant = wc;
			WF.isIsotropic = true;
			WF.phase_flipped = false;
			WF.applyWienerFilter(ptrImg, ctf);

			MultidimArray<double> cumMask, tiMask;
			tiMask.initZeros(ptrImg);
			gaussianMask(cumMask, tiMask, ptrImg, x0, stripeSize);

			ptrImgCor+=ptrImg;
		}

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(ptrImgCor)
		{
			DIRECT_MULTIDIM_ELEM(ptrImgCor, n) /= DIRECT_MULTIDIM_ELEM(cumMask, n);
		}

		MDRowVec rowOut;
		rowOut = row;
		auto fnBase = fnImg.removeLastExtension();
		Image<double> saveImg;
		saveImg() = ptrImgCor;
		saveImg.write(fnOut+"/"+fnBase + ".mrcs", idx+FIRST_IMAGE, true, WRITE_APPEND);

		FileName composedFn;
		auto fn = fnBase + ".mrcs";
		composedFn.compose(idx, fn);
		rowOut.setValue(MDL_IMAGE, composedFn);
		mdOut.addRow(rowOut);


	}
	
	mdOut.write(fnOut);

}

