/***************************************************************************
 *
 * Authors:    Javier Vargas   (jvargas@cnb.csic.es)
 * Authors:    Jose Luis Vilas (jlvilas@cnb.csic.es)
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

#include "wiener2d.h"


void Wiener2D::wienerFilter(MultidimArray<double> &Mwien, CTFDescription & ctf)
{
	int paddimY;
	paddimY = Ydim*pad;
	int paddimX = Xdim*pad;
	ctf.enable_CTF = true;
	ctf.enable_CTFnoise = false;
	ctf.produceSideInfo();

	MultidimArray<std::complex<double>> ctfComplex;
	MultidimArray<double> ctfIm;

	Mwien.resize(paddimY,paddimX);

	ctf.Tm = sampling_rate;

	if (isIsotropic)
	{
		double avgdef = (ctf.DeltafU + ctf.DeltafV)/2.;
		ctf.DeltafU = avgdef;
		ctf.DeltafV = avgdef;
	}

	ctfIm.resize(1, 1, paddimY, paddimX,false);
	//Esto puede estar mal. Cuidado con el sampling de la ctf!!!
	if (correct_envelope)
		ctf.generateCTF(paddimY, paddimX, ctfComplex);
	else
		ctf.generateCTFWithoutDamping(paddimY, paddimX, ctfComplex);

	if (phase_flipped)
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(ctfIm)
			dAij(ctfIm, i, j) = fabs(dAij(ctfComplex, i, j).real());
	}
	else
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(ctfIm)
			dAij(ctfIm, i, j) = dAij(ctfComplex, i, j).real();
	}

	double result;
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(Mwien)
	{
		result = (DIRECT_N_YX_ELEM (ctfIm, 0, i, j));
		dAij(Mwien,i,j) = (result *result);
	}

	// Add Wiener constant
	if (wiener_constant < 0.)
	{

		// Use Grigorieff's default for Wiener filter constant: 10% of average over all Mwien terms
		// Grigorieff JSB 157(1) (2006), pp 117-125
		double valueW = 0.1*Mwien.computeAvg();
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(Mwien)
		{
			dAij(Mwien,i,j) += valueW;
			dAij(Mwien,i,j) = dAij(ctfIm, i, j)/dAij(Mwien, i, j);
		}
	}
	else
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(Mwien)
		{
			dAij(Mwien,i,j) += wiener_constant;
			dAij(Mwien,i,j) = dAij(ctfIm, i, j)/dAij(Mwien, i, j);
		}
	}
}

void Wiener2D::applyWienerFilter(MultidimArray<double> &ptrImg, CTFDescription &ctf)
{
	ptrImg.setXmippOrigin();
	Ydim = YSIZE(ptrImg);
	Xdim = XSIZE(ptrImg);
	int paddimY = Ydim*pad;
	int paddimX = Xdim*pad;

	wienerFilter(Mwien, ctf);

    if (paddimX >= Xdim)
    {
        // pad real-space image
        int x0 = FIRST_XMIPP_INDEX(paddimX);
        int xF = LAST_XMIPP_INDEX(paddimX);
        int y0 = FIRST_XMIPP_INDEX(paddimY);
        int yF = LAST_XMIPP_INDEX(paddimY);
        ptrImg.selfWindow(y0, x0, yF, xF);
    }

	MultidimArray<std::complex<double> > Faux;
    transformer.FourierTransform(ptrImg, Faux);
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(Faux)
    {
        dAij(Faux,i,j) *= dAij(Mwien,i,j);
    }

    transformer.inverseFourierTransform(Faux, ptrImg);


	if (paddimX >= Xdim)
    {
        // de-pad real-space image
        int x0 = FIRST_XMIPP_INDEX(Xdim);
        int y0 = FIRST_XMIPP_INDEX(Ydim);
        int xF = LAST_XMIPP_INDEX(Xdim);
        int yF = LAST_XMIPP_INDEX(Ydim);
        ptrImg.selfWindow(y0, x0, yF, xF);
    }
}


void Wiener2D::applyWienerFilter(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
	rowOut = rowIn;

	img.read(fnImg);
	ctf.readFromMdRow(rowIn);
	ctf.phase_shift = (ctf.phase_shift*PI)/180;

	MultidimArray<double> &ptrImg =img();
	applyWienerFilter(ptrImg, ctf);

    img.write(fnImgOut);
    rowOut.setValue(MDL_IMAGE, fnImgOut);
}