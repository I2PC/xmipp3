/***************************************************************************
 * Authors:     Carlos Oscar Sorzano (coss@cnb.csic.es)
 *
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

#include "ctf_correct_phase.h"

// Read arguments ==========================================================
void ProgCorrectPhaseFlip2D::readParams()
{
    XmippMetadataProgram::readParams();
    sampling_rate = getDoubleParam("--sampling_rate");
}

// Define parameters ==========================================================
void ProgCorrectPhaseFlip2D::defineParams()
{
    addUsageLine("Perform CTF correction to 2D projection images with estimated ctfs using a phase flip filter.");
    each_image_produces_an_output = true;
    XmippMetadataProgram::defineParams();
    addKeywords("correct CTF by phase flipping");
    addParamsLine("   [--sampling_rate <float=1.0>]     : Sampling rate of the input particles");
}

void ProgCorrectPhaseFlip2D::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut)
{
	rowOut = rowIn;
	img.read(fnImg);
	ctf.readFromMdRow(rowIn);
    ctf.produceSideInfo();
	img().setXmippOrigin();
	ctf.correctPhase(img(),sampling_rate);
    img.write(fnImgOut);
    rowOut.setValue(MDL_IMAGE, fnImgOut);
}
