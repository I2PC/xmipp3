/***************************************************************************
 *
 * Authors:    Javier Vargas   (jvargas@cnb.csic.es)
 * Authors:    Jose Luis Vilas (jlvilas@cnb.csic.es)
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

#ifndef _WIENER2D
#define _WIENER2D

#include "core/xmipp_metadata_program.h"
#include "data/ctf.h"
#include "core/xmipp_image.h"
#include "data/filters.h"

//template<typename T>
class Wiener2D
{
	public:
		bool phase_flipped;

		/** Padding factor */
		double pad;

		bool isIsotropic;

		bool correct_envelope;

		/// Wiener filter constant
		double wiener_constant;

		/// Sampling rate
		double sampling_rate;

	public:
		void applyWienerFilter(MultidimArray<double> &ptrImg, CTFDescription &ctf);

		void applyWienerFilter(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn, MDRow &rowOut);

   		void wienerFilter(MultidimArray<double> &Mwien, CTFDescription &ctf);

	public:
		Image<double> img;

		CTFDescription ctf;

		size_t Ydim, Xdim;

		MultidimArray<double> Mwien;
		MultidimArray<std::complex<double> > Faux;
		FourierTransformer transformer;
};
#endif