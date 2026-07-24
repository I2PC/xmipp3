/***************************************************************************
 *
 * Authors:    David Strelak (davidstrelak@gmail.com)
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

#ifndef LIBRARIES_RECONSTRUCTION_PSD_ESTIMATOR_H_
#define LIBRARIES_RECONSTRUCTION_PSD_ESTIMATOR_H_

#include "core/multidim_array.h"
#include "data/fft_settings.h"
#include "data/rectangle.h"

class Dimensions;


/**@defgroup PSDEstimator PSD Estimator
   @ingroup ReconsLibrary */
//@{
template<typename T>
class PSDEstimator {
public:
    static std::vector<Rectangle<Point2D<size_t>>> getPatchesLocation(
            const std::pair<size_t, size_t> &borders,
            const Dimensions &micrograph,
            const Dimensions &patch,
            float overlap);

    static void estimatePSD(const MultidimArray<T> &micrograph,
            float overlap, const Dimensions &tileDim, MultidimArray<T> &psd,
            unsigned fftThreads, bool normalize);

    template<typename F>
    static void half2whole(const T *in,
            T* __restrict__ out,
            const FFTSettings<T> &settings, F func) {
        for (size_t y = 0; y < settings.sDim().y(); ++y) {
            for (size_t x = 0; x < settings.sDim().x(); ++x) {
                bool mirror = x >= settings.fDim().x();
                size_t xS = mirror
                        ? (settings.sDim().x() - x)
                        : x;
                size_t yS = mirror
                        ? ((y == 0) ? 0 : (settings.sDim().y() - y))
                        : y;

                assert(xS < settings.fDim().x());
                assert(yS < settings.fDim().y());
                size_t indexD = y * settings.sDim().x() + x;
                size_t indexS = yS * settings.fDim().x() + xS;
                out[indexD] = func(mirror, in[indexS]);
            }
        }
    }


};
//@}
#endif /* LIBRARIES_RECONSTRUCTION_PSD_ESTIMATOR_H_ */
