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

#ifndef LIBRARIES_RECONSTRUCTION_CUDA_FIND_EXTREMA_H_
#define LIBRARIES_RECONSTRUCTION_CUDA_FIND_EXTREMA_H_

#include "reconstruction_cuda/gpu.h"
#include <limits>
#include <thread>
#include <condition_variable>
#include <core/utils/memory_utils.h>
#include "reconstruction/aextrema_finder.h"

/**@defgroup CudaExtremaFinder CudaExtremaFinder
   @ingroup ReconsLibrary */
//@{
namespace ExtremaFinder {

template<typename T>
class CudaExtremaFinder : public AExtremaFinder<T> {
public:
    // FIXME DS add support for min
    CudaExtremaFinder() {
        setDefault();
    }

    virtual ~CudaExtremaFinder() {
        release();
    }

    CudaExtremaFinder(CudaExtremaFinder& o) = delete;
    CudaExtremaFinder& operator=(const CudaExtremaFinder& other) = delete;
    CudaExtremaFinder const & operator=(CudaExtremaFinder &&o) = delete;
    CudaExtremaFinder(CudaExtremaFinder&& o) noexcept {
        m_loadStream = o.m_loadStream;
        m_workStream = o.m_workStream;

        // device memory
        m_d_values = o.m_d_values;
        m_d_positions = o.m_d_positions;
        m_d_batch = o.m_d_batch;

        // synch primitives
        m_mutex = o.m_mutex;
        m_cv = o.m_cv;
        m_isDataReady = o.m_isDataReady;

        // host memory
        m_h_batchResult = o.m_h_batchResult;

        // clean original
        o.setDefault();
    }

    template<typename C>
    static void sFindUniversal(
        const C &comp,
        T startVal,
        const GPU &gpu,
        const Dimensions &dims,
        const T *d_data,
        float *d_positions,
        T *d_values);

    static void sFindMax(const GPU &gpu,
        const Dimensions &dims,
        const T *d_data,
        float *d_positions,
        T *d_values);

    static void sFindLowest(const GPU &gpu,
        const Dimensions &dims,
        const T *d_data,
        float *d_positions,
        T *d_values);

    template<typename C>
    static void sFindUniversal2DAroundCenter(
        const C &comp,
        T startVal,
        const GPU &gpu,
        const Dimensions &dims,
        const T *data,
        float *d_positions, // can be nullptr
        T * d_values, // can be nullptr
        size_t maxDist);

    static void sFindMax2DAroundCenter(const GPU &gpu,
        const Dimensions &dims,
        const T *d_data,
        float *d_positions, // can be nullptr
        T * d_values, // can be nullptr
        size_t maxDist);

    static void sFindLowest2DAroundCenter(const GPU &gpu,
        const Dimensions &dims,
        const T *d_data,
        float *d_positions, // can be nullptr
        T * d_values, // can be nullptr
        size_t maxDist);

    static void sRefineLocation(const GPU &gpu,
        const Dimensions &dims, // of the data
        const float * d_indices, // array holding indices to position of the extrema
        float * d_positions, // ND array of refined position (in X-Y-Z order)
        const T * d_data); // where the extrema is located

    static size_t ceilPow2(size_t x); // FIXME DS move this to somewhere else

private:
    GPU *m_loadStream;
    GPU *m_workStream;

    // device memory
    T *m_d_values;
    float *m_d_positions;
    T *m_d_batch;

    // synch primitives
    std::mutex *m_mutex;
    std::condition_variable *m_cv;
    bool m_isDataReady;

    // host memory
    T *m_h_batchResult;

    void setDefault();
    void release();

    void check() const override;

    void initMax() override;
    void findMax(const T *h_data) override;
    bool canBeReusedMax(const ExtremaFinderSettings &s) const override;

    void initLowest() override;
    void findLowest(const T *h_data) override;
    bool canBeReusedLowest(const ExtremaFinderSettings &s) const override;

    void initMaxAroundCenter() override;
    void findMaxAroundCenter(const T *h_data) override;
    bool canBeReusedMaxAroundCenter(const ExtremaFinderSettings &s) const override;

    void initLowestAroundCenter() override;
    void findLowestAroundCenter(const T *h_data) override;
    bool canBeReusedLowestAroundCenter(const ExtremaFinderSettings &s) const override;

    void loadThreadRoutine(const T *h_data);
    void downloadPositionsFromGPU(size_t offset, size_t count);
    void downloadValuesFromGPU(size_t offset, size_t count);

    void initBasic();
    template<typename KERNEL>
    void findBasic(const T *h_data, const KERNEL &k);
    bool canBeReusedBasic(const ExtremaFinderSettings &s) const;
};

} /* namespace ExtremaFinder */
//@}
#endif /* LIBRARIES_RECONSTRUCTION_CUDA_FIND_EXTREMA_H_ */
