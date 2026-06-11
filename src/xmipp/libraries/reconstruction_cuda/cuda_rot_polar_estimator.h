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

#ifndef LIBRARIES_RECONSTRUCTION_CUDA_CUDA_ROT_POLAR_ESTIMATOR_H_
#define LIBRARIES_RECONSTRUCTION_CUDA_CUDA_ROT_POLAR_ESTIMATOR_H_

#include "reconstruction/arotation_estimator.cpp"
#include "reconstruction_cuda/gpu.h"
#include "cuda_fft.h"
#include <thread>
#include <condition_variable>
#include <algorithm>

#include "cuda_single_extrema_finder.h"

/**@defgroup CudaRotPolarEstimator Cuda Rotation Polar Estimator
   @ingroup ReconsLibrary */
//@{
namespace Alignment {

template<typename T>
class CudaRotPolarEstimator : public ARotationEstimator<T> {
public :
    CudaRotPolarEstimator() {
        setDefault();
    }

    virtual ~CudaRotPolarEstimator() {
        release();
    }

    CudaRotPolarEstimator(CudaRotPolarEstimator& o) = delete;
    CudaRotPolarEstimator& operator=(const CudaRotPolarEstimator& other) = delete;
    CudaRotPolarEstimator(CudaRotPolarEstimator&& o) noexcept {
        m_mainStream = o.m_mainStream;
        m_backgroundStream = o.m_backgroundStream;
        m_samples = o.m_samples;

        // device memory
        m_d_ref = o.m_d_ref;
        m_d_batch = o.m_d_batch;
        m_d_batchPolarOrCorr = o.m_d_batchPolarOrCorr;
        m_d_batchPolarFD = o.m_d_batchPolarFD;
        m_d_sumsOrMaxPos = o.m_d_sumsOrMaxPos;
        m_d_sumsSqr = o.m_d_sumsSqr;
        m_d_batchCorrSumFD = o.m_d_batchCorrSumFD;

        // FT plans
        m_singleToFD = o.m_singleToFD;
        m_batchToFD = o.m_batchToFD;
        m_batchToSD = o.m_batchToSD;

        // synch primitives
        m_mutex = o.m_mutex;
        m_cv = o.m_cv;
        m_isDataReady = o.m_isDataReady;

        // host memory
        m_h_batchMaxPositions = o.m_h_batchMaxPositions;

        // remove data from other
        o.setDefault();
    }
    CudaRotPolarEstimator const & operator=(CudaRotPolarEstimator &&o) = delete;

    static void sComputeCorrelationsOneToN(
            const GPU &gpu,
            const std::complex<T> *d_in,
            std::complex<T> *d_out,
            const std::complex<T> *d_ref,
            const Dimensions &dims,
            int firstRingRadius);

    template<bool FULL_CIRCLE>
    static void sComputePolarTransform(
            const GPU &gpu,
            const Dimensions &dimIn,
            T * d_in,
            const Dimensions &dimOut,
            T * d_out,
            int posOfFirstRing);

    template<bool FULL_CIRCLE>
    static void sNormalize(
            const GPU &gpu,
            const Dimensions &dimIn,
            T *d_in,
            T *d_1,
            T *d_2,
            int posOfFirstRing);

private:
    GPU *m_backgroundStream;
    GPU *m_mainStream;
    int m_samples;

    // device memory
    std::complex<T> *m_d_ref;
    T *m_d_batch;
    T *m_d_batchPolarOrCorr;
    std::complex<T> *m_d_batchPolarFD;
    std::complex<T> *m_d_batchCorrSumFD;
    T *m_d_sumsOrMaxPos;
    T *m_d_sumsSqr;

    // FT plans
    cufftHandle *m_singleToFD;
    cufftHandle *m_batchToFD;
    cufftHandle *m_batchToSD;

    // synch primitives
    std::mutex *m_mutex;
    std::condition_variable *m_cv;
    bool m_isDataReady;

    // host memory
    float *m_h_batchMaxPositions;

    void check() override;
    void setDefault();
    void release();

    void init2D() override;
    bool canBeReused2D(const RotationEstimationSetting &s) const override {
        return false; // FIXME DS implement
    }

    template<bool FULL_CIRCLE>
    void load2DReferenceOneToN(const T *h_ref);
    void load2DReferenceOneToN(const T *h_ref) override;

    template<bool FULL_CIRCLE>
    void computeRotation2DOneToN(T *others);
    void computeRotation2DOneToN(T *others) override;

    void loadThreadRoutine(T *h_others);

    template<bool FULL_CIRCLE>
    void waitAndConvert(
            const Dimensions &inCart,
            const Dimensions &outPolar,
            unsigned firstRing);
};

}/* namespace Alignment */
//@}
#endif /* LIBRARIES_RECONSTRUCTION_CUDA_CUDA_ROT_POLAR_ESTIMATOR_H_ */
