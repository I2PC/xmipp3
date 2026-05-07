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

#include <cuda_runtime_api.h>
#include "reconstruction_cuda/cuda_asserts.h"
#include "cuda_shift_corr_estimator.h"
#include "reconstruction_cuda/cuda_gpu_movie_alignment_correlation_kernels.cu"

namespace Alignment {

template<typename T>
void CudaShiftCorrEstimator<T>::init2D(const std::vector<HW*> &hw, AlignType type,
        const FFTSettings<T> &settings, size_t maxShift,
        bool includingBatchFT, bool includingSingleFT,
        bool allowDataOvewrite) {
    // FIXME DS consider tunning the size of the input (e.g. 436x436x50)
    if (2 != hw.size()) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Two GPU streams are needed");
    }
    release(); // FIXME DS implement lazy init
    m_workStream = dynamic_cast<GPU*>(hw.at(0));
    m_loadStream = dynamic_cast<GPU*>(hw.at(1));
    if ((nullptr == m_workStream)
        || (nullptr == m_loadStream)) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Instance of GPU is expected");
    }
    m_workStream->set();
    m_loadStream->set();
    AShiftCorrEstimator<T>::init2D(type, settings, maxShift,
        includingBatchFT, includingSingleFT, allowDataOvewrite);

    // synch primitives
    m_mutex = new std::mutex();
    m_cv = new std::condition_variable();

    this->m_isInit = true;
}

template<typename T>
void CudaShiftCorrEstimator<T>::setDefault() {
    AShiftCorrEstimator<T>::setDefault();

    m_workStream = nullptr;
    m_loadStream = nullptr;

    // device memory
    m_d_single_FD = nullptr;
    m_d_batch_FD = nullptr;
    m_d_single_SD = nullptr;
    m_d_batch_SD_work = nullptr;
    m_d_batch_SD_load = nullptr;

    // host memory
    m_h_centers = nullptr;

    // FT plans
    m_singleToFD = nullptr;
    m_batchToFD = nullptr;
    m_batchToSD = nullptr;

    // synch primitives
    m_mutex = nullptr;
    m_cv = nullptr;
    m_isDataReady = false;
}

template<typename T>
void CudaShiftCorrEstimator<T>::load2DReferenceOneToN(const std::complex<T> *h_ref) {
    auto isReady = (this->m_isInit && (AlignType::OneToN == this->m_type));
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to load a reference signal");
    }

    m_loadStream->set();
    auto loadStream = *(cudaStream_t*)m_loadStream->stream();
    // copy reference to GPU
    gpuErrchk(cudaMemcpyAsync(m_d_single_FD, h_ref, this->m_settingsInv->fBytesSingle(),
            cudaMemcpyHostToDevice, loadStream));

    // update state
    m_is_d_single_FD_loaded = true;
}

template<typename T>
void CudaShiftCorrEstimator<T>::load2DReferenceOneToN(const T *h_ref) {
    auto isReady = (this->m_isInit && (AlignType::OneToN == this->m_type) && this->m_includingSingleFT);
    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to load a reference signal");
    }
    m_loadStream->set();
    m_workStream->set();
    auto loadStream = *(cudaStream_t*)m_loadStream->stream();
    auto workStream = *(cudaStream_t*)m_workStream->stream();

    // copy reference to GPU
    gpuErrchk(cudaMemcpyAsync(m_d_single_SD, h_ref, this->m_settingsInv->sBytesSingle(),
            cudaMemcpyHostToDevice, loadStream));
    m_loadStream->synch();

    // perform FT
    CudaFFT<T>::fft(*m_singleToFD, m_d_single_SD, m_d_single_FD);

    // update state
    m_is_d_single_FD_loaded = true;
}

template<typename T>
void CudaShiftCorrEstimator<T>::release() {
    // device memory
    gpuErrchk(cudaFree(m_d_single_FD));
    gpuErrchk(cudaFree(m_d_batch_FD));
    gpuErrchk(cudaFree(m_d_single_SD));
    gpuErrchk(cudaFree(m_d_batch_SD_work));
    if ( ! this->m_allowDataOverwrite) {
        gpuErrchk(cudaFree(m_d_batch_SD_load));
    }

    // host memory
    delete[] m_h_centers;

    // FT plans
    CudaFFT<T>::release(m_singleToFD);
    CudaFFT<T>::release(m_batchToFD);
    CudaFFT<T>::release(m_batchToSD);

    // synch primitives
    delete m_mutex;
    delete m_cv;

    AShiftCorrEstimator<T>::release();

    CudaShiftCorrEstimator<T>::setDefault();
}

template<typename T>
void CudaShiftCorrEstimator<T>::init2DOneToN() {
    AShiftCorrEstimator<T>::init2DOneToN();
    // allocate space for data in Fourier domain
    gpuErrchk(cudaMalloc(&m_d_single_FD, this->m_settingsInv->fBytesSingle()));
    gpuErrchk(cudaMalloc(&m_d_batch_FD, this->m_settingsInv->fBytesBatch()));

    // allocate host memory
    m_h_centers = new T[this->m_centerSize * this->m_centerSize * this->m_batch]();

    // allocate plans and additional space needed
    m_batchToSD = CudaFFT<T>::createPlan(*m_workStream, *this->m_settingsInv);
    auto settingsForw = this->m_settingsInv->createInverse();
    if (this->m_includingBatchFT) {
        gpuErrchk(cudaMalloc(&m_d_batch_SD_work, this->m_settingsInv->sBytesBatch()));
        if ( ! this->m_allowDataOverwrite) {
            gpuErrchk(cudaMalloc(&m_d_batch_SD_load, this->m_settingsInv->sBytesBatch()));
        }
        m_batchToFD = CudaFFT<T>::createPlan(*m_workStream, settingsForw);
    }
    if (this->m_includingSingleFT) {
        gpuErrchk(cudaMalloc(&m_d_single_SD, this->m_settingsInv->sBytesSingle()));
        m_singleToFD = CudaFFT<T>::createPlan(*m_workStream, settingsForw.createSingle());
    }
}

template<typename T>
void CudaShiftCorrEstimator<T>::computeCorrelations2DOneToN(
        std::complex<T> *h_inOut, bool center) {
    bool isReady = (this->m_isInit && (AlignType::OneToN == this->m_type) && m_is_d_single_FD_loaded);

    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to execute. Call init() before");
    }

    m_loadStream->set();
    m_workStream->set();
    auto loadStream = *(cudaStream_t*)m_loadStream->stream();
    auto workStream = *(cudaStream_t*)m_workStream->stream();
    // process signals in batches
    for (size_t offset = 0; offset < this->m_settingsInv->fDim().n(); offset += this->m_settingsInv->batch()) {
        // how many signals to process
        size_t toProcess = std::min(this->m_settingsInv->batch(), this->m_settingsInv->fDim().n() - offset);

        // copy memory
        gpuErrchk(cudaMemcpyAsync(
                m_d_batch_FD,
                h_inOut + offset * this->m_settingsInv->fDim().xy(),
                toProcess * this->m_settingsInv->fBytesSingle(),
                cudaMemcpyHostToDevice, loadStream));
        m_loadStream->synch();

        auto dims = Dimensions(
                this->m_settingsInv->fDim().x(),
                this->m_settingsInv->fDim().y(),
                1,
                toProcess);
        if (center) {
            sComputeCorrelations2DOneToN<true>(*m_workStream, m_d_batch_FD, m_d_single_FD, dims);
        } else {
            sComputeCorrelations2DOneToN<false>(*m_workStream, m_d_batch_FD, m_d_single_FD, dims);
        }
        m_workStream->synch();

        // copy data back
        gpuErrchk(cudaMemcpyAsync(
                h_inOut + offset * this->m_settingsInv->fDim().xy(),
                m_d_batch_FD,
                toProcess * this->m_settingsInv->fBytesSingle(),
                cudaMemcpyDeviceToHost, loadStream));
    }
}

template<typename T>
void CudaShiftCorrEstimator<T>::loadThreadRoutine(T *others) {
    m_loadStream->set();
    auto lStream = *(cudaStream_t*)m_loadStream->stream();
    for (size_t offset = 0; offset < this->m_settingsInv->fDim().n(); offset += this->m_settingsInv->batch()) {
        // how many signals to process
        size_t toProcess = std::min(this->m_settingsInv->batch(), this->m_settingsInv->fDim().n() - offset);
        std::unique_lock<std::mutex> lk(*m_mutex);
        // wait till the data is processed
        m_cv->wait(lk, [this] {
            return !m_isDataReady;
        });
        T *src = others + offset * this->m_settingsInv->sDim().xy();
        size_t bytes = toProcess * this->m_settingsInv->sBytesSingle();

        auto kind = m_loadStream->isGpuPointer(src)
                ? cudaMemcpyDeviceToDevice
                : cudaMemcpyHostToDevice;
        // copy memory
        gpuErrchk(cudaMemcpyAsync(
               m_d_batch_SD_load,
               src,
               bytes,
               kind, lStream));
        // block until data is loaded
        m_loadStream->synch();

        // notify processing stream that it can work
        m_isDataReady = true;
        m_cv->notify_one();
    }
}

template<typename T>
void CudaShiftCorrEstimator<T>::waitAndConvert() {
    // block until data is loaded
    // mutex will be freed once leaving this block
    std::unique_lock<std::mutex> lk(*m_mutex);
    while(!m_isDataReady) {
        m_cv->wait(lk);
    }
    // perform FT
    CudaFFT<T>::fft(*m_batchToFD, m_d_batch_SD_load, m_d_batch_FD);

    // notify that buffer is processed (new will be loaded in background)
    m_workStream->synch();
    m_isDataReady = false;
    m_cv->notify_one();
}

template<typename T>
void CudaShiftCorrEstimator<T>::computeShift2DOneToN(
        T *others) {
    bool isReady = (this->m_isInit && (AlignType::OneToN == this->m_type)
            && m_is_d_single_FD_loaded && this->m_includingBatchFT);

    if ( ! isReady) {
        REPORT_ERROR(ERR_LOGIC_ERROR, "Not ready to execute. Call init() before");
    }
    const auto isGpuPtr = m_workStream->isGpuPointer(others);
    if(!isGpuPtr) {
        if ( ! GPU::isMemoryPinned(others)) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "Input memory has to be pinned (page-locked)");
        }
        if (this->m_allowDataOverwrite) {
            REPORT_ERROR(ERR_LOGIC_ERROR, "Incompatible parameters: allowDataOverwrite && 'others' data on host");
        }
    }

    m_workStream->set();
    // make sure that all work (e.g. loading and processing of the reference) is done
    m_loadStream->synch();
    m_workStream->synch();
    // start loading data at the background
    m_isDataReady = false;
    const bool createLocalCopy = ( ! this->m_allowDataOverwrite);
    std::thread loadingThread;
    if (createLocalCopy) {
        loadingThread = std::thread(&CudaShiftCorrEstimator<T>::loadThreadRoutine, this, others);
    }

    // reserve enough space for shifts
    this->m_shifts2D.reserve(this->m_settingsInv->fDim().n());
    // process signals
    for (size_t offset = 0; offset < this->m_settingsInv->fDim().n(); offset += this->m_settingsInv->batch()) {
        size_t batchOffset = createLocalCopy
                ? offset
                : std::min(offset, this->m_settingsInv->fDim().n() - this->m_settingsInv->batch());
        // how many signals to process
        size_t toProcess = std::min(
                this->m_settingsInv->batch(),
                this->m_settingsInv->fDim().n() - batchOffset);

        if (createLocalCopy) {
            waitAndConvert();
        } else {
            T *in = others + (batchOffset * this->m_settingsInv->sDim().sizeSingle());
            CudaFFT<T>::fft(*m_batchToFD, in, m_d_batch_FD);
        }

        // compute shifts
        auto shifts = computeShifts2DOneToN(
                std::vector<GPU*>{m_workStream, m_loadStream},
                m_d_batch_FD,
                m_d_batch_SD_work,
                m_d_single_FD,
                this->m_settingsInv->createSubset(toProcess),
                *m_batchToSD,
                this->m_h_centers,
                this->m_maxShift);

        size_t start = createLocalCopy
                ? 0
                : (offset - batchOffset);
        // append shifts to existing results
        this->m_shifts2D.insert(this->m_shifts2D.end(),
                shifts.begin() + start,
                shifts.end());
    }
    if (createLocalCopy) {
        loadingThread.join();
    }
    // update state
    this->m_is_shift_computed = true;
}

template<typename T>
std::vector<Point2D<float>> CudaShiftCorrEstimator<T>::computeShifts2DOneToN(
        const std::vector<GPU*> &gpus,
        std::complex<T> *d_othersF,
        T *d_othersS,
        std::complex<T> *d_ref,
        const FFTSettings<T> &settings,
        cufftHandle plan,
        T *h_centers,
        size_t maxShift) {
    // we need even input in order to perform the shift (in FD, while correlating) properly
    assert(0 == (settings.sDim().x() % 2));
    assert(0 == (settings.sDim().y() % 2));
    assert(1 == settings.sDim().zPadded());
    assert(2 == gpus.size());

    size_t centerSize = maxShift * 2 + 1;

    // make sure we have enough memory for centers of the correlation functions
    assert(settings.fBytesBatch() >= (settings.sDim().n() * centerSize * centerSize * sizeof(T)));

    auto workGPU = gpus.at(0);
    auto loadGPU = gpus.at(1);
    loadGPU->set();
    workGPU->set();
    auto loadStream = *(cudaStream_t*)loadGPU->stream();
    auto workStream = *(cudaStream_t*)workGPU->stream();

    // correlate signals and shift FT so that it will be centered after IFT
    sComputeCorrelations2DOneToN<true>(*workGPU,
            d_othersF, d_ref,
            settings.fDim());

    // perform IFT
    CudaFFT<T>::ifft(plan, d_othersF, d_othersS);

    // locate maxima
    auto p_pos = (float*)d_othersF;
    auto h_pos = (float*)h_centers;
    ExtremaFinder::CudaExtremaFinder<T>::sFindMax2DAroundCenter(
            *workGPU, settings.sDim(), d_othersS, p_pos, nullptr, maxShift);
    workGPU->synch();
    // copy data back
    gpuErrchk(cudaMemcpyAsync(h_pos, p_pos,
            settings.batch() * sizeof(float),
            cudaMemcpyDeviceToHost, loadStream));
    loadGPU->synch();

    // convert absolute 1D index to logical 2D index
    // FIXME DS extract this to some utils
    auto result = std::vector<Point2D<float>>();
    result.reserve(settings.sDim().n());
    for (size_t n = 0; n < settings.batch(); ++n) {
        float y = (size_t)h_pos[n] / settings.sDim().x();
        float x = (size_t)h_pos[n] % settings.sDim().x();
        result.emplace_back(
            x - settings.sDim().x() / 2,
            y - settings.sDim().y() / 2);
    }
    return result;
}

template<typename T>
void CudaShiftCorrEstimator<T>::computeCorrelations2DOneToN(
        const HW &hw,
        std::complex<T> *inOut,
        const std::complex<T> *ref,
        const Dimensions &dims,
        bool center) {
    const GPU *gpu;
    try {
        gpu = &dynamic_cast<const GPU&>(hw);
    } catch (std::bad_cast&) {
        REPORT_ERROR(ERR_ARG_INCORRECT, "Instance of GPU expected");
    }
    if (center) {
        return sComputeCorrelations2DOneToN<true>(*gpu, inOut, ref, dims);
    } else {
        return sComputeCorrelations2DOneToN<false>(*gpu, inOut, ref, dims);
    }
}

template<typename T>
template<bool center>
void CudaShiftCorrEstimator<T>::sComputeCorrelations2DOneToN(
        const GPU &gpu,
        std::complex<T> *d_inOut,
        const std::complex<T> *d_ref,
        const Dimensions &dims) {
    if (center) {
        // we cannot assert xDim, as we don't know if the spatial size was even
        assert(0 == (dims.y() % 2));
    }
    assert(0 < dims.x());
    assert(0 < dims.y());
    assert(1 == dims.z());
    assert(0 < dims.n());

    // create threads / blocks
    dim3 dimBlock(64, 1, 1);
    dim3 dimGrid(
            std::ceil(dims.x() / (float)dimBlock.x),
            dims.n(), 1);
    auto stream = *(cudaStream_t*)gpu.stream();
    if (std::is_same<T, float>::value) {
        computeCorrelations2DOneToNKernel<float2, center>
            <<<dimGrid, dimBlock, 0, stream>>> (
                (float2*)d_inOut, (float2*)d_ref,
                dims.x(), dims.y(), dims.n());
    } else if (std::is_same<T, double>::value) {
        computeCorrelations2DOneToNKernel<double2, center>
            <<<dimGrid, dimBlock, 0, stream>>> (
                (double2*)d_inOut, (double2*)d_ref,
                dims.x(), dims.y(), dims.n());
    } else {
        REPORT_ERROR(ERR_TYPE_INCORRECT, "Not implemented");
    }
}

template<typename T>
void CudaShiftCorrEstimator<T>::check() {
    if (this->m_settingsInv->isInPlace()) {
        REPORT_ERROR(ERR_VALUE_INCORRECT, "Only out-of-place transform is supported");
    }
    if ((this->m_centerSize * this->m_centerSize * sizeof(T)) > this->m_settingsInv->fBytesBatch()) {
        // see computeShifts2DOneToN
        REPORT_ERROR(ERR_NOT_IMPLEMENTED, "This implementation is not able to handle cases when"
                "maxShift is more than half of the dimension of the input");
    }
}

// explicit instantiation
template class CudaShiftCorrEstimator<float>;
template class CudaShiftCorrEstimator<double>;

} /* namespace Alignment */
