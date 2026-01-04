/***************************************************************************
 *
 * Authors:    Martin Horacek (horacek1martin@gmail.com)
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
#include "cuda_volume_restoration_kernels.h"

#include <iostream>

#include <cuda_runtime_api.h>

#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#include "cuda_asserts.h"
#include "cuda_vec2.h"

#include "cuda_volume_restoration_kernels.cu"

namespace Gpu {

template< typename T >
void VolumeRestorationKernels<T>::computeWeights(const T* d_Vfiltered1, const T* d_Vfiltered2, T* d_V1r, T* d_V2r, T* d_S, size_t volume_size, const Gpu::CDF<T>& cdf_mN,
	T weightPower, int weightFun) {

	unsigned block_size = 256;

	dim3 dimBlock{ block_size };
    dim3 dimGrid{ static_cast<unsigned>( ceil( static_cast<double>(volume_size) / dimBlock.x) ) };

	computeWeightsKernel<<<dimGrid, dimBlock>>>(d_Vfiltered1, d_Vfiltered2, d_V1r, d_V2r, d_S, cdf_mN.d_x,
		cdf_mN.d_probXLessThanx, cdf_mN.d_V, volume_size, cdf_mN.Nsteps, weightPower, weightFun);
	gpuErrchk(cudaPeekAtLastError());
}

template< typename T >
void VolumeRestorationKernels<T>::filterFourierVolume(const T* d_R2, const std::complex<T>* d_fV, std::complex<T>* d_buffer, size_t volume_size, T w2, T w2Step) {
	unsigned block_size = 256;

	dim3 dimBlock{ block_size };
	dim3 dimGrid{ static_cast<unsigned>( ceil( static_cast<double>(volume_size) / dimBlock.x) ) };

	filterFourierVolumeKernel<<<dimGrid, dimBlock>>>(d_R2, (vec2_type<T>*)d_fV, (vec2_type<T>*)d_buffer, volume_size, w2, w2Step);
	gpuErrchk(cudaPeekAtLastError());
}

template< typename T >
void VolumeRestorationKernels<T>::computeAveragePositivity(const T* d_V1, const T* d_V2, T* d_S, size_t volume_size) {
	unsigned block_size = 256;

	dim3 dimBlock{ block_size };
	dim3 dimGrid{ static_cast<unsigned>( ceil( static_cast<double>(volume_size) / dimBlock.x) ) };

	computeAveragePositivityKernel<<<dimGrid, dimBlock>>>(d_V1, d_V2, d_S, volume_size, static_cast<T>(1.0) / volume_size);
	gpuErrchk(cudaPeekAtLastError());
}

template< typename T >
void VolumeRestorationKernels<T>::computeAveragePositivity(const T* d_V1, const T* d_V2, T* d_S, const int* d_mask, size_t volume_size) {
	unsigned block_size = 256;

	dim3 dimBlock{ block_size };
	dim3 dimGrid{ static_cast<unsigned>( ceil( static_cast<double>(volume_size) / dimBlock.x) ) };

	computeAveragePositivityKernel<<<dimGrid, dimBlock>>>(d_V1, d_V2, d_S, d_mask, volume_size, static_cast<T>(1.0) / volume_size);
	gpuErrchk(cudaPeekAtLastError());
}

template< typename T >
void VolumeRestorationKernels<T>::filterS(const T* d_R2, std::complex<T>* d_fVol, size_t volume_size) {
	unsigned block_size = 256;

	dim3 dimBlock{ block_size };
	dim3 dimGrid{ static_cast<unsigned>( ceil( static_cast<double>(volume_size) / dimBlock.x) ) };

	filterSKernel<<<dimGrid, dimBlock>>>(d_R2, (vec2_type<T>*)d_fVol, volume_size);
	gpuErrchk(cudaPeekAtLastError());
}

template< typename T >
void VolumeRestorationKernels<T>::maskForCDF(T* __restrict__ d_aux, const T* __restrict__ d_S, const int* __restrict__ d_mask, size_t volume_size) {
	auto k = [] __device__ (int x) {
		return x;
	};

	thrust::copy_if(thrust::device, d_S, d_S + volume_size, d_mask, d_aux, k);
}

template< typename T >
void VolumeRestorationKernels<T>::maskWithNoiseProbability(T* d_V, const Gpu::CDF<T>& cdf_S, const Gpu::CDF<T>& cdf_N, size_t volume_size) {
	unsigned block_size = 256;

	dim3 dimBlock{ block_size };
	dim3 dimGrid{ static_cast<unsigned>( ceil( static_cast<double>(volume_size) / dimBlock.x) ) };

	maskWithNoiseProbabilityKernel<<<dimGrid, dimBlock>>>(d_V, cdf_S.d_x,
		cdf_S.d_probXLessThanx, cdf_S.d_V, cdf_S.Nsteps, cdf_S.volume_size, cdf_N.d_x,
		cdf_N.d_probXLessThanx, cdf_N.d_V, cdf_N.Nsteps, cdf_N.volume_size);
	gpuErrchk(cudaPeekAtLastError());
}

template< typename T >
void VolumeRestorationKernels<T>::deconvolveRestored(std::complex<T>* d_fVol, std::complex<T>* d_fV1, std::complex<T>* d_fV2, const T* d_R2, T K1, T K2, T lambda, size_t volume_size, size_t fourier_size) {
	unsigned block_size = 256;

	dim3 dimBlock{ block_size };
	dim3 dimGrid{ static_cast<unsigned>( ceil( static_cast<double>(fourier_size) / dimBlock.x) ) };

	deconvolveRestoredKernel<<<dimGrid, dimBlock>>>((vec2_type<T>*)d_fVol, (vec2_type<T>*)d_fV1, (vec2_type<T>*)d_fV2, d_R2, K1, K2, lambda, fourier_size, static_cast<T>(1.0) / volume_size);
	gpuErrchk(cudaPeekAtLastError());
}

template< typename T >
void VolumeRestorationKernels<T>::convolveFourierVolume(std::complex<T>* d_fVol, const T* d_R2, T K, size_t volume_size) {
	unsigned block_size = 256;

	dim3 dimBlock{ block_size };
	dim3 dimGrid{ static_cast<unsigned>( ceil( static_cast<double>(volume_size) / dimBlock.x) ) };

	convolveFourierVolumeKernel<<<dimGrid, dimBlock>>>((vec2_type<T>*)d_fVol, d_R2, K, volume_size);
	gpuErrchk(cudaPeekAtLastError());
}

template< typename T >
void VolumeRestorationKernels<T>::normalizeForFFT(T* d_V1, T* d_V2, size_t volume_size) {
	unsigned block_size = 256;

	dim3 dimBlock{ block_size };
	dim3 dimGrid{ static_cast<unsigned>( ceil( static_cast<double>(volume_size) / dimBlock.x) ) };

	normalizeForFFTKernel<<<dimGrid, dimBlock>>>(d_V1, d_V2, volume_size, static_cast<T>(1.0) / volume_size);
	gpuErrchk(cudaPeekAtLastError());
}

template< typename T >
void VolumeRestorationKernels<T>::normalizeForFFT(T* d_V1, size_t volume_size) {
	unsigned block_size = 256;

	dim3 dimBlock{ block_size };
	dim3 dimGrid{ static_cast<unsigned>( ceil( static_cast<double>(volume_size) / dimBlock.x) ) };

	normalizeForFFTKernel<<<dimGrid, dimBlock>>>(d_V1, volume_size, static_cast<T>(1.0) / volume_size);
	gpuErrchk(cudaPeekAtLastError());
}

template< typename T >
void VolumeRestorationKernels<T>::restorationSigmaCostError(T& error, const std::complex<T>* _d_fVol, const std::complex<T>* _d_fV1, const std::complex<T>* _d_fV2, const T* __restrict__ d_R2, T K1, T K2, size_t fourier_size) {
	const T inv_size = 1.0 / (2 * fourier_size);

	const vec2_type<T>* __restrict__ d_fVol = (vec2_type<T>*)_d_fVol;
	const vec2_type<T>* __restrict__ d_fV1 = (vec2_type<T>*)_d_fV1;
	const vec2_type<T>* __restrict__ d_fV2 = (vec2_type<T>*)_d_fV2;

	auto error_func = [=] __host__ __device__ (int n) -> T {
		const T R2n = d_R2[n];
		if (R2n <= 0.25) {
			const T H1 = exp(K1 * R2n);
			const T H2 = exp(K2 * R2n);

			const T diff1_x = (d_fVol[n].x*H1 - d_fV1[n].x) * inv_size;
			const T diff1_y = (d_fVol[n].y*H1 - d_fV1[n].y) * inv_size;

			const T diff2_x = (d_fVol[n].x*H2 - d_fV2[n].x) * inv_size;
			const T diff2_y = (d_fVol[n].y*H2 - d_fV2[n].y) * inv_size;

			return sqrt(diff1_x*diff1_x + diff1_y*diff1_y) + sqrt(diff2_x*diff2_x + diff2_y*diff2_y);
		}

		return static_cast<T>(0.0);
	};

	error = thrust::transform_reduce(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(fourier_size), error_func, static_cast<T>(0), thrust::plus<T>());
}

template< typename T >
void VolumeRestorationKernels<T>::computeDiffAndAverage(const T* __restrict__ d_V1, const T* __restrict__ d_V2, T* __restrict__ d_S, T* __restrict__ d_N, size_t volume_size) {
	auto k = [=] __host__ __device__ (int n) {
		d_N[n] = d_V1[n] - d_V2[n];
		d_S[n] = (d_V1[n] + d_V2[n]) * static_cast<T>(0.5);
	};

	thrust::for_each_n(thrust::device, thrust::counting_iterator<int>(0), volume_size, k);
}

template< typename T >
std::pair<T, T> normAvgStd(T avg, T std, size_t size) {
	avg /= size;

	if (size > 1) {
		std = std / size - avg * avg;
		std *= static_cast<T>(size) / (size - 1);
		std = sqrt(abs(std));
	} else {
		std = 0;
	}

	return { avg, std };
}

template< typename T >
std::pair<T, T> VolumeRestorationKernels<T>::computeAvgStd(const T* __restrict__ d_N, size_t volume_size) {
	const T avg = thrust::reduce(thrust::device, d_N, d_N + volume_size);

	auto square_kernel = [=] __host__ __device__ (T x) -> T {
		return x * x;
	};

	const T std = thrust::transform_reduce(thrust::device, d_N, d_N + volume_size, square_kernel, static_cast<T>(0), thrust::plus<T>());

	return normAvgStd(avg, std, volume_size);
}

template< typename T >
std::pair<T, T> VolumeRestorationKernels<T>::computeAvgStdWithMask(const T* __restrict__ d_N, const int* __restrict__ d_mask, size_t mask_size, size_t volume_size) {
	auto masked_k = [=] __host__ __device__ (int n) -> T {
		if (d_mask[n]) {
			return d_N[n];
		} else {
			return static_cast<T>(0);
		}
	};

	const T avg = thrust::transform_reduce(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(volume_size), masked_k,
										static_cast<T>(0), thrust::plus<T>());

	auto masked_square_k = [=] __host__ __device__ (int n) -> T {
		if (d_mask[n]) {
			return d_N[n] * d_N[n];
		} else {
			return static_cast<T>(0);
		}
	};

	const T std = thrust::transform_reduce(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(volume_size), masked_square_k,
										static_cast<T>(0), thrust::plus<T>());

	return normAvgStd(avg, std, mask_size);
}

template< typename T >
void VolumeRestorationKernels<T>::computeDifference(T* __restrict__ d_V1, T* __restrict__ d_V2, const T* __restrict__ d_S, const T* __restrict__ d_N, T k, size_t volume_size) {
	auto ker = [=] __host__ __device__ (int n) {
		const T Nn = d_N[n];
		const T w = exp(k * Nn * Nn);
		const T s = d_S[n];
		d_V1[n] = s + (d_V1[n] - s) * w;
		d_V2[n] = s + (d_V2[n] - s) * w;
	};

	thrust::for_each_n(thrust::device, thrust::counting_iterator<int>(0), volume_size, ker);
}

template< typename T >
size_t VolumeRestorationKernels<T>::computeMaskSize(const int* __restrict__ d_mask, size_t volume_size) {
	return thrust::reduce(thrust::device, d_mask, d_mask + volume_size);
}

template< typename T >
void VolumeRestorationKernels<T>::multiplyByConstant(T* __restrict__ d_array, T c, size_t volume_size) {
	auto k = [=] __host__ __device__ (int n) {
		d_array[n] = d_array[n] * c;
	};

	thrust::for_each_n(thrust::device, thrust::counting_iterator<int>(0), volume_size, k);
}

template class VolumeRestorationKernels<double>;
template class VolumeRestorationKernels<float>;

} // namespace Gpu