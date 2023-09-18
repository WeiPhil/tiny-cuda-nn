#pragma once
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/loss.h>

namespace tcnn {

template <typename T>
__global__ void binary_cross_entropy_loss(
	const uint32_t n_elements,
	const uint32_t stride,
	const uint32_t dims,
	const float loss_scale,
	const T * __restrict__ predictions,
	const float * __restrict__ targets,
	float * __restrict__ values,
	T * __restrict__ gradients)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements)
		return;

	const uint32_t dim_idx = i % stride;
	const uint32_t elem_idx = i / stride;
	if (dim_idx >= dims)
	{
		values[i] = 0;
		gradients[i] = 0;
		return;
	}

	const uint32_t target_idx = elem_idx * dims + dim_idx;

	const uint32_t n_total = n_elements / stride * dims;

	const float epsilon = 1e-5;
    const float target = targets[target_idx];
	const float prediction = min(max((float)predictions[i],epsilon), 1.f - epsilon);

    values[i] = -(target * log(prediction) + (1.0 - target) * log(1.0 - prediction)) / n_total;

    float gradient = -(target-prediction) / (prediction - (prediction*prediction));
	gradients[i] = (T)(loss_scale * gradient / n_total);
}

template <typename T>
class BinaryCrossEntropyLoss : public Loss<T> {
public:
	void evaluate(
		cudaStream_t stream,
		const float loss_scale,
		const GPUMatrix<T>& prediction,
		const GPUMatrix<float>& target,
		GPUMatrix<float>& values,
		GPUMatrix<T>& gradients,
		const GPUMatrix<float>* data_pdf = nullptr) const override
	{
		const uint32_t dims = target.m();
		const uint32_t stride = prediction.m();

		CHECK_THROW(prediction.n() == target.n());
		CHECK_THROW(values.m() == stride);
		CHECK_THROW(gradients.m() == stride);
		CHECK_THROW(!data_pdf || data_pdf->m() == dims);
		
		tcnn::linear_kernel(
			binary_cross_entropy_loss<T>,
			0,
			stream,
			prediction.n_elements(),
			stride,
			dims,
			loss_scale,
			prediction.data(),
			target.data(),
			values.data(),
			gradients.data());
	}

	void update_hyperparams(const nlohmann::json & params) override
	{
	}

	nlohmann::json hyperparams() const override
	{
		return {
			{"otype", "BinaryCrossEntropy"},
		};
	}
};

}