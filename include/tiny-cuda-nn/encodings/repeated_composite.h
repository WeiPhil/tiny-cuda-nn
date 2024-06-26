#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>

#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

namespace tcnn {

template <typename T>
class RepeatedCompositeEncoding : public Encoding<T> {
public:
	RepeatedCompositeEncoding(const json& params, uint32_t n_dims_to_encode, uint32_t n_repetitions)
	: m_n_dims_to_encode{n_dims_to_encode},m_n_repetitions(n_repetitions) {
		if (!params.contains("nested") || !params["nested"].is_array()) {
			throw std::runtime_error{"Must provide an array of nested encodings to RepeatedCompositeEncoding."};
		}

		m_reduction_type = string_to_reduction_type(params.value("reduction", "Concatenation"));

		const json::array_t& nested = params["nested"];

		uint32_t total_nested_dims_to_encode = 0;
		for (size_t i = 0; i < nested.size(); ++i) {
			total_nested_dims_to_encode += nested[i].value("n_dims_to_encode", 0);
			if (nested[i].contains("dims_to_encode_begin")) {
				total_nested_dims_to_encode = 0xFFFFFFFF;
				break;
			}
		}

		total_nested_dims_to_encode *= n_repetitions;

		if (total_nested_dims_to_encode != 0xFFFFFFFF && total_nested_dims_to_encode != n_dims_to_encode) {
			throw std::runtime_error{fmt::format("RepeatedCompositeEncoding: nested encodings must not encode more dims {} than composite {}", total_nested_dims_to_encode, n_dims_to_encode)};
		}

		uint32_t unspecified_dims_to_encode = total_nested_dims_to_encode == 0xFFFFFFFF ? 0xFFFFFFFF : (n_dims_to_encode - total_nested_dims_to_encode);
		uint32_t offset = 0;

		// Create encodings with somewhat arbitrary alignment
		for (size_t rep_i = 0; rep_i < m_n_repetitions; rep_i++)
		{
			for (size_t i = 0; i < nested.size(); ++i) {
				uint32_t nested_dims_to_encode;
				if (nested[i].contains("n_dims_to_encode")) {
					if (nested[i].contains("dims_to_encode_begin")) {
						offset = nested[i]["dims_to_encode_begin"];
						throw std::runtime_error{"dims_to_encode_begin not supported with repeated encodings"};
					}

					nested_dims_to_encode = nested[i]["n_dims_to_encode"];
				} else {
					if (unspecified_dims_to_encode == 0xFFFFFFFF) {
						throw std::runtime_error{"RepeatedCompositeEncoding: may only leave 'n_dims_to_encode' unspecified for a single nested encoding"};
					}
					nested_dims_to_encode = unspecified_dims_to_encode;
					unspecified_dims_to_encode = 0xFFFFFFFF;
				}

				if (nested_dims_to_encode > 0) {
					// Only keep one encoding
					if(rep_i == 0)
						m_nested.emplace_back(create_encoding<T>(nested_dims_to_encode, nested[i], 1));
					m_dims_to_encode_begin.emplace_back(offset);
				}

				offset += nested_dims_to_encode;
			}
		}

		// Fix alignment such that min_alignment of each individual encoding's output is ensured
		if (m_reduction_type == ReductionType::Concatenation) {
			uint32_t dims_encoded_so_far = 0;
			for (size_t i = 0; i < m_nested.size()-1; ++i) {
				uint32_t desired_alignment = m_nested[i+1]->required_output_alignment();
				uint32_t padded_output_width_required = next_multiple(dims_encoded_so_far + m_nested[i]->output_width(), desired_alignment) - dims_encoded_so_far;

				m_nested[i]->set_padded_output_width(padded_output_width_required);

				dims_encoded_so_far += m_nested[i]->padded_output_width();
			}
		} else {
			uint32_t alignment = required_output_alignment();
			for (const auto& nested : m_nested) {
				nested->set_alignment(alignment);
			}

			if (!m_nested.empty()) {
				uint32_t output_width = m_nested.front()->output_width();
				for (const auto& nested : m_nested) {
					CHECK_THROW(nested->output_width() == output_width);
				}
			}
		}
	}

	std::unique_ptr<Context> forward_impl(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		GPUMatrixDynamic<T>* output = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false
	) override {
		if (m_n_dims_to_encode == 0) {
			return std::make_unique<ForwardContext>();
		}

		auto forward = std::make_unique<ForwardContext>();
		forward->nested.resize(m_nested.size());

		GPUMatrixDynamic<T>* reduced_output = output;
		if (m_reduction_type != ReductionType::Concatenation) {
			forward->to_reduce = GPUMatrixDynamic<T>{padded_output_width() * (uint32_t)m_nested.size() * m_n_repetitions, input.n(), stream, preferred_output_layout()};
			output = &forward->to_reduce;
		}

		uint32_t output_offset = 0;

		for (size_t rep_i = 0; rep_i < m_n_repetitions; rep_i++)
		{
			for (size_t i = 0; i < m_nested.size(); ++i) {
				const auto& nested = m_nested[i];
                uint32_t input_offset = m_dims_to_encode_begin[i + rep_i * m_nested.size()];
                uint32_t input_width = nested->input_width();
				uint32_t output_width = nested->output_width();

				GPUMatrixDynamic<T> sliced_output;
				if (output) {
					sliced_output = output->slice_rows(output_offset, output_width);
				}

				forward->nested[i] = nested->forward(
					stream, // TODO: use SyncedMultiStream but ensure memory arena allocations happen on `stream`
					input.slice_rows(input_offset, input_width),
					output ? &sliced_output : nullptr,
					use_inference_params,
					prepare_input_gradients
				);

				input_offset += input_width;
				output_offset += output_width;
			}
		}

		if (reduced_output && m_reduction_type != ReductionType::Concatenation) {
			switch (m_reduction_type) {
				case ReductionType::Sum: linear_kernel(reduce_sum_forward<T>, 0, stream,
					input.n(),
					padded_output_width(),
					(uint32_t)m_nested.size() * m_n_repetitions,
					forward->to_reduce.view(),
					reduced_output->view()
				); break;
				case ReductionType::Product: linear_kernel(reduce_product_forward<T>, 0, stream,
					input.n(),
					padded_output_width(),
					(uint32_t)m_nested.size() * m_n_repetitions,
					forward->to_reduce.view(),
					reduced_output->view()
				); break;
				default: throw std::runtime_error{"RepeatedCompositeEncoding::forward: invalid reduction type."};
			}
		}

		return forward;
	}

	void backward_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) override {
		if (m_n_dims_to_encode == 0) {
			return;
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		if (forward.nested.size() != m_nested.size()) {
			throw std::runtime_error{"RepeatedCompositeEncoding::backward called with incompatible context size."};
		}

		const GPUMatrixDynamic<T>* dL_dunreduced_output = &dL_doutput;
		GPUMatrixDynamic<T> dL_dnested_output;
		if (m_reduction_type != ReductionType::Concatenation) {
			dL_dnested_output = GPUMatrixDynamic<T>{forward.to_reduce.m(), forward.to_reduce.n(), stream, forward.to_reduce.layout()};
			dL_dunreduced_output = &dL_dnested_output;
			switch (m_reduction_type) {
				case ReductionType::Sum: linear_kernel(reduce_sum_backward<T>, 0, stream,
					input.n(),
					padded_output_width(),
					(uint32_t)m_nested.size() * m_n_repetitions,
					dL_dunreduced_output->view(),
					dL_doutput.view()
				); break;
				case ReductionType::Product: linear_kernel(reduce_product_backward<T>, 0, stream,
					input.n(),
					padded_output_width(),
					(uint32_t)m_nested.size() * m_n_repetitions,
					forward.to_reduce.view(),
					dL_dunreduced_output->view(),
					dL_doutput.view()
				); break;
				default: throw std::runtime_error{"RepeatedCompositeEncoding::backward: invalid reduction type."};
			}
		}

		SyncedMultiStream synced_streams{stream, m_nested.size()};

		uint32_t output_offset = 0;

		for (size_t rep_i = 0; rep_i < m_n_repetitions; rep_i++)
		{
		
			for (size_t i = 0; i < m_nested.size(); ++i) {
				const auto& nested = m_nested[i];
				uint32_t input_offset = m_dims_to_encode_begin[i + rep_i * m_nested.size()];
				uint32_t input_width = nested->input_width();
				uint32_t output_width = nested->output_width();

				GPUMatrixDynamic<float> sliced_dL_dinput;
				if (dL_dinput) {
					sliced_dL_dinput = dL_dinput->slice_rows(input_offset, input_width);
				}

				nested->backward(
					synced_streams.get(i),
					*forward.nested[i],
					input.slice_rows(input_offset, input_width),
					output.slice_rows(output_offset, output_width),
					dL_dunreduced_output->slice_rows(output_offset, output_width),
					dL_dinput ? &sliced_dL_dinput : nullptr,
					use_inference_params,
					rep_i == 0 ? param_gradients_mode : GradientMode::Accumulate 
				);

				output_offset += output_width;
			}

		}
		
	}

	uint32_t input_width() const override {
		return m_n_dims_to_encode;
	}

	uint32_t padded_output_width() const override {
		if (m_reduction_type != ReductionType::Concatenation) {
			return m_nested.empty() ? 0 : m_nested.front()->padded_output_width();
		}

		uint32_t total = 0;
		for (const auto& nested : m_nested) {
			total += nested->padded_output_width();
		}
		total *= m_n_repetitions;
		return total;
	}

	uint32_t output_width() const override {
		return padded_output_width();
	}

	uint32_t required_input_alignment() const override {
		return 1;
	}

	void set_padded_output_width(uint32_t padded_output_width) override {
		if (m_reduction_type == ReductionType::Concatenation) {
			uint32_t prev_n_dims = this->padded_output_width() - m_n_repetitions * m_nested.back()->padded_output_width();
			CHECK_THROW(padded_output_width >= prev_n_dims);
			CHECK_THROW((padded_output_width - prev_n_dims) % m_n_repetitions == 0);
			m_nested.back()->set_padded_output_width((padded_output_width - prev_n_dims)/ m_n_repetitions);
		} else {
			for (const auto& nested : m_nested) {
				nested->set_padded_output_width(padded_output_width);
			}
		}
	}

	uint32_t required_output_alignment() const override {
		// uint32_t alignment = 1; 
		uint32_t alignment = m_n_repetitions;
		for (const auto& nested : m_nested) {
			alignment = lcm(alignment, nested->required_output_alignment());
		}
		return alignment;
	}

	MatrixLayout preferred_output_layout() const override {
		// Output layout of first nested encoding (tends to be the most significant, i.e. hash encoding)
		return m_nested.empty() ? AoS : m_nested.front()->preferred_output_layout();
	}

	void set_params_impl(T* params, T* inference_params, T* gradients) override {
		size_t offset = 0;
		for (auto& nested : m_nested) {
			nested->set_params(params + offset, inference_params + offset, gradients + offset);
			offset += nested->n_params();
		}
	}

	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
		for (auto& nested : m_nested) {
			nested->initialize_params(rnd, params_full_precision, scale);
			params_full_precision += nested->n_params();
		}
	}

	size_t n_params() const override {
		size_t total = 0;
		for (const auto& nested : m_nested) {
			total += nested->n_params();
		}
		return total;
	}

	json hyperparams() const override {
		json::array_t nested;
		for (auto& n : m_nested) {
			nested.emplace_back(n->hyperparams());
		}

		return {
			{"otype", "Composite"},
			{"nested", nested}
		};
	}

private:
	struct ForwardContext : public Context {
		std::vector<std::unique_ptr<Context>> nested;
		GPUMatrixDynamic<T> to_reduce;
	};

	std::vector<std::shared_ptr<Encoding<T>>> m_nested;
	std::vector<uint32_t> m_dims_to_encode_begin;
	uint32_t m_n_dims_to_encode;
	uint32_t m_n_repetitions;
	ReductionType m_reduction_type = ReductionType::Concatenation;
};

}
