#include <torch/extension.h>

namespace mmsegmentation {

void psamask_forward_cpu(const int psa_type, const at::Tensor& input,
                         at::Tensor& output, const int num_,
                         const int h_feature, const int w_feature,
                         const int h_mask, const int w_mask,
                         const int half_h_mask, const int half_w_mask);
void psamask_backward_cpu(const int psa_type, const at::Tensor& grad_output,
                          at::Tensor& grad_input, const int num_,
                          const int h_feature, const int w_feature,
                          const int h_mask, const int w_mask,
                          const int half_h_mask, const int half_w_mask);

void psamask_forward_cuda(const int psa_type, const at::Tensor& input,
                          at::Tensor& output, const int num_,
                          const int h_feature, const int w_feature,
                          const int h_mask, const int w_mask,
                          const int half_h_mask, const int half_w_mask);
void psamask_backward_cuda(const int psa_type, const at::Tensor& grad_output,
                           at::Tensor& grad_input, const int num_,
                           const int h_feature, const int w_feature,
                           const int h_mask, const int w_mask,
                           const int half_h_mask, const int half_w_mask);

void psamask_forward(const int psa_type, const at::Tensor& input,
                     at::Tensor& output, const int num_, const int h_feature,
                     const int w_feature, const int h_mask, const int w_mask,
                     const int half_h_mask, const int half_w_mask) {
  if (input.device().is_cuda()) {
#ifdef WITH_CUDA
    psamask_forward_cuda(psa_type, input, output, num_, h_feature, w_feature,
                         h_mask, w_mask, half_h_mask, half_w_mask);
#else
    AT_ERROR("psamask is not compiled with GPUS support");
#endif
  } else {
    psamask_forward_cpu(psa_type, input, output, num_, h_feature, w_feature,
                        h_mask, w_mask, half_h_mask, half_w_mask);
  }
}

void psamask_backward(const int psa_type, const at::Tensor& grad_output,
                      at::Tensor& grad_input, const int num_,
                      const int h_feature, const int w_feature,
                      const int h_mask, const int w_mask, const int half_h_mask,
                      const int half_w_mask) {
  if (grad_input.device().is_cuda()) {
#ifdef WITH_CUDA
    psamask_backward_cuda(psa_type, grad_output, grad_input, num_, h_feature,
                          w_feature, h_mask, w_mask, half_h_mask, half_w_mask);
#else
    AT_ERROR("psamask is not compiled with GPUS support");
#endif
  } else {
    psamask_backward_cpu(psa_type, grad_output, grad_input, num_, h_feature,
                         w_feature, h_mask, w_mask, half_h_mask, half_w_mask);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("psamask_forward", &psamask_forward, "PSAMASK forward (CPU/CUDA)");
  m.def("psamask_backward", &psamask_backward, "PSAMASK backward (CPU/CUDA)");
}

}  // namespace mmsegmentation
