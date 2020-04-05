#include <torch/extension.h>

#include <vector>

namespace mmsegmentation {
at::Tensor ca_forward_cuda(const at::Tensor& t, const at::Tensor& f);

std::tuple<at::Tensor, at::Tensor> ca_backward_cuda(const at::Tensor& dw,
                                                    const at::Tensor& t,
                                                    const at::Tensor& f);

at::Tensor ca_map_forward_cuda(const at::Tensor& weight, const at::Tensor& g);

std::tuple<at::Tensor, at::Tensor> ca_map_backward_cuda(
    const at::Tensor& dout, const at::Tensor& weight, const at::Tensor& g);

at::Tensor ca_forward(const at::Tensor& t, const at::Tensor& f) {
  if (t.type().is_cuda()) {
#ifdef WITH_CUDA
    return ca_forward_cuda(t, f);
#else
    AT_ERROR("ca is not compiled with GPU support");
#endif
  }
  AT_ERROR("ca is not implemented on the CPU");
}

std::tuple<at::Tensor, at::Tensor> ca_backward(const at::Tensor& dw,
                                               const at::Tensor& t,
                                               const at::Tensor& f) {
  if (dw.type().is_cuda()) {
#ifdef WITH_CUDA
    return ca_backward_cuda(dw, t, f);
#else
    AT_ERROR("ca is not compiled with GPU support");
#endif
  }
  AT_ERROR("ca is not implemented on the CPU");
}

at::Tensor ca_map_forward(const at::Tensor& weight, const at::Tensor& g) {
  if (weight.type().is_cuda()) {
#ifdef WITH_CUDA
    return ca_map_forward_cuda(weight, g);
#else
    AT_ERROR("ca is not compiled with GPU support");
#endif
  }
  AT_ERROR("ca is not implemented on the CPU");
}

std::tuple<at::Tensor, at::Tensor> ca_map_backward(const at::Tensor& dout,
                                                   const at::Tensor& weight,
                                                   const at::Tensor& g) {
  if (dout.type().is_cuda()) {
#ifdef WITH_CUDA
    return ca_map_backward_cuda(dout, weight, g);
#else
    AT_ERROR("ca is not compiled with GPU support");
#endif
  }
  AT_ERROR("ca is not implemented on the CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ca_forward", &ca_forward, "ca forward");
  m.def("ca_backward", &ca_backward, "ca backward");
  m.def("ca_map_forward", &ca_map_forward, "ca map forward");
  m.def("ca_map_backward", &ca_map_backward, "ca map backward");
}

}  // namespace mmsegmentation
