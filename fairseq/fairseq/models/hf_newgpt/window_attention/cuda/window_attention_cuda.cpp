#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor window_attention_cuda_forward(
    torch::Tensor query, // [batch, head, len, dim]
    torch::Tensor key,
    int window);

std::vector<torch::Tensor> window_attention_cuda_backward(
    torch::Tensor grad_o,
    torch::Tensor query,
    torch::Tensor key);

torch::Tensor window_value_cuda_forward(
    torch::Tensor attn_weight, // [batch, head, len, window]
    torch::Tensor value);

std::vector<torch::Tensor> window_value_cuda_backward(
    torch::Tensor grad_o,
    torch::Tensor attn_weight,
    torch::Tensor value);
// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")

torch::Tensor window_attention_forward(
    torch::Tensor query, // [batch, head, len, dim]
    torch::Tensor key,
    int window) {
  CHECK_CUDA(query);
  CHECK_CUDA(key);

  return window_attention_cuda_forward(
    query.contiguous(),
    key.contiguous(),
    window
  );
}

std::vector<torch::Tensor> window_attention_backward(
    torch::Tensor grad_o,
    torch::Tensor query,
    torch::Tensor key) {
  CHECK_CUDA(grad_o);
  CHECK_CUDA(query);
  CHECK_CUDA(key);

  return window_attention_cuda_backward(
    grad_o.contiguous(),
    query.contiguous(),
    key.contiguous()
  );
}

torch::Tensor window_value_forward(
    torch::Tensor attn_weight, // [batch, head, len, window]
    torch::Tensor value) {
  CHECK_CUDA(attn_weight);
  CHECK_CUDA(value);

  return window_value_cuda_forward(
    attn_weight.contiguous(),
    value.contiguous()
  );
}

std::vector<torch::Tensor> window_value_backward(
    torch::Tensor grad_o,
    torch::Tensor attn_weight,
    torch::Tensor value) {
  CHECK_CUDA(grad_o);
  CHECK_CUDA(attn_weight);
  CHECK_CUDA(value);

  return window_value_cuda_backward(
    grad_o.contiguous(),
    attn_weight.contiguous(),
    value.contiguous()
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_forward", &window_attention_forward, "Window Attention forward (CUDA)");
  m.def("attn_backward", &window_attention_backward, "Window Attention backward (CUDA)");
  m.def("value_forward", &window_value_forward, "Window Value forward (CUDA)");
  m.def("value_backward", &window_value_backward, "Window Value backward (CUDA)");
}
