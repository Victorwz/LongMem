#include <torch/extension.h>

#include <vector>

torch::Tensor window_attention_forward(
    torch::Tensor query, // [batch, head, len, dim]
    torch::Tensor key,
    int window) {
  query = query.permute({2, 0, 1, 3}); // [len, batch, head, dim]
  key = key.permute({2, 0, 1, 3});
  auto output = torch::zeros({query.size(0), window, query.size(1), query.size(2)}); // [len, window, batch, head]
  auto begin_offset = key.size(0) - window + 1 - query.size(0);
  for(int posi = 0; posi < output.size(0); posi++) {
    output[posi] = query[posi].mul(key.slice(0, posi + begin_offset, posi + begin_offset + window)).sum(/*dim=*/-1);
  }
  output = output.permute({2, 3, 0, 1});
  return output;
}

std::vector<torch::Tensor> window_attention_backward(
    torch::Tensor grad_o,
    torch::Tensor query,
    torch::Tensor key) {
  query = query.permute({2, 0, 1, 3}); // [len, batch, head, dim]
  key = key.permute({2, 0, 1, 3}); 
  grad_o = grad_o.permute({2, 3, 0, 1}); // [len, window, batch, head]
  auto d_query = torch::zeros_like(query);
  auto d_key = torch::zeros_like(key);
  auto window = grad_o.size(1);
  auto begin_offset = key.size(0) - window + 1 - query.size(0);
  for(int posi = 0; posi < grad_o.size(0); posi++) {
    d_query[posi] = grad_o[posi].unsqueeze(-1).mul(key.slice(0, posi + begin_offset, posi + begin_offset + window)).sum(0);
    d_key.slice(0, posi + begin_offset, posi + begin_offset + window) += grad_o[posi].unsqueeze(-1) * query[posi];
  }
  return {d_query.permute({1, 2, 0, 3}), d_key.permute({1, 2, 0, 3})};
}

torch::Tensor window_value_forward(
    torch::Tensor attn_weight, // [batch, head, len, window]
    torch::Tensor value) { // [batch, head, len, dim]
  const int batch = attn_weight.size(0), head = attn_weight.size(1), len_o = attn_weight.size(2), window = attn_weight.size(3), len_v = value.size(2), dim = value.size(3);
  attn_weight = attn_weight.permute({2, 3, 0, 1}); // [len, window, batch, head]
  value = value.permute({2, 0, 1, 3}); // [len, batch, head, dim]
  auto output = torch::zeros({len_o, batch, head, dim});
  auto begin_offset = len_v - window + 1 - len_o;
  for(int posi = 0; posi < output.size(0); posi++) {
    output[posi] = attn_weight[posi].unsqueeze(-1).mul(value.slice(0, posi + begin_offset, posi + begin_offset + window)).sum(/*dim=*/0);
  }
  output = output.permute({1, 2, 0, 3});
  return output;
}

std::vector<torch::Tensor> window_value_backward(
    torch::Tensor grad_o,
    torch::Tensor attn_weight, // [batch, head, len, window]
    torch::Tensor value) { // [batch, head, len, dim]
  const int batch = attn_weight.size(0), head = attn_weight.size(1), len_o = attn_weight.size(2), window = attn_weight.size(3), len_v = value.size(2);
  attn_weight = attn_weight.permute({2, 3, 0, 1}); // [len, window, batch, head]
  value = value.permute({2, 0, 1, 3}); // [len, batch, head, dim]
  grad_o = grad_o.permute({2, 0, 1, 3}); // [len, batch, head, dim]
  auto d_attn_weight = torch::zeros_like(attn_weight);
  auto d_value = torch::zeros_like(value);
  auto begin_offset = len_v - window + 1 - len_o;
  for(int posi = 0; posi < grad_o.size(0); posi++) {
    d_attn_weight[posi] = grad_o[posi].mul(value.slice(0, posi + begin_offset, posi + begin_offset + window)).sum(-1);
    d_value.slice(0, posi + begin_offset, posi + begin_offset + window) += grad_o[posi] * attn_weight[posi].unsqueeze(-1);
  }
  return {d_attn_weight.permute({2, 3, 0, 1}), d_value.permute({1, 2, 0, 3})};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_forward", &window_attention_forward, "Window Attention forward");
  m.def("attn_backward", &window_attention_backward, "Window Attention backward");
  m.def("value_forward", &window_value_forward, "Window Value forward");
  m.def("value_backward", &window_value_backward, "Window Value backward");
}


