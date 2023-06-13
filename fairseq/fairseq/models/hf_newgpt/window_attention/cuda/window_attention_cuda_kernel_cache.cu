#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define PATCH 32

namespace {
template <typename scalar_t>
__global__ void window_attention_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> query,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> key,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> output,
    const int begin_offset) {
  int batch = blockIdx.x, posi = blockIdx.y * PATCH + threadIdx.x, window_start = blockIdx.z * PATCH;

  __shared__ scalar_t shared_query[PATCH][PATCH];
  __shared__ scalar_t shared_key[2 * PATCH - 1][PATCH];
  scalar_t out = 0;
  for(int i = 0; i < (query.size(2) + PATCH - 1) / PATCH; i++) {
    if(i * PATCH + threadIdx.y < query.size(2) && posi < query.size(1)) {
      shared_query[threadIdx.x][threadIdx.y] = query[batch][posi][i * PATCH + threadIdx.y];
      shared_key[(int)threadIdx.x << 1][threadIdx.y] = key[batch][window_start + begin_offset + posi + threadIdx.x][i * PATCH + threadIdx.y];
      if(threadIdx.x < PATCH - 1) {
        shared_key[((int)threadIdx.x << 1) + 1][threadIdx.y] = key[batch][window_start + begin_offset + posi + threadIdx.x + 1][i * PATCH + threadIdx.y];
      }
    } else {
      shared_query[threadIdx.x][threadIdx.x] = 0;
      shared_key[(int)threadIdx.x << 1][threadIdx.y] = 0;
      if(threadIdx.x < PATCH - 1) {
        shared_key[((int)threadIdx.x << 1) + 1][threadIdx.y] = 0;
      }
    }
    __syncthreads();
    for(int j = 0; j < PATCH; j++) {
      out += shared_query[threadIdx.x][j] * shared_key[threadIdx.x + threadIdx.y][j];
    }
    __syncthreads();
  }
  if(posi < output.size(1) && window_start + threadIdx.y < output.size(2)) {
    output[batch][posi][window_start + threadIdx.y] = out;
  }
}
} // namespace

torch::Tensor window_attention_cuda_forward(
    torch::Tensor query, // [batch, head, len, dim]
    torch::Tensor key,
    int window) {
  int batch = query.size(0), head = query.size(1), len_q = query.size(2), len_k = key.size(2), dim = query.size(3);
  query = query.view({-1, len_q, dim}); // [batch * head, len, dim]
  key = key.view({-1, len_k, dim});
  auto output = torch::zeros({batch * head, len_q, window}, query.device());
  auto begin_offset = len_k - window + 1 - len_q;
  
  const dim3 threads(PATCH, PATCH);
  const dim3 blocks(batch * head, (len_q + PATCH -1) / PATCH, (window + PATCH - 1) / PATCH);
  AT_DISPATCH_FLOATING_TYPES(query.type(), "window_attention_forward_cuda", ([&] {
    window_attention_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        query.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        key.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        output.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        (const int)begin_offset);
  }));
  output = output.view({batch, head, len_q, window});
  return output;
}

namespace {
template <typename scalar_t>
__global__ void query_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> key, // [batch * head, len_k, dim]
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_o, // [batch * head, len_q, window]
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> d_query, // [batch * head, len_q, dim]
    const int begin_offset) {
  int batch = blockIdx.x, posi = blockIdx.y * PATCH + threadIdx.x, dim = blockIdx.z * PATCH + threadIdx.y;

  __shared__ scalar_t shared_grad_o[PATCH][PATCH];
  __shared__ scalar_t shared_key[2 * PATCH - 1][PATCH];
  scalar_t out = 0;
  for(int i = 0; i < (grad_o.size(2) + PATCH - 1) / PATCH; i++) {
    if(posi < grad_o.size(1) && i * PATCH + threadIdx.y < grad_o.size(2)) {
      shared_grad_o[threadIdx.x][threadIdx.y] = grad_o[batch][posi][i * PATCH + threadIdx.y];
    } else {
      shared_grad_o[threadIdx.x][threadIdx.y] = 0;
    }
    if(posi < grad_o.size(1) && dim < key.size(2)) {
      shared_key[(int)threadIdx.x << 1][threadIdx.y] = key[batch][begin_offset + i * PATCH + posi + threadIdx.x][dim];
      if(threadIdx.x < PATCH - 1) {
        shared_key[((int)threadIdx.x << 1) + 1][threadIdx.y] = key[batch][begin_offset + i * PATCH + posi + threadIdx.x + 1][dim];
      }
    } else {
      shared_key[(int)threadIdx.x << 1][threadIdx.y] = 0;
      if(threadIdx.x < PATCH - 1) {
        shared_key[((int)threadIdx.x << 1) + 1][threadIdx.y] = 0;
      }
    }
    __syncthreads();
    for(int j = 0; j < PATCH; j++) {
      out += shared_grad_o[threadIdx.x][j] * shared_key[threadIdx.x + j][threadIdx.y];
    }
    __syncthreads();
  }
  if(posi < d_query.size(1) && dim < d_query.size(2)) {
    d_query[batch][posi][dim] = out;
  }
}

template <typename scalar_t>
__global__ void key_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> query,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_o,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> d_key,
    const int begin_offset) {
  int batch = blockIdx.x, posi = blockIdx.y * PATCH + threadIdx.x, dim = blockIdx.z * PATCH + threadIdx.y;
  int window = grad_o.size(2);
  __shared__ scalar_t shared_grad_o[PATCH][PATCH];
  __shared__ scalar_t shared_query[2 * PATCH - 1][PATCH];
  scalar_t out = 0;
  for(int i = 0; i < (window + PATCH - 1) / PATCH; i++) {
    int posi_biased = posi - window + 1;
    int offset = i * PATCH + threadIdx.y;
    int grad_o_posi = posi_biased + offset;
    int query_posi = posi_biased + i * PATCH + threadIdx.x;
    if(grad_o_posi >= 0 && grad_o_posi < grad_o.size(1) && window - 1 - offset >= 0) {
      shared_grad_o[threadIdx.x][threadIdx.y] = grad_o[batch][grad_o_posi][window - 1 - offset];
    } else {
      shared_grad_o[threadIdx.x][threadIdx.y] = 0;
    }
    if(query_posi >= 0 && query_posi < query.size(1) && dim < query.size(2)) {
      shared_query[(int)threadIdx.x << 1][threadIdx.y] = query[batch][query_posi][dim];
    } else {
      shared_query[(int)threadIdx.x << 1][threadIdx.y] = 0;
    }
    if(threadIdx.x < PATCH - 1) {
      if(query_posi + 1 >= 0 && query_posi + 1 < query.size(1) && dim < query.size(2)) {
        shared_query[((int)threadIdx.x << 1) + 1][threadIdx.y] = query[batch][query_posi + 1][dim];
      } else {
        shared_query[((int)threadIdx.x << 1) + 1][threadIdx.y] = 0;
      }
    }
    __syncthreads();
    for(int j = 0; j < PATCH; j++) {
      out += shared_grad_o[threadIdx.x][j] * shared_query[threadIdx.x + j][threadIdx.y];
    }
    __syncthreads();
  }
  if(posi < d_key.size(1) && dim < d_key.size(2)) {
    d_key[batch][posi + begin_offset][dim] = out;
  }
}
} // namespace

std::vector<torch::Tensor> window_attention_cuda_backward(
    torch::Tensor grad_o,
    torch::Tensor query,
    torch::Tensor key) {
  int batch = query.size(0), head = query.size(1), dim = query.size(3), window = grad_o.size(3);
  int len_q = query.size(2), len_k = key.size(2);
  query = query.view({-1, len_q, dim}); // [batch * head, len, dim]
  key = key.view({-1, len_k, dim});
  grad_o = grad_o.view({-1, len_q, window}); // [batch * head, len, window]
  auto d_query = torch::zeros_like(query);
  auto d_key = torch::zeros_like(key);
  auto begin_offset = len_k - window + 1 - len_q;

  const dim3 threads(PATCH, PATCH);
  const dim3 blocks_q(batch * head, (len_q + PATCH - 1) / PATCH, (dim + PATCH - 1) / PATCH);
  const dim3 blocks_k(batch * head, (len_q + window + PATCH - 2) / PATCH, (dim + PATCH - 1) / PATCH);
  AT_DISPATCH_FLOATING_TYPES(query.type(), "window_attention_backward_cuda", ([&] {
    query_backward_kernel<scalar_t><<<blocks_q, threads>>>(
        key.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        grad_o.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        d_query.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        (const int)begin_offset);
    key_backward_kernel<scalar_t><<<blocks_k, threads>>>(
        query.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        grad_o.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        d_key.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        (const int)begin_offset);
  }));
  return {d_query.view({batch, head, len_q, dim}), d_key.view({batch, head, len_k, dim})};
}

namespace {
template <typename scalar_t>
__global__ void window_value_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> attn_weight,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> value,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> output,
    const int begin_offset) {
  int batch = blockIdx.x, posi = blockIdx.y * PATCH + threadIdx.x, dim = blockIdx.z * PATCH + threadIdx.y;

  __shared__ scalar_t shared_attn_weight[PATCH][PATCH];
  __shared__ scalar_t shared_value[2 * PATCH - 1][PATCH];

  scalar_t out = 0;
  for(int i = 0; i < (attn_weight.size(2) + PATCH - 1) / PATCH; i++) {
    if(posi < attn_weight.size(1) && i * PATCH + threadIdx.y < attn_weight.size(2)) {
      shared_attn_weight[threadIdx.x][threadIdx.y] = attn_weight[batch][posi][i * PATCH + threadIdx.y];
    } else {
      shared_attn_weight[threadIdx.x][threadIdx.y] = 0;
    }
    if(posi < attn_weight.size(1) && dim < value.size(2)) {
      shared_value[(int)threadIdx.x << 1][threadIdx.y] = value[batch][begin_offset + i * PATCH + posi + threadIdx.x][dim];
      if(threadIdx.x < PATCH - 1) {
        shared_value[((int)threadIdx.x << 1) + 1][threadIdx.y] = value[batch][begin_offset + i * PATCH + posi + threadIdx.x + 1][dim];
      }
    } else {
      shared_value[(int)threadIdx.x << 1][threadIdx.y] = 0;
      if(threadIdx.x < PATCH - 1) {
        shared_value[((int)threadIdx.x << 1) + 1][threadIdx.y] = 0;
      }
    }
    __syncthreads();
    for(int j = 0; j < PATCH; j++) {
      out += shared_attn_weight[threadIdx.x][j] * shared_value[threadIdx.x + j][threadIdx.y];
    }
    __syncthreads();
  }
  if(posi < output.size(1) && dim < output.size(2)) {
    output[batch][posi][dim] = out;
  }
}
} // namespace

torch::Tensor window_value_cuda_forward(
    torch::Tensor attn_weight, // [batch, head, len, window]
    torch::Tensor value) { // [batch, head, len, dim]
  const int batch = attn_weight.size(0), head = attn_weight.size(1), len_o = attn_weight.size(2), window = attn_weight.size(3), len_v = value.size(2), dim = value.size(3);
  attn_weight = attn_weight.view({-1, len_o, window}); // [batch * head, len, window]
  value = value.view({-1, len_v, dim}); // [batch * head, len, dim]
  auto output = torch::zeros({batch * head, len_o, dim}, attn_weight.device());
  auto begin_offset = len_v - window + 1 - len_o;

  const dim3 threads(PATCH, PATCH);
  const dim3 blocks(batch * head, (len_o + PATCH - 1) / PATCH, (dim + PATCH - 1) / PATCH);
  AT_DISPATCH_FLOATING_TYPES(attn_weight.type(), "window_value_forward_cuda", ([&] {
    window_value_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        attn_weight.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        value.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        output.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        (const int)begin_offset);
  }));
  output = output.view({batch, head, len_o, dim});
  return output;
}

namespace {
template <typename scalar_t>
__global__ void attn_weight_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> value,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_o,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> d_attn_weight,
    const int begin_offset) {
  int batch = blockIdx.x, posi = blockIdx.y * PATCH + threadIdx.x, window_start = blockIdx.z * PATCH;

  __shared__ scalar_t shared_grad_o[PATCH][PATCH];
  __shared__ scalar_t shared_value[2 * PATCH - 1][PATCH];
  scalar_t out = 0;
  for(int i = 0; i < (value.size(2) + PATCH - 1) / PATCH; i++) {
    if(i * PATCH + threadIdx.y < grad_o.size(2) && posi < grad_o.size(1)) {
      shared_grad_o[threadIdx.x][threadIdx.y] = grad_o[batch][posi][i * PATCH + threadIdx.y];
      shared_value[(int)threadIdx.x << 1][threadIdx.y] = value[batch][window_start + begin_offset + posi + threadIdx.x][i * PATCH + threadIdx.y];
      if(threadIdx.x < PATCH - 1) {
        shared_value[((int)threadIdx.x << 1) + 1][threadIdx.y] = value[batch][window_start + begin_offset + posi + threadIdx.x + 1][i * PATCH + threadIdx.y];
      }
    } else {
      shared_grad_o[threadIdx.x][threadIdx.x] = 0;
      shared_value[(int)threadIdx.x << 1][threadIdx.y] = 0;
      if(threadIdx.x < PATCH - 1) {
        shared_value[((int)threadIdx.x << 1) + 1][threadIdx.y] = 0;
      }
    }
    __syncthreads();
    for(int j = 0; j < PATCH; j++) {
      out += shared_grad_o[threadIdx.x][j] * shared_value[threadIdx.x + threadIdx.y][j];
    }
    __syncthreads();
  }
  if(posi < d_attn_weight.size(1) && window_start + threadIdx.y < d_attn_weight.size(2)) {
    d_attn_weight[batch][posi][window_start + threadIdx.y] = out;
  }
}

template <typename scalar_t>
__global__ void value_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> attn_weight,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_o,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> d_value,
    const int begin_offset) {
  int batch = blockIdx.x, posi = blockIdx.y * PATCH + threadIdx.x, dim = blockIdx.z * PATCH + threadIdx.y;
  int window = attn_weight.size(2);
  __shared__ scalar_t shared_attn_weight[PATCH][PATCH];
  __shared__ scalar_t shared_grad_o[2 * PATCH - 1][PATCH];
  scalar_t out = 0;
  for(int i = 0; i < (window + PATCH - 1) / PATCH; i++) {
    int posi_biased = posi - window + 1;
    int offset = i * PATCH + threadIdx.y;
    int attn_weight_posi = posi_biased + offset;
    int grad_o_posi = posi_biased + i * PATCH + threadIdx.x;
    if(attn_weight_posi >= 0 && attn_weight_posi < attn_weight.size(1) && window - 1 - offset >= 0) {
      shared_attn_weight[threadIdx.x][threadIdx.y] = attn_weight[batch][attn_weight_posi][window - 1 - offset];
    } else {
      shared_attn_weight[threadIdx.x][threadIdx.y] = 0;
    }
    if(grad_o_posi >= 0 && grad_o_posi < grad_o.size(1) && dim < grad_o.size(2)) {
      shared_grad_o[(int)threadIdx.x << 1][threadIdx.y] = grad_o[batch][grad_o_posi][dim];
    } else {
      shared_grad_o[(int)threadIdx.x << 1][threadIdx.y] = 0;
    }
    if(threadIdx.x < PATCH - 1) {
      if(grad_o_posi + 1 >= 0 && grad_o_posi + 1 < grad_o.size(1) && dim < grad_o.size(2)) {
        shared_grad_o[((int)threadIdx.x << 1) + 1][threadIdx.y] = grad_o[batch][grad_o_posi + 1][dim];
      } else {
        shared_grad_o[((int)threadIdx.x << 1) + 1][threadIdx.y] = 0;
      }
    }
    __syncthreads();
    for(int j = 0; j < PATCH; j++) {
      out += shared_attn_weight[threadIdx.x][j] * shared_grad_o[threadIdx.x + j][threadIdx.y];
    }
    __syncthreads();
  }
  if(posi < d_value.size(1) && dim < d_value.size(2)) {
    d_value[batch][posi + begin_offset][dim] = out;
  }
}
} // namespace

std::vector<torch::Tensor> window_value_cuda_backward(
    torch::Tensor grad_o,
    torch::Tensor attn_weight,
    torch::Tensor value) {
  const int batch = attn_weight.size(0), head = attn_weight.size(1), len_o = attn_weight.size(2), window = attn_weight.size(3), len_v = value.size(2), dim = value.size(3);
  attn_weight = attn_weight.view({-1, len_o, window}); // [batch * head, len, window]
  value = value.view({-1, len_v, dim}); // [batch * head, len, dim]
  grad_o = grad_o.view({-1, len_o, dim}); // [batch * head, len, dim]
  auto d_attn_weight = torch::zeros_like(attn_weight);
  auto d_value = torch::zeros_like(value);
  auto begin_offset = len_v - window + 1 - len_o;

  const dim3 threads_o(PATCH, PATCH);
  const dim3 blocks_o(batch * head, (len_o + PATCH - 1) / PATCH, (window + PATCH - 1) / PATCH);

  const dim3 threads_v(PATCH, PATCH);
  const dim3 blocks_v(batch * head, (len_o + window + PATCH - 2) / PATCH, (dim + PATCH - 1) / PATCH);
  AT_DISPATCH_FLOATING_TYPES(grad_o.type(), "window_value_backward_cuda", ([&] {
    attn_weight_backward_kernel<scalar_t><<<blocks_o, threads_o>>>(
        value.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        grad_o.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        d_attn_weight.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        (const int)begin_offset);
    value_backward_kernel<scalar_t><<<blocks_v, threads_v>>>(
        attn_weight.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        grad_o.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        d_value.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        (const int)begin_offset);
  }));
  return {d_attn_weight.view({batch, head, len_o, window}), d_value.view({batch, head, len_v, dim})};
}
