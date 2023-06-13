import os

from torch.autograd import Function

from torch.utils.cpp_extension import load

file_dir = os.path.dirname(os.path.abspath(__file__))
window_attention_cuda = load(
    name="window_attention_cuda", 
    sources=[os.path.join(file_dir, "cuda/window_attention_cuda.cpp"), os.path.join(file_dir, "cuda/window_attention_cuda_kernel_cache.cu")], 
    verbose=True)

window_attention_cpp = load(name="window_attention", sources=[os.path.join(file_dir, "cpp/window_attention.cpp")], verbose=True)

class WindowAttentionFunction(Function):
    @staticmethod
    def forward(ctx, query, key, window):
        if query.is_cuda:
            outputs = window_attention_cuda.attn_forward(query, key, window)
        else:
            outputs = window_attention_cpp.attn_forward(query, key, window)

        variables = [query, key]
        ctx.save_for_backward(*variables)
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out.is_cuda:
            d_query, d_key = window_attention_cuda.attn_backward(
                grad_out, *ctx.saved_tensors)
        else:
            d_query, d_key = window_attention_cpp.attn_backward(
                grad_out, *ctx.saved_tensors)

        return d_query, d_key, None

class WindowValueFunction(Function):
    @staticmethod
    def forward(ctx, attn_weight, value):
        if attn_weight.is_cuda:
            outputs = window_attention_cuda.value_forward(attn_weight, value)
        else:
            outputs = window_attention_cpp.value_forward(attn_weight, value)

        variables = [attn_weight, value]
        ctx.save_for_backward(*variables)
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out.is_cuda:
            d_attn_weight, d_value = window_attention_cuda.value_backward(
                grad_out, *ctx.saved_tensors)
            
        else:
            d_attn_weight, d_value = window_attention_cpp.value_backward(
                grad_out, *ctx.saved_tensors)
            
        return d_attn_weight, d_value
