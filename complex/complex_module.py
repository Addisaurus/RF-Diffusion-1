import torch
import torch.nn as nn
from torch.nn import functional as F
from tfdiff.debug_utils import shape_logger, value_logger
import numpy as np
import math


def apply_complex(F_r, F_i, X):
    """Apply complex linear transformation
    
    Args:
        F_r (nn.Linear): Real component linear transformation
        F_i (nn.Linear): Imaginary component linear transformation 
        X (torch.Tensor): Input tensor of shape [..., 2] where last dim is [real, imag]
    """
    # print("\n=== Complex Module Debug ===")
    # print(f"Input X shape: {X.shape}")
    # print(f"F_r input features: {F_r.in_features}, output features: {F_r.out_features}")
    
    # Split into real and imaginary components
    split_tensors = torch.split(X, 1, dim=-1)
    # print(f"Number of split tensors: {len(split_tensors)}")
    # print(f"Split tensor shapes: {[t.shape for t in split_tensors]}")
    
    if len(split_tensors) != 2:
        raise ValueError(f"Expected tensor with 2 components in last dimension, got {len(split_tensors)}")
        
    # Remove the extra dimension from split
    X_r, X_i = [x.squeeze(dim=-1) for x in split_tensors]
    # print(f"X_r shape after squeeze: {X_r.shape}")
    # print(f"X_i shape after squeeze: {X_i.shape}")
    
    # Apply complex multiplication:
    # (a + bi)(x + yi) = (ax - by) + (ay + bx)i
    real_output = F_r(X_r) - F_i(X_i)
    imag_output = F_r(X_i) + F_i(X_r)
    
    # print(f"Output shape before stack: {real_output.shape}")
    
    # Stack back into complex form
    result = torch.stack((real_output, imag_output), dim=-1)
    # print(f"Final output shape: {result.shape}")
    return result

def apply_complex_sep(F_r, F_i, X):
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]
    return torch.stack((F_r(X_r), F_i(X_i)), dim=-1)

@torch.jit.script
def complex_mul(X, Y):
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]
    Y_r, Y_i = [y.squeeze(dim=-1) for y in torch.split(Y, 1, dim=-1)]
    Z_r = torch.mul(X_r, Y_r) - torch.mul(X_i, Y_i)
    Z_i = torch.mul(X_r, Y_i) + torch.mul(X_i, Y_r)
    return torch.stack((Z_r, Z_i), dim=-1)

@torch.jit.script
def complex_bmm(X, Y):
    #print(f"complex_bmm input shapes: X:{X.shape}, Y:{Y.shape}")
    """Ensure consistent ordering in batched matrix multiply"""
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]  # Real and imaginary parts
    Y_r, Y_i = [y.squeeze(dim=-1) for y in torch.split(Y, 1, dim=-1)]
    
    # Maintain consistent dimension order
    Z_r = torch.bmm(X_r, Y_r) - torch.bmm(X_i, Y_i)  
    Z_i = torch.bmm(X_r, Y_i) + torch.bmm(X_i, Y_r)
    
    return torch.stack((Z_r, Z_i), dim=-1)  # Return [batch, out_dim1, out_dim2, 2]

@torch.jit.script
def complex_softmax(X):
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]
    return torch.stack((F.softmax(X_r, dim=-1), F.softmax(X_i, dim=-1)), dim=-1)

@torch.jit.script
def transpose_qkv(x, num_heads: int):
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1, 2)
    x = x.transpose(1, 2)
    return x.reshape(-1, x.shape[2], x.shape[3], 2)

@torch.jit.script
def transpose_output(x, num_heads: int):
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2], 2)
    x = x.transpose(1, 2)
    return x.reshape(x.shape[0], x.shape[1], -1, 2)


class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, X):
        device = X.device
        dtype = X.dtype
        mask = torch.ones(*X.shape[-3:], device=device, dtype=dtype)
        mask = F.dropout1d(mask, p=0.5, training=self.training)
        return torch.mul(X, mask)


class ComplexGELU(nn.Module):
    def __init__(self, approximate='none'):
        super().__init__()
        self.gelu_r = nn.GELU(approximate)
        self.gelu_i = nn.GELU(approximate)
    
    def forward(self, X):
        return apply_complex_sep(self.gelu_r, self.gelu_i, X)


class ComplexSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu_r = nn.SiLU()
        self.silu_i = nn.SiLU()

    def forward(self, X):
        return apply_complex_sep(self.silu_r, self.silu_i, X)


class ComplexReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_r = nn.ReLU()
        self.relu_i = nn.ReLU()

    def forward(self, X):
        return apply_complex_sep(self.relu_r, self.relu_i, X)


class ComplexAvgPool3d(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.avg_pool_r = nn.AvgPool3d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.avg_pool_i = nn.AvgPool3d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, X):
        return apply_complex_sep(self.avg_pool_r, self.avg_pool_i, X)


class ComplexFlatten(nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.flt_r = nn.Flatten(start_dim=start_dim, end_dim=end_dim)
        self.flt_i = nn.Flatten(start_dim=start_dim, end_dim=end_dim)

    def forward(self, X):
        return apply_complex_sep(self.flt_r, self.flt_i, X)


class NaiveComplexBatchNorm3d(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(NaiveComplexBatchNorm3d, self).__init__()
        self.bn_r = nn.BatchNorm3d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.bn_i = nn.BatchNorm3d(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, X):
        return apply_complex_sep(self.bn_r, self.bn_i, X)


class NaiveComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(NaiveComplexLayerNorm, self).__init__()
        self.ln_r = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
        self.ln_i = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, X):
        return apply_complex_sep(self.ln_r, self.ln_i, X)


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.l_r = nn.Linear(in_features, out_features, bias=bias, dtype=torch.float32)
        self.l_i = nn.Linear(in_features, out_features, bias=bias, dtype=torch.float32)

    def forward(self, X):
        return apply_complex(self.l_r, self.l_i, X)


class ComplexMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=ComplexGELU, bias=True, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ComplexLinear(in_features, hidden_features, bias)
        self.act = act_layer()
        self.drop1 = ComplexDropout(dropout)
        self.fc2 = ComplexLinear(hidden_features, out_features, bias)
        self.drop2 = ComplexDropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class ComplexConv3d(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, padding, stride=1):
        super().__init__()
        self.conv_r = nn.Conv3d(
            input_channels,
            num_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dtype=torch.float32,
        )
        self.conv_i = nn.Conv3d(
            input_channels,
            num_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dtype=torch.float32,
        )

    def forward(self, X):
        return apply_complex(self.conv_r, self.conv_i, X)


class ComplexResidual3d(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, padding, stride=1):
        super().__init__()
        self.conv1 = ComplexConv3d(
            input_channels,
            num_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.conv2 = ComplexConv3d(
            num_channels,
            num_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.conv3 = ComplexConv3d(
            input_channels, num_channels, kernel_size=1, padding=0, stride=stride
        )
        self.bn1 = NaiveComplexBatchNorm3d(num_channels)
        self.bn2 = NaiveComplexBatchNorm3d(num_channels)
        self.relu1 = ComplexReLU()
        self.relu2 = ComplexReLU()

    def forward(self, X):
        Y = self.relu1(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y)) + self.conv3(X)
        return self.relu2(Y)


# [32 32 10 10 3] -> [32 10 32*10*3]
class ComplexSegment(nn.Module):
    def __init__(self, input_channels, seg_channels, seg_size):
        super().__init__()
        self.seg_conv = ComplexResidual3d(
            input_channels,
            seg_channels,
            kernel_size=seg_size,
            padding=(0, 0, 0),
            stride=seg_size,
        )
        self.flt = ComplexFlatten(start_dim=2, end_dim=-1)

    def forward(self, X):
        Y = self.seg_conv(X)
        Y = Y.transpose(1, 2)
        Y = self.flt(Y)
        return Y


class Complex2Real(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, X):
        X = self.linear1(X)
        X = self.linear2(F.relu(X))
        return X.squeeze(dim=-1)

class ComplexDotProductAttention(nn.Module):
    def __init__(self, dropout, chunk_size=32):
        super().__init__()
        self.dropout = ComplexDropout(dropout)
        self.chunk_size = chunk_size

    def forward(self, queries, keys, values):
        shape_logger.debug(f"\n=== DotProductAttention Input Shapes ===")
        shape_logger.debug(f"Q:{queries.shape}, K:{keys.shape}, V:{values.shape}")
        
        batch_size, seq_len, feature_dim = queries.shape[:-1]
        
        # Always maintain consistent ordering of dimensions
        output = torch.zeros_like(values)
        
        # Process sequence in chunks
        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            
            # Get current chunk of queries
            q_chunk = queries[:, chunk_start:chunk_end]  # [batch, chunk, feature, 2]
            k_trans = keys.transpose(1, 2)  # [batch, feature, seq, 2]
            
            # Calculate attention scores 
            scores = complex_bmm(q_chunk, k_trans) / math.sqrt(feature_dim)  
            
            # Apply softmax and dropout - maintain shape consistency
            attention_weights = complex_softmax(scores)  
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values - enforce consistent ordering
            chunk_output = complex_bmm(attention_weights, values)  
            
            # Store chunk output
            output[:, chunk_start:chunk_end] = chunk_output
            
        return output

class ComplexMultiHeadAttention(nn.Module):
    def __init__(self, query_size, hidden_dim, num_heads, dropout, key_size=None, 
                 value_size=None, bias=False, chunk_size=32):  # Add chunk_size parameter
        super().__init__()
        key_size = key_size or query_size
        value_size = value_size or query_size
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attention = ComplexDotProductAttention(dropout=dropout, chunk_size=chunk_size)
        
        self.hidden_per_head = hidden_dim // num_heads
        assert self.hidden_per_head * num_heads == hidden_dim, \
            "Hidden dimension must be divisible by number of heads"
        
        self.w_q = ComplexLinear(query_size, hidden_dim, bias=bias)
        self.w_k = ComplexLinear(key_size, hidden_dim, bias=bias)
        self.w_v = ComplexLinear(value_size, hidden_dim, bias=bias)
        self.w_o = ComplexLinear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, queries, keys, values):
        batch_size, seq_len = queries.shape[:2]
        
        # Linear transformations with consistent shape ordering
        queries = self.w_q(queries)  # [batch, seq, hidden, 2]
        keys = self.w_k(keys)        # [batch, seq, hidden, 2]
        values = self.w_v(values)    # [batch, seq, hidden, 2]
        
        def reshape_for_heads(x):
            # Maintain consistent dimension ordering throughout
            return (x.reshape(batch_size, seq_len, self.num_heads, self.hidden_per_head, 2)
                    .permute(0, 2, 1, 3, 4)  # [batch, heads, seq, head_dim, 2]
                    .reshape(batch_size * self.num_heads, seq_len, self.hidden_per_head, 2))
        
        # Reshape maintaining dimension order
        queries = reshape_for_heads(queries)
        keys = reshape_for_heads(keys)
        values = reshape_for_heads(values)
        
        # Apply attention
        output = self.attention(queries, keys, values)
        
        # Reshape back with consistent ordering
        output = (output.reshape(batch_size, self.num_heads, seq_len, self.hidden_per_head, 2)
                 .permute(0, 2, 1, 3, 4)  # [batch, seq, heads, head_dim, 2]
                 .reshape(batch_size, seq_len, self.hidden_dim, 2))
        
        # Final linear transformation
        output = self.w_o(output)
        
        return output


class ComplexPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout, max_len=10000):
        super(ComplexPositionalEncoding, self).__init__()
        self.dropout = ComplexDropout(dropout)
        pcode = torch.zeros((1, max_len, hidden_dim, 2), dtype=torch.float32)
        pos = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, hidden_dim, dtype=torch.float32) / hidden_dim
        )
        pcode[:, :, :, 0] = torch.cos(pos)
        pcode[:, :, :, 1] = torch.sin(pos)
        self.register_buffer("pcode", pcode, persistent=False)

    def forward(self, X):
        X = complex_mul(X, self.pcode[:, : X.shape[1], :, :].to(X.device))
        Y = self.dropout(X)
        return Y


class PositionWiseFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.linear1 = ComplexLinear(input_dim, hidden_dim)
        self.relu = ComplexReLU()
        self.linear2 = ComplexLinear(hidden_dim, output_dim)

    def forward(self, X):
        Y = self.linear2(self.relu(self.linear1(X)))
        return Y


class ComplexAddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(ComplexAddNorm, self).__init__(**kwargs)
        self.dropout = ComplexDropout(dropout)
        self.ln = NaiveComplexLayerNorm(normalized_shape)

    def forward(self, X, Y):
        Y = self.ln(self.dropout(Y) + X)
        return Y


class ComplexEncoderBlock(nn.Module):
    def __init__(
        self,
        key_dim,
        query_dim,
        value_dim,
        hidden_dim,
        norm_shape,
        ffn_input_dim,
        ffn_hidden_dim,
        num_heads,
        dropout,
        use_bias=False,
        **kwargs
    ):
        super(ComplexEncoderBlock, self).__init__(**kwargs)
        self.attention = ComplexMultiHeadAttention(
            key_dim, query_dim, value_dim, hidden_dim, num_heads, dropout, use_bias
        )
        self.addnorm1 = ComplexAddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_input_dim, ffn_hidden_dim, ffn_hidden_dim)
        self.addnorm2 = ComplexAddNorm(norm_shape, dropout)

    def forward(self, X):
        Y = self.attention(X, X, X)
        Z = self.addnorm1(X, Y)
        return self.addnorm2(Z, self.ffn(Y))


class ComplexTransformerEncoder(nn.Module):
    def __init__(
        self,
        key_dim,
        query_dim,
        value_dim,
        hidden_dim,
        norm_shape,
        ffn_input_dim,
        ffn_hidden_dim,
        num_heads,
        num_layers,
        dropout,
        use_bias=False,
        **kwargs
    ):
        super(ComplexTransformerEncoder, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.pos_encoding = ComplexPositionalEncoding(hidden_dim, dropout)
        self.blks = nn.Sequential()
        for n in range(num_layers):
            self.blks.add_module(
                "Block" + str(n),
                ComplexEncoderBlock(
                    key_dim,
                    query_dim,
                    value_dim,
                    hidden_dim,
                    norm_shape,
                    ffn_input_dim,
                    ffn_hidden_dim,
                    num_heads,
                    dropout,
                    use_bias,
                ),
            )

    def forward(self, X, *args):
        X = self.pos_encoding(X * math.sqrt(self.hidden_dim))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
