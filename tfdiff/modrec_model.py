import torch
from torch import nn
import complex.complex_module as cm
from .conditioning import ConditioningManager
import math
from tfdiff.memory_utils import track_memory, clear_memory

def init_weight_norm(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def init_weight_zero(module):
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def init_weight_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

@torch.jit.script
def modulate(x, shift, scale):
    """Modulate input x with shift and scale parameters.
    
    Args:
        x: Input tensor [B, seq_len, hidden_dim, 2]
        shift: Shift tensor [B, seq_len, hidden_dim, 2] or [B, hidden_dim, 2]
        scale: Scale tensor [B, seq_len, hidden_dim, 2] or [B, hidden_dim, 2]
    """
    if shift.shape != x.shape:
        print(f"Broadcasting modulation params - x:{x.shape}, shift:{shift.shape}, scale:{scale.shape}")
        # Expand if needed
        if len(shift.shape) == 3:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
        shift = shift.expand_as(x)
        scale = scale.expand_as(x)
    
    # Clamp scale to prevent explosion
    scale = torch.clamp(scale, -5.0, 5.0)
    
    # Add numerical stability
    x = torch.nan_to_num(x, nan=0.0)
    result = x * (1 + scale) + shift
    
    # Final safety check
    result = torch.nan_to_num(result, nan=0.0)
    return result

class DiffusionEmbedding(nn.Module):
    def __init__(self, max_step, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_step, embed_dim), persistent=False)
        self.projection = nn.Sequential(
            cm.ComplexLinear(embed_dim, hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, hidden_dim, bias=True),
        )
        self.hidden_dim = hidden_dim
        self.apply(init_weight_norm)

    def forward(self, t):
        if t.dtype in [torch.int32, torch.int64]:
            x = self.embedding[t]
        else:
            x = self._lerp_embedding(t)
        return self.projection(x)

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_step, embed_dim):
        steps = torch.arange(max_step).unsqueeze(1)  # [T, 1]
        dims = torch.arange(embed_dim).unsqueeze(0)  # [1, E]
        table = steps * torch.exp(-math.log(max_step)
                                  * dims / embed_dim)  # [T, E]
        table = torch.view_as_real(torch.exp(1j * table))
        return table

class MLPConditionEmbedding(nn.Module):
    def __init__(self, cond_dim, hidden_dim=256):
        super().__init__()
        self.projection = nn.Sequential(
            cm.ComplexLinear(cond_dim, hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, hidden_dim*4, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim*4, hidden_dim, bias=True),
        )
        self.apply(init_weight_norm)

    def forward(self, c):
        return self.projection(c)

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, input_dim, hidden_dim):
        super().__init__()
        # print(f"\n=== PositionEmbedding Init ===")
        # print(f"max_len: {max_len}, input_dim: {input_dim}, hidden_dim: {hidden_dim}")
        self.register_buffer('embedding', self._build_embedding(
            max_len, hidden_dim), persistent=False)
        self.projection = cm.ComplexLinear(input_dim, hidden_dim)
        # print(f"Embedding shape: {self.embedding.shape}")
        self.apply(init_weight_xavier)

    def forward(self, x): 
        # print("\n=== PositionEmbedding Forward Pass ===")
        # print(f"Input x shape: {x.shape}")  # Should be [B, N, 2]
        batch_size, seq_len, _ = x.shape
        
        # Reshape to [B*N, 1, 2] for the linear projection
        x = x.reshape(-1, 1, 2)
        # print(f"Reshaped input shape: {x.shape}")
        
        # Apply projection
        x = self.projection(x)  # Now [B*N, hidden_dim, 2]
        # print(f"After projection shape: {x.shape}")
        
        # Reshape back to [B, N, hidden_dim, 2]
        x = x.reshape(batch_size, seq_len, -1, 2)
        # print(f"Reshaped back: {x.shape}")
        
        # Get embedding and ensure it's on the right device
        embedding = self.embedding.to(x.device)  # [N, hidden_dim, 2]
        # print(f"Embedding shape: {embedding.shape}")
        
        # Multiply with positional embedding 
        result = cm.complex_mul(x, embedding[:seq_len, :, :].unsqueeze(0))
        # print(f"Final output shape: {result.shape}")
        return result

    def _build_embedding(self, max_len, hidden_dim):
        steps = torch.arange(max_len).unsqueeze(1)  # [P,1]
        dims = torch.arange(hidden_dim).unsqueeze(0)  # [1,E]
        table = steps * torch.exp(-math.log(max_len) * dims / hidden_dim)  # [P,E]
        table = torch.view_as_real(torch.exp(1j * table))  # [P, E, 2]
        return table

class DiA(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, params, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = cm.NaiveComplexLayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.attn = cm.ComplexMultiHeadAttention(hidden_dim, hidden_dim, num_heads, dropout, bias=True, chunk_size=params.chunk_size)
        self.norm2 = cm.NaiveComplexLayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            cm.ComplexLinear(hidden_dim, mlp_hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(mlp_hidden_dim, hidden_dim, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 6*hidden_dim, bias=True)
        )
        #self.use_checkpointing = True
        self.use_checkpointing = False  # Temporarily disable for debugging

    def _forward(self, x, c):
        print(f"\n=== DiA Layer Processing ===")
        print(f"Input x: {x.shape}")
        print(f"Input c: {c.shape}")
        
        batch_size, seq_len, hidden_dim, _ = x.shape
        
        # Ensure condition has proper shape before modulation
        if len(c.shape) == 3:  # [B, H, 2]
            c = c.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [B, seq_len, H, 2]
        print(f"Expanded condition: {c.shape}")
        
        # Get modulation parameters
        modulation = self.adaLN_modulation(c)
        print(f"Modulation output: {modulation.shape}")
        
        chunks = modulation.chunk(6, dim=2)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks
        print(f"Modulation chunks: {shift_msa.shape}")
        
        # Pre-normalize inputs
        normed_x = self.norm1(x)
        
        # Modulate normalized input
        mod_x = modulate(normed_x, shift_msa, scale_msa)
        
        # Apply attention with consistent shapes
        attn_out = self.attn(mod_x, mod_x, mod_x)
        print(f"Attention output shape: {attn_out.shape}")
        x = x + gate_msa * attn_out
        
        # MLP path with consistent shapes
        normed_x = self.norm2(x)
        mod_x = modulate(normed_x, shift_mlp, scale_mlp)
        mlp_out = self.mlp(mod_x)
        x = x + gate_mlp * mlp_out
        
        return x
        
    def forward(self, x, c):
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, 
                x, 
                c,
                use_reentrant=False
            )
        return self._forward(x, c)

class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm = cm.NaiveComplexLayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.linear = cm.ComplexLinear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 2*hidden_dim, bias=True)
        )
        self.apply(init_weight_zero)

    def forward(self, x, c):
        print("\n=== Final Layer Processing ===")
        print(f"Input shapes - x:{x.shape}, c:{c.shape}")
        print(f"Initial input stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}")
        
        batch_size, seq_len, hidden_dim, _ = x.shape
        
        # Get modulation parameters and expand
        modulation = self.adaLN_modulation(c)  # [B, 2*hidden_dim, 2]
        shift, scale = modulation.chunk(2, dim=1)  # Each is [B, hidden_dim, 2]
        shift = shift.unsqueeze(1).expand(-1, seq_len, -1, -1)
        scale = scale.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        print(f"Modulation stats - shift: [{shift.min().item():.4f}, {shift.max().item():.4f}], scale: [{scale.min().item():.4f}, {scale.max().item():.4f}]")
        
        # Apply layer norm and modulation
        x = self.norm(x)
        print(f"After norm stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}")
        
        x = modulate(x, shift, scale)
        print(f"After modulation stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}")
        
        # Apply linear layer
        x = x.reshape(batch_size * seq_len, hidden_dim, 2)
        x = self.linear(x)  # [B*seq_len, out_dim, 2]
        x = x.reshape(batch_size, seq_len, -1, 2)  # [B, seq_len, out_dim, 2]
        
        # Remove extra dimension if out_dim is 1
        if x.shape[2] == 1:
            x = x.squeeze(2)  # [B, seq_len, 2]
                
        print(f"After linear stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}")
        
        # Define target range
        target_min = -4.0
        target_max = 4.0
        target_range = target_max - target_min
        
        # Use a safe normalization approach
        x_std = torch.std(x)
        if x_std < 1e-8:  # If std is too small
            x_std = torch.ones_like(x_std)
            
        x_mean = torch.mean(x)
        x = (x - x_mean) / (x_std + 1e-8)
        
        # Scale to target range using tanh
        x = torch.tanh(x) * (target_range/2.0)  # tanh output is [-1, 1], so multiply by half the range
        # Center at target midpoint
        x = x + (target_max + target_min)/2.0
        
        print(f"Final output stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
        print(f"Output shape: {x.shape}")
        
        # Add final sanity check
        if not torch.isfinite(x).all():
            print("WARNING: Non-finite values in output!")
            # Replace NaN/inf with finite values
            x = torch.nan_to_num(x, nan=0.0, posinf=target_max, neginf=target_min)
            
        return x

class tfdiff_ModRec(nn.Module):
    def __init__(self, params):
        super().__init__()
        # print("\n=== Initializing ModRec Model ===")
        # print(f"Input dim: {params.input_dim}")
        # print(f"Hidden dim: {params.hidden_dim}")
        # print(f"Target sequence length: {params.target_sequence_length}")
        
        self.params = params
        self.input_dim = params.input_dim  
        self.hidden_dim = params.hidden_dim
        self.num_heads = params.num_heads
        
        # Initialize conditioning manager first
        self.conditioning_manager = ConditioningManager(params)
        
        # Position embedding for sequence of I/Q samples
        self.p_embed = PositionEmbedding(
            params.target_sequence_length,  # Use target length here
            params.input_dim,    
            params.hidden_dim    
        )
            
        self.t_embed = DiffusionEmbedding(
            params.max_step,
            params.embed_dim,
            params.hidden_dim
        )
            
        self.c_embed = MLPConditionEmbedding(
            self.conditioning_manager.conditioning_dim,
            params.hidden_dim
        )
        
        # print(f"Conditioning dimension: {self.conditioning_manager.conditioning_dim}")
        
        # Attention blocks
        self.blocks = nn.ModuleList([
            DiA(
                self.hidden_dim,
                self.num_heads,
                params.dropout,
                params,
                params.mlp_ratio
            ) for _ in range(params.num_block)
        ])
        
        # Final layer goes back to input dimension
        self.final_layer = FinalLayer(
            self.hidden_dim,
            1  # Set output dimension to 1
        )

    def forward(self, x, t, c):
        print("\n=== Model Forward Pass Start ===")
        print(f"Input x shape: {x.shape}")
        
        # Embed positions
        x = self.p_embed(x)
        print(f"After position embedding: {x.shape}")
        
        # Embed diffusion timestep
        t = self.t_embed(t)
        print(f"Timestep embedding: {t.shape}")
        
        # Embed conditioning info
        c = self.c_embed(c)
        print(f"Condition embedding: {c.shape}")
        
        # Combine condition with diffusion step
        c = c + t
        
        # Process through attention blocks
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            # print(f"After block {i}: {x.shape}")
            
        # Generate final output
        x = self.final_layer(x, c)

        # Ensure output shape is correct [B, seq_len, 2]
        if len(x.shape) == 4 and x.shape[2] == 1:
            x = x.squeeze(2)
            
        print(f"Model output shape: {x.shape}")
        return x