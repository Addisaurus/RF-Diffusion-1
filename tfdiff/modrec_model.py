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
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

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
    def __init__(self, hidden_dim, num_heads, dropout, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = cm.NaiveComplexLayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.attn = cm.ComplexMultiHeadAttention(hidden_dim, hidden_dim, num_heads, dropout, bias=True)
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
        
        # Enable gradient checkpointing
        self.use_checkpointing = True
        
    def forward(self, x, c):
        # Replace checkpoint usage with explicit reentrant parameter
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, 
                x, 
                c,
                use_reentrant=False  # Add this parameter
            )
        return self._forward(x, c)
        
    def _forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
            
        # Modulate input for attention
        mod_x = modulate(self.norm1(x), shift_msa, scale_msa)
        
        # Apply attention
        attn_out = self.attn(mod_x, mod_x, mod_x)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # Modulate for MLP
        mod_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mod_x)
        
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.linear = cm.ComplexLinear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 2*hidden_dim, bias=True)
        )
        self.apply(init_weight_zero)

    def forward(self, x, c):
        # print("\n=== Final Layer Shapes ===")
        # print(f"Input x shape: {x.shape}")
        
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        
        # Keep the sequence dimension
        B, N, H, _ = x.shape
        x_reshaped = x.reshape(B*N, H, 2)
        x = self.linear(x_reshaped)
        x = x.reshape(B, N, -1, 2)
        
        # print(f"Output shape: {x.shape}")
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
                params.mlp_ratio
            ) for _ in range(params.num_block)
        ])
        
        # Final layer goes back to input dimension
        self.final_layer = FinalLayer(
            self.hidden_dim,
            self.input_dim  # Output dimension should match input
        )

    def forward(self, x, t, c):
        # print("\n=== Model Forward Pass Start ===")
        # print(f"Input x shape: {x.shape}")
        
        # Embed positions
        x = self.p_embed(x)
        # print(f"After position embedding: {x.shape}")
        
        # Embed diffusion timestep
        t = self.t_embed(t)
        # print(f"Timestep embedding: {t.shape}")
        
        # Embed conditioning info
        c = self.c_embed(c)
        # print(f"Condition embedding: {c.shape}")
        
        # Combine condition with diffusion step
        c = c + t
        
        # Process through attention blocks
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            # print(f"After block {i}: {x.shape}")
            
        # Generate final output
        x = self.final_layer(x, c)
        # print(f"Final output shape: {x.shape}")
        
        return x