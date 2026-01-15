import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_softmax

class MultiScaleInformationBottleneck(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1): # Add dropout_rate
        super().__init__()
        self.bottleneck_dims = [16, 32, 64]
        self.bottlenecks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, output_dim),
                nn.ReLU()
            ) for bottleneck_dim in self.bottleneck_dims
        ])
        
        self.merger = nn.Linear(len(self.bottleneck_dims) * output_dim, output_dim)

        # --- MODIFICATION: Add Dropout to FFN ---
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Add dropout here
            nn.Linear(output_dim * 2, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)

        # --- NEW: small learnable scale to avoid immediate bypass by residuals ---
        self.layer_scale = nn.Parameter(torch.zeros(1))

    def forward(self, h):
        # compute each scale output (each bottleneck already maps -> output_dim)
        scale_outputs = [b(h) for b in self.bottlenecks]                   # list of (N, output_dim)
        scale_cat = torch.cat(scale_outputs, dim=-1)                       # (N, output_dim * K)
        merged = self.merger(scale_cat)                                    # (N, output_dim)

        # apply residual with small learnable scale, then norm + FFN (PreNorm style)
        h_mid = h + self.layer_scale * merged
        h_norm = self.norm(h_mid)
        correction = self.ffn(h_norm)
        out = h_mid + correction
        return out

class MultiHeadCoAttentionWithGating(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout_rate=0.1): # Add dropout_rate
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = feature_dim // num_heads
        assert self.attention_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"

        # Parameters for Protein -> Ligand attention
        self.to_q_ligand = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_k_protein = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_v_protein = nn.Linear(feature_dim, feature_dim, bias=False)
        self.ligand_gate = nn.Linear(feature_dim * 2, feature_dim)
        self.ligand_update = nn.Linear(feature_dim, feature_dim)

        # Parameters for Ligand -> Protein attention
        self.to_q_protein = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_k_ligand = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_v_ligand = nn.Linear(feature_dim, feature_dim, bias=False)
        self.protein_gate = nn.Linear(feature_dim * 2, feature_dim)
        self.protein_update = nn.Linear(feature_dim, feature_dim)
        
        # Post-update processing
        self.norm_protein = nn.LayerNorm(feature_dim)
        self.norm_ligand = nn.LayerNorm(feature_dim)
        self.ffn_protein = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate), # Add dropout here
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.ffn_ligand = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate), # Add dropout here
            nn.Linear(feature_dim * 4, feature_dim)
        )


    def forward(self, h_protein, h_ligand, protein_batch, ligand_batch):
        # 1. Robust build of attention edges: for each batch id, take cartesian product of local indices
        device = h_protein.device
        unique_batches = torch.unique(torch.cat([protein_batch, ligand_batch]))
        edge_pairs = []
        for b in unique_batches.tolist():
            p_idx = torch.nonzero(protein_batch == b, as_tuple=False).view(-1)
            l_idx = torch.nonzero(ligand_batch == b, as_tuple=False).view(-1)
            if p_idx.numel() == 0 or l_idx.numel() == 0:
                continue
            p_expand = p_idx.repeat_interleave(l_idx.numel())
            l_expand = l_idx.repeat(p_idx.numel())
            edge_pairs.append(torch.stack([p_expand, l_expand], dim=1))

        if not edge_pairs:
            return h_protein, h_ligand

        attention_edges = torch.cat(edge_pairs, dim=0).to(device)
        p_edge_idx, l_edge_idx = attention_edges[:, 0], attention_edges[:, 1]

        # --- BIDIRECTIONAL ATTENTION CALCULATION ---

        # 2.a. Protein -> Ligand (Update Ligand)
        q_ligand = self.to_q_ligand(h_ligand).view(-1, self.num_heads, self.attention_dim)
        k_protein = self.to_k_protein(h_protein).view(-1, self.num_heads, self.attention_dim)
        v_protein = self.to_v_protein(h_protein).view(-1, self.num_heads, self.attention_dim)

        attention_scores_pl = (q_ligand[l_edge_idx] * k_protein[p_edge_idx]).sum(dim=-1) / (self.attention_dim ** 0.5)
        ligand_attention = scatter_softmax(attention_scores_pl, l_edge_idx, dim=0)
        ligand_context_edges = v_protein[p_edge_idx] * ligand_attention.unsqueeze(-1)
        ligand_context = scatter_add(ligand_context_edges, l_edge_idx, dim=0, dim_size=h_ligand.size(0))
        ligand_context = ligand_context.view(-1, self.num_heads * self.attention_dim)
        
        gate_l = torch.sigmoid(self.ligand_gate(torch.cat([h_ligand, ligand_context], dim=1)))
        updated_ligand_info = self.ligand_update(ligand_context)
        h_ligand_updated = h_ligand + gate_l * updated_ligand_info

        # 2.b. Ligand -> Protein (Update Protein)
        q_protein = self.to_q_protein(h_protein).view(-1, self.num_heads, self.attention_dim)
        k_ligand = self.to_k_ligand(h_ligand).view(-1, self.num_heads, self.attention_dim)
        v_ligand = self.to_v_ligand(h_ligand).view(-1, self.num_heads, self.attention_dim)

        attention_scores_lp = (q_protein[p_edge_idx] * k_ligand[l_edge_idx]).sum(dim=-1) / (self.attention_dim ** 0.5)
        protein_attention = scatter_softmax(attention_scores_lp, p_edge_idx, dim=0)
        protein_context_edges = v_ligand[l_edge_idx] * protein_attention.unsqueeze(-1)
        protein_context = scatter_add(protein_context_edges, p_edge_idx, dim=0, dim_size=h_protein.size(0))
        protein_context = protein_context.view(-1, self.num_heads * self.attention_dim)

        gate_p = torch.sigmoid(self.protein_gate(torch.cat([h_protein, protein_context], dim=1)))
        updated_protein_info = self.protein_update(protein_context)
        h_protein_updated = h_protein + gate_p * updated_protein_info

        # 3. Final processing with LayerNorm and FFN (Transformer-style block)
        protein_final = h_protein_updated + self.ffn_protein(self.norm_protein(h_protein_updated))
        ligand_final = h_ligand_updated + self.ffn_ligand(self.norm_ligand(h_ligand_updated))

        return protein_final, ligand_final

# 综合模型：信息瓶颈和协同注意力机制结合
class CombinedModel(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout_rate=0.1): # Pass dropout_rate down
        super(CombinedModel, self).__init__()
        self.spp = MultiScaleInformationBottleneck(feature_dim, feature_dim, dropout_rate=dropout_rate)
        self.co_attention = MultiHeadCoAttentionWithGating(feature_dim, num_heads, dropout_rate=dropout_rate)

    def forward(self, h, mask_ligand, batch):
        # 1. Separate protein and ligand features
        protein_mask = (mask_ligand == 0)
        ligand_mask = (mask_ligand == 1)

        protein_features = h[protein_mask]
        ligand_features = h[ligand_mask]
        
        protein_batch_map = batch[protein_mask]
        ligand_batch_map = batch[ligand_mask]

        # 2. Apply SPP module for multi-scale feature extraction
        protein_features_spp = self.spp(protein_features)
        ligand_features_spp = self.spp(ligand_features)

        # 3. Apply fully bidirectional co-attention for feature interaction
        protein_final, ligand_final = self.co_attention(protein_features_spp, ligand_features_spp, protein_batch_map, ligand_batch_map)

        # 4. Re-assemble the updated features
        h_updated = h.clone()
        h_updated[protein_mask] = protein_final
        h_updated[ligand_mask] = ligand_final

        return h_updated
