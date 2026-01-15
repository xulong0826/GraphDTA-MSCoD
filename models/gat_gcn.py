import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# 引入 CombinedModel（用于最小改动下验证融合效果）
from .combined_model import CombinedModel

# GCN-CNN based model

class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, combined_steps=1):

        super(GAT_GCN, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        # 修正 in_channels 为 embed_dim
        self.conv_xt_1 = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=8)
        # 使用全局池化后将 n_filters 映射到 output_dim（比固定扁平化更稳健）
        self.fc1_xt = nn.Linear(n_filters, output_dim)

        # CombinedModel: minimal change, 固定 heads=4 用于验证
        self.combined_steps = combined_steps
        self.combined = CombinedModel(feature_dim=output_dim, num_heads=4, dropout_rate=dropout)

        # combined layers
        self.fc1 = nn.Linear(output_dim * 2, 1024)  # 256 -> 1024 when output_dim=128
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        # 图分支
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)  # (batch_size, output_dim)

        # 序列分支：embedding -> permute -> conv1d -> 全局池化 -> 映射到 output_dim
        embedded_xt = self.embedding_xt(target)               # (batch, seq_len, embed_dim)
        embedded_xt = embedded_xt.permute(0, 2, 1).contiguous()  # -> (batch, embed_dim, seq_len)
        conv_xt = self.conv_xt_1(embedded_xt)                 # -> (batch, n_filters, L_out)
        conv_xt_pool = torch.max(conv_xt, dim=2)[0]          # (batch, n_filters)
        xt = self.fc1_xt(conv_xt_pool)                       # (batch, output_dim)

        # 构造 node-level tensor 以便传入 CombinedModel（每个样本两节点：protein, ligand）
        batch_size = x.size(0)
        feat_dim = x.size(1)
        h_nodes = torch.stack([x, xt], dim=1).view(batch_size * 2, feat_dim)
        mask_ligand = torch.arange(batch_size * 2, device=h_nodes.device) % 2
        batch_map = torch.arange(batch_size, device=h_nodes.device).repeat_interleave(2)

        # 应用 CombinedModel（SPP + co-attention）
        # for _ in range(self.combined_steps):
        #     h_updated = self.combined(h_nodes, mask_ligand, batch_map)  # (batch*2, feat)

        h_updated = h_nodes
        for _ in range(self.combined_steps):
            h_updated = self.combined(h_updated, mask_ligand, batch_map)  # (batch*2, feat)

        # 恢复每样本的 protein/ligand 向量并拼接分类
        h_updated = h_updated.view(batch_size, 2, feat_dim)
        protein_final = h_updated[:, 0, :]
        ligand_final = h_updated[:, 1, :]

        xc = torch.cat((protein_final, ligand_final), dim=1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out