import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp
from .combined_model import CombinedModel

# GAT  model
class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, combined_steps=1):
        super(GATNet, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt1 = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=8)
        self.fc_xt1 = nn.Linear(n_filters, output_dim)

        # CombinedModel for fusion
        self.combined_steps = combined_steps
        self.combined = CombinedModel(feature_dim=output_dim, num_heads=4, dropout_rate=dropout)

        # combined layers
        self.fc1 = nn.Linear(output_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.dropout(x, p=self.dropout.p, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=self.dropout.p, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        # protein input feed-forward:
        target = data.target
        embedded_xt = self.embedding_xt(target).permute(0, 2, 1).contiguous()
        conv_xt = self.conv_xt1(embedded_xt)
        conv_xt = self.relu(conv_xt)
        xt = torch.max(conv_xt, dim=2)[0] # Global Max Pooling
        xt = self.fc_xt1(xt)

        # Fusion with CombinedModel
        batch_size = x.size(0)
        feat_dim = x.size(1)
        h_nodes = torch.stack([x, xt], dim=1).view(batch_size * 2, feat_dim)
        mask_ligand = torch.arange(batch_size * 2, device=h_nodes.device) % 2
        batch_map = torch.arange(batch_size, device=h_nodes.device).repeat_interleave(2)

        h_updated = h_nodes
        for _ in range(self.combined_steps):
            h_updated = self.combined(h_updated, mask_ligand, batch_map)

        h_updated = h_updated.view(batch_size, 2, feat_dim)
        protein_final = h_updated[:, 0, :]
        ligand_final = h_updated[:, 1, :]

        # Concat and classify
        xc = torch.cat((protein_final, ligand_final), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out