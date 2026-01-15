import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from .combined_model import CombinedModel

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, combined_steps=1):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(n_filters, output_dim)

        # CombinedModel for fusion
        self.combined_steps = combined_steps
        self.combined = CombinedModel(feature_dim=output_dim, num_heads=4, dropout_rate=dropout)

        # combined layers
        self.fc1 = nn.Linear(output_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        # Graph branch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=self.dropout.p, training=self.training)

        # Sequence branch
        embedded_xt = self.embedding_xt(target).permute(0, 2, 1).contiguous()
        conv_xt = self.conv_xt_1(embedded_xt)
        xt = torch.max(conv_xt, dim=2)[0] # Global Max Pooling
        xt = self.fc1_xt(xt)

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