import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from .combined_model import CombinedModel


# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=25, output_dim=128, dropout=0.2, combined_steps=1):

        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(n_filters, output_dim)

        # CombinedModel for fusion
        self.combined_steps = combined_steps
        self.combined = CombinedModel(feature_dim=output_dim, num_heads=4, dropout_rate=dropout)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
        target = data.target

        # Graph branch
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling

        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

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