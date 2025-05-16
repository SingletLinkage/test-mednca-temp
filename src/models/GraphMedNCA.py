import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, knn_graph
from torch_geometric.data import Data
from colorama import Fore, Back, Style
from src.utils.helper import log_message


class GrapherModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(GrapherModule, self).__init__()
        self.gat = GATConv(in_channels, out_channels // num_heads, heads=num_heads)

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        return F.relu(x)

def image_to_graph(feature_map, k=8, log_enabled=False):
    B, C, H, W = feature_map.size()
    graphs = []
    module = f"{__name__}:image_to_graph" if log_enabled else ""

    for i in range(B):
        fmap = feature_map[i]
        # log_message(f"Processing batch {i+1}/{B} with shape {fmap.shape}", "INFO", module, log_enabled)
        nodes = fmap.view(C, -1).permute(1, 0)
        # log_message(f"passing to knn graph: nodes shape {nodes.shape}", "STATUS", module, log_enabled)
        edge_index = knn_graph(nodes, k=k)
        # log_message(f"Graph {i+1}: nodes shape {nodes.shape}, edge_index shape {edge_index.shape}", "STATUS", module, log_enabled)
        graphs.append(Data(x=nodes, edge_index=edge_index))

    return graphs

class GraphMedNCA(nn.Module):
    def __init__(self, hidden_channels=16, n_channels=1, fire_rate=0.5, slice_dim=None, device=None, log_enabled=True):
        super(GraphMedNCA, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.n_channels = n_channels
        self.fire_rate = fire_rate
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_enabled = log_enabled
        
        module = f"{__name__}:init" if log_enabled else ""
        # log_message(f"Initializing GraphMedNCA with {hidden_channels} hidden channels on {self.device}", "INFO", module, log_enabled)
        
        # Graph-based perception module
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Grapher module for generating perception vectors
        self.grapher = GrapherModule(hidden_channels, hidden_channels)
        
        # Update network - standard NCA components
        self.update_net = nn.Sequential(
            nn.Conv2d(hidden_channels, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, hidden_channels, 1)
        )
        
        # Output layers
        self.to_output = nn.Conv2d(hidden_channels, n_channels, 1)
        
    def graph_process(self, fmap):
        try:
            module = f"{__name__}:graph_process" if self.log_enabled else ""
            graphs = image_to_graph(fmap, log_enabled=self.log_enabled)
            # log_message(f"Processing {len(graphs)} graphs", "INFO", module, self.log_enabled)
            outputs = []

            for i, g in enumerate(graphs):
                out = self.grapher(g.x.to(fmap.device), g.edge_index.to(fmap.device))
                # log_message(f"Graph {i}: output shape {out.shape}", "STATUS", module, self.log_enabled)
                H, W = fmap.shape[2], fmap.shape[3]
                out = out.permute(1, 0).view(-1, H, W)
                outputs.append(out)

            return torch.stack(outputs)
        except Exception as e:
            log_message(f"Error in graph_process: {str(e)}", "ERROR", module, self.log_enabled)
            import traceback
            traceback.print_exc()
            # Return input as fallback
            return fmap
        
    def forward(self, x, steps=1):
        """
        Forward pass with error handling and progress tracking
        """
        try:
            module = f"{__name__}:forward" if self.log_enabled else ""
            # log_message(f"Starting forward pass with input shape: {x.shape}", "INFO", module, self.log_enabled)
            
            # Initial encoding
            h = self.encoder(x)
            # log_message(f"After encoding, shape: {h.shape}", "STATUS", module, self.log_enabled)
            
            # Run NCA steps
            for step in range(steps):
                # log_message(f"NCA step {step+1}/{steps}", "STATUS", module, self.log_enabled)
                # Apply graph-based perception to create perception vector
                p = self.graph_process(h)
                # log_message(f"Perception vector shape: {p.shape}", "STATUS", module, self.log_enabled)
                
                # Update cell states using perception vector
                update = self.update_net(p)
                # log_message(f"Update shape: {update.shape}", "STATUS", module, self.log_enabled)
                
                # Stochastic update with fire rate
                mask = torch.rand_like(update[:, :1], device=self.device) < self.fire_rate
                mask = mask.float().repeat(1, self.hidden_channels, 1, 1)
                h = h + mask * update
                # log_message(f"Updated state shape: {h.shape}", "STATUS", module, self.log_enabled)
            
            # Generate output
            out = self.to_output(h)
            # log_message(f"Output shape before sigmoid: {out.shape}", "SUCCESS", module, self.log_enabled)
            return torch.sigmoid(out)
            
        except Exception as e:
            module = f"{__name__}:forward" if self.log_enabled else ""
            log_message(f"Error in GraphMedNCA forward pass: {str(e)}", "ERROR", module, self.log_enabled)
            log_message(f"Input shape: {x.shape}, dtype: {x.dtype}", "WARNING", module, self.log_enabled)
            import traceback
            traceback.print_exc()
            # Return a dummy output for debugging
            return torch.zeros_like(x)