import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, knn_graph
from torch_geometric.data import Data
from colorama import Fore, Back, Style
from src.utils.helper import log_message
from matplotlib import pyplot as plt
import numpy as np

class GrapherModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(GrapherModule, self).__init__()
        self.gat = GATConv(in_channels, out_channels // num_heads, heads=num_heads)

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        return F.relu(x)
    
def visualize_graph(image_tensor, edge_index, num_nodes=200):
    """Visualize graph connections on image patches"""
    # Create figure for this visualization
    fig = plt.figure(figsize=(10, 10))
    
    # Convert tensor to numpy image (handle both 1-channel and 3-channel)
    img = image_tensor.cpu()
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)  # Convert to RGB
    img = img.permute(1, 2, 0).numpy()
    plt.imshow(img)
    
    # Get node positions
    H, W = image_tensor.shape[1], image_tensor.shape[2]
    coords = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, H, device='cpu'),
        torch.linspace(0, 1, W, device='cpu'),
        indexing='ij'
    ), dim=-1).view(H*W, 2)
    
    # Plot subset of nodes and connections
    nodes = np.random.choice(len(coords), min(num_nodes, len(coords)))
    for node in nodes:
        x, y = coords[node]
        plt.scatter(y*W, x*H, c='red', s=10)
        
        # Plot connections (limit to 4 for clarity)
        neighbors = edge_index[1][edge_index[0] == node]
        for neighbor in neighbors[:4]:
            nx, ny = coords[neighbor]
            plt.plot([y*W, ny*W], [x*H, nx*H], 'r-', alpha=0.3)
            
    plt.title('Graph Connections')
    plt.axis('off')
    
    # Instead of saving, return the figure for embedding in subplots
    return fig


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
        
    def graph_process(self, fmap, return_graph=False):
        try:
            module = f"{__name__}:graph_process" if self.log_enabled else ""
            graphs = image_to_graph(fmap, log_enabled=self.log_enabled)
            outputs = []
            
            # Store the first graph's edge_index for visualization if requested
            edge_index = None
            if return_graph and len(graphs) > 0:
                edge_index = graphs[0].edge_index

            for i, g in enumerate(graphs):
                out = self.grapher(g.x.to(fmap.device), g.edge_index.to(fmap.device))
                H, W = fmap.shape[2], fmap.shape[3]
                out = out.permute(1, 0).view(-1, H, W)
                outputs.append(out)

            result = torch.stack(outputs)
            
            if return_graph:
                return result, edge_index
            else:
                return result
                
        except Exception as e:
            log_message(f"Error in graph_process: {str(e)}", "ERROR", module, self.log_enabled)
            import traceback
            traceback.print_exc()
            # Return input as fallback
            if return_graph:
                return fmap, None
            else:
                return fmap
        
    def forward(self, x, steps=1, return_graph=False):
        """
        Forward pass with error handling and progress tracking
        
        Args:
            x: Input tensor
            steps: Number of NCA steps to run
            return_graph: If True, returns graph visualization data along with output
        """
        try:
            module = f"{__name__}:forward" if self.log_enabled else ""
            
            # Initial encoding
            h = self.encoder(x)
            
            # Store graph visualization data if requested
            graph_edge_index = None
            
            # Run NCA steps
            for step in range(steps):
                # Apply graph-based perception to create perception vector
                if return_graph and step == steps-1:  # Only keep the last step's graph
                    p, graph_edge_index = self.graph_process(h, return_graph=True)
                else:
                    p = self.graph_process(h)
                
                # Update cell states using perception vector
                update = self.update_net(p)
                
                # Stochastic update with fire rate
                mask = torch.rand_like(update[:, :1], device=self.device) < self.fire_rate
                mask = mask.float().repeat(1, self.hidden_channels, 1, 1)
                h = h + mask * update
            
            # Generate output
            out = self.to_output(h)
            output = torch.sigmoid(out)
            
            if return_graph:
                # Return the output and graph visualization data
                return output, (x[0], graph_edge_index)  # Return first input image and edge_index
            else:
                return output
            
        except Exception as e:
            module = f"{__name__}:forward" if self.log_enabled else ""
            log_message(f"Error in GraphMedNCA forward pass: {str(e)}", "ERROR", module, self.log_enabled)
            log_message(f"Input shape: {x.shape}, dtype: {x.dtype}", "WARNING", module, self.log_enabled)
            import traceback
            traceback.print_exc()
            
            # Return dummy outputs based on return_graph parameter
            if return_graph:
                return torch.zeros_like(x), (x[0], None)
            else:
                return torch.zeros_like(x)