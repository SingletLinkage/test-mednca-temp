import torch
import torch.nn as nn
from src.models.Model_BasicNCA import BasicNCA
 
class BackboneNCA(BasicNCA):
    r"""Implementation of the backbone NCA of Med-NCA
    """
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
        """
        super(BackboneNCA, self).__init__(channel_n, fire_rate, device, hidden_size)
        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

    # def perceive(self, x):
    #     r"""Perceptive function, combines 2 conv outputs with the identity of the cell
    #         #Args:
    #             x: image
    #     """
    #     y1 = self.p0(x)
    #     y2 = self.p1(x)
    #     y = torch.cat((x,y1,y2),1)
    #     return y

# /group21/med_nca/M3D-NCA/src/models/Model_BackboneNCA.py
    def perceive(self, x):
        """This function creates the perception vector for the NCA by
        creating various derivatives of the input using different
        convolutional operations.
        
        Args:
            x: Input tensor of shape [batch_size, height, width, channels]

        Returns:
            Perception tensor
        """
        # Check if dimensions are too small for standard convolution with padding=1
        if x.shape[1] <= 2 or x.shape[2] <= 2:
            # Use alternative approach for extremely small inputs
            # Option 1: No padding
            y1 = torch.nn.functional.conv2d(
                x.permute(0, 3, 1, 2), 
                self.sobel_weights, 
                padding=0
            )
            
            # Ensure output has expected dimensions by interpolating if needed
            if y1.shape[2] < x.shape[1] or y1.shape[3] < x.shape[2]:
                y1 = torch.nn.functional.interpolate(
                    y1, 
                    size=(x.shape[1], x.shape[2]),
                    mode='nearest'
                )
        else:
            # Original implementation for adequate-sized inputs
            y1 = self.p0(x.permute(0, 3, 1, 2))
        
        return y1.permute(0, 2, 3, 1)