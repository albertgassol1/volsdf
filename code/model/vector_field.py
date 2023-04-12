from typing import List, Optional

import torch
import torch.nn.functional as F
from model.embedder import Embedder, get_embedder
from torch import nn


class VectorField(nn.Module):
    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 feature_vector_dims: int,
                 dimensions: List[int],
                 embedder_multires: Optional[int] = 0,
                 weight_norm: Optional[bool] = True,
                 skip_connection_in: Optional[List[int]] = None,
                 bias_init: Optional[float] = 0.0,
                 dropout: Optional[bool] = True,
                 dropout_probability: Optional[float] = 0.0) -> None:
        """
        VectorField class for the vector field network.
        :params input_dims: The number of input dimensions.
        :params output_dims: The number of output dimensions.
        :params feature_vector_dims: The number of feature vector dimensions.
        :params dimensions: The number of dimensions for each layer.
        :params embedder_multires: The number of multiresolution levels.
        :params weight_norm: Whether to use weight normalization.
        :params skip_connection_in: The indices of the layers to use skip connections.
        :params bias_init: The bias initialization value.
        :params dropout: Whether to use dropout.
        :params dropout_probability: The dropout probability.
        """

        # Initialize the super class.
        super().__init__()

        # Add input and output dimensions to the dimensions list.
        dimensions = [input_dims] + dimensions + [output_dims]

        # Create the embedder.
        self.embedder: Optional[Embedder] = None
        if embedder_multires > 0:
            self.embedder, input_channels = get_embedder(embedder_multires, input_dims)
            dimensions[0] = input_channels

        # Create the layers.
        self.num_layers = len(dimensions) - 1
        self.layers = nn.ModuleList()

        # Create the skip connections.
        self.skip_connection_in = skip_connection_in
        if self.skip_connection_in is None:
            self.skip_connection_in = []

        # Create the layers.
        for i in range(self.num_layers):
            # Create layers
            if i + 1 in self.skip_connection_in:
                out_dimensions = dimensions[i + 1] - dimensions[0]
            else:
                out_dimensions = dimensions[i + 1]

            self.layers.append(nn.Linear(dimensions[i], out_dimensions))

            # Weight normalization.
            if weight_norm:
                self.layers[-1] = nn.utils.weight_norm(self.layers[-1])

            # Xavier initialization.
            nn.init.xavier_uniform_(self.layers[-1].weight)
            nn.init.constant_(self.layers[-1].bias, bias_init)

        # Create the activation function.
        self.activation = nn.ReLU()

        # Create activation function for the last layer.
        self.last_activation = nn.Tanh()

        # Create the dropout.
        self.dropout = dropout
        if self.dropout:
            self.dropout_probability = dropout_probability

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the vector field network.
        :params input_tensor: The input tensor.
        :returns: The output tensor.
        """

        # Embed the input.
        if self.embedder is not None:
            input_tensor = self.embedder(input_tensor)

        # Pass through the layers.
        x = input_tensor
        for i in range(self.num_layers):
            # Pass through the layer.
            if i in self.skip_connection_in:
                x = torch.cat([x, input_tensor], 1) / torch.sqrt(torch.tensor([2]).to(x.device).float())

            x = self.layers[i](x)

            # Apply the activation function.
            if i < self.num_layers - 1:
                x = self.activation(x)
            else:
                x = self.last_activation(x)

            # Apply dropout.
            if self.dropout and i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout_probability, training=self.training)

        # Return the output.
        return x
