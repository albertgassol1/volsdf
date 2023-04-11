import os
from pyhocon import ConfigFactory
import torch
from dataclasses import asdict

from utils.vector_field_config import Config
from model.vector_field_loss import VectorFieldLoss
from model.vector_field import VectorField
from model.sdf_sampler import SDFSampler


class VectorFieldRunner:
    def __init__(self, config: Config) -> None:
        """
        VectorFieldRunner class.
        :params config: The configuration.
        """

        # Set the default data type and number of threads.
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        # Store the configuration.
        self.config = config

        # Create the network.
        self.network = VectorField(**asdict(self.config.vector_field_network))

        # Create the optimizer.
        self.optimizer = torch.optim.Adam(self.network.parameters(), 
                                          lr=self.config.vector_field_config.learning_rate)

        # Create the loss.
        self.loss = VectorFieldLoss()

        # Create the sampler.
        self.sampler = SDFSampler(**asdict(self.config.vector_field_sampler))
