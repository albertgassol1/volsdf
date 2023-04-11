from abc import ABC, abstractmethod

import torch
from typing import Tuple, Optional

from model.network import ImplicitNetwork


class Sampler(ABC):
    def __init__(self, number_of_samples: int) -> None:
        """
        Sampler class.
        :params number_of_samples: The number of samples to generate.
        """

        # Store the number of samples.
        self.number_of_samples = number_of_samples

    @abstractmethod
    def sample(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample a set of points.
        :returns: The sampled points, attribute of the sampled points.
        """
        pass


class UniformSampler(Sampler):
    def __init__(self, 
                 min_bounds: torch.Tensor,
                 max_bounds: torch.Tensor,
                 number_of_samples: int) -> None:
        """
        Uniform sampler class in a 3D rectangular box.
        :params min_bounds: The minimum bounds of the box.
        :params max_bounds: The maximum bounds of the box.
        :params number_of_samples: The number of samples to generate.
        """

        # Initialize the super class.
        super(UniformSampler, self).__init__(number_of_samples)

        # Store the minimum and maximum bounds.
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

    def sample(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample a set of points.
        :params device: The device to use.
        :returns: The sampled points.
        """

        # Sample the points.
        points = torch.rand((self.number_of_samples, 3), device=self.max_bounds.device)
        points = points * (self.max_bounds - self.min_bounds) + self.min_bounds

        return points, None


class SDFSampler(Sampler):
    def __init__(self, 
                 min_bounds: torch.Tensor,
                 max_bounds: torch.Tensor,
                 number_of_samples: int,
                 network: ImplicitNetwork,
                 n_upsample: int,
                 max_n_samples: int,
                 sdf_epsilon: float,
                 upsampling_radius: float) -> None:
        """
        SDF sampler class in a 3D rectangular box. Upsample in regions where the SDF is small.
        :params min_bounds: The minimum bounds of the box.
        :params max_bounds: The maximum bounds of the box.
        :params number_of_samples: The number of uniform samples to generate.
        :params network: The SDF network.
        :params n_upsample: The number of samples to upsample in regions where the SDF is sall.
        :params max_n_samples: The maximum number of samples to upsample.
        :params sdf_epsilon: The epsilon to use for the values. When the abs(SDF) is smaller than this value, upsample.
        :params upsampling_radius: The radius to use for the upsampling.
        """

        # Initialize the super class.
        super(SDFSampler, self).__init__(number_of_samples)

        # Create uniform sampler
        self.uniform_sampler = UniformSampler(min_bounds, max_bounds, number_of_samples)

        # Store the number of samples to upsample.
        self.n_upsample = n_upsample

        # Store the maximum number of samples to upsample.
        self.max_n_samples = max_n_samples

        # Store epsilon.
        self.sdf_epsilon = torch.tensor([sdf_epsilon]).to(min_bounds.device)

        # Store the upsampling radius.
        self.upsampling_radius = torch.tensor([upsampling_radius]).to(min_bounds.device)

        # Store the SDF network.
        self.sdf_network = network

        # Set the network to eval mode.
        self.sdf_network.eval()

    def sample(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample a set of points.
        :params network: The implicit network.
        :returns: The sampled points, the sdf value of the points.
        """

        # Sample the uniform points.
        points, _ = self.uniform_sampler.sample()

        # Compute the SDF values.
        sdf_values: torch.Tensor = self.sdf_network(points)

        # Compute areas to upsample.
        centroids_to_upsample = torch.where(torch.abs(sdf_values) < self.sdf_epsilon)[0]

        # Upsample in each region.
        for i in range(min(centroids_to_upsample.shape[0], int(self.max_n_samples/self.n_upsample))):
            # Get the centroid.
            centroid = points[centroids_to_upsample[i]]

            # Sample the points.
            points_to_upsample = self.__upsample_region(centroid, self.n_upsample)

            # Add the points to the list.
            points = torch.cat((points, points_to_upsample), dim=0)

            # Compute the SDF values.
            sdf_values = torch.cat((sdf_values, self.sdf_network(points_to_upsample)), dim=0)

        # If there were too many centroids, sample the remaining allowed points in the next centroid.
        if centroids_to_upsample.shape[0] > int(self.max_n_samples/self.n_upsample):
            # Get the centroid.
            centroid = points[centroids_to_upsample[int(self.max_n_samples/self.n_upsample)]]

            # Sample the points.
            points_to_upsample = self.__upsample_region(centroid, self.max_n_samples - points.shape[0])

            # Add the points to the list.
            points = torch.cat((points, points_to_upsample), dim=0)

            # Compute the SDF values.
            sdf_values = torch.cat((sdf_values, self.sdf_network(points_to_upsample)), dim=0)
        
        return points, sdf_values
    
    def __upsample_region(self, centroid: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Upsample a region.
        :params centroid: The centroid of the region.
        :params n_samples: The number of samples to generate.
        :returns: The sampled points.
        """

        # Sample the points.
        points_to_upsample = torch.rand((n_samples, 3), device=centroid.device)
        points_to_upsample = points_to_upsample * self.upsampling_radius + centroid

        return points_to_upsample
