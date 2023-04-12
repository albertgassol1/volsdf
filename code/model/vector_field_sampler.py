from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from model.network import ImplicitNetwork


class Sampler(ABC):
    def __init__(self, number_of_samples: int) -> None:
        """
        Sampler class.
        :params number_of_samples: The number of samples to generate.
        """

        # Store the number of samples.
        self._number_of_samples = number_of_samples

    @property
    def number_of_samples(self) -> int:
        """
        Get the number of samples.
        :returns: The number of samples.
        """
        return self._number_of_samples

    @number_of_samples.setter
    def number_of_samples(self, n_samples: int) -> None:
        """
        Set the number of samples.
        :param n_samples: New number of samples.
        """
        self._number_of_samples = n_samples

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """
        Sample a set of points.
        :returns: The sampled points.
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
        super().__init__(number_of_samples)

        # Store the minimum and maximum bounds.
        self._min_bounds = min_bounds
        self._max_bounds = max_bounds

    @property
    def max_bounds(self) -> torch.Tensor:
        """
        The maximum bounds.
        :returns: The maximum bounds.
        """
        return self._max_bounds

    @property
    def min_bounds(self) -> torch.Tensor:
        """
        The minimum bounds.
        :returns: The minimum bounds.
        """
        return self._min_bounds

    def sample(self) -> torch.Tensor:
        """
        Sample a set of points.
        :params device: The device to use.
        :returns: The sampled points.
        """

        # Sample the points.
        points = torch.rand((self.number_of_samples, 3), device=self._max_bounds.device)
        points = points * (self._max_bounds - self._min_bounds) + self._min_bounds

        return points


class SDFSampler(Sampler):
    def __init__(self,
                 min_bounds: torch.Tensor,
                 max_bounds: torch.Tensor,
                 number_of_samples: int,
                 n_upsample: int,
                 max_n_samples: int,
                 sdf_epsilon: float,
                 upsampling_radius: float,
                 network: ImplicitNetwork
                 ) -> None:
        """
        SDF sampler class in a 3D rectangular box. Upsample in regions where the SDF is small.
        :params min_bounds: The minimum bounds of the box.
        :params max_bounds: The maximum bounds of the box.
        :params number_of_samples: The number of uniform samples to generate.
        :params n_upsample: The number of samples to upsample in regions where the SDF is sall.
        :params max_n_samples: The maximum number of samples to upsample.
        :params sdf_epsilon: The epsilon to use for the values. When the abs(SDF) is smaller than this value, upsample.
        :params upsampling_radius: The radius to use for the upsampling.
        :params network: The SDF network.
        """

        # Initialize the super class.
        super().__init__(number_of_samples)

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
        self._sdf_network = network

        # Set the network to eval mode.
        self._sdf_network.eval()

    @property
    def max_bounds(self) -> torch.Tensor:
        """
        Get the maximum bounds.
        :returns: The maximum bounds.
        """

        return self.uniform_sampler.max_bounds

    @property
    def min_bounds(self) -> torch.Tensor:
        """
        Get the minimum bounds.
        :returns: The minimum bounds.
        """

        return self.uniform_sampler.min_bounds

    @property
    def sdf_network(self) -> ImplicitNetwork:
        """
        Get the SDF network.
        :returns: The SDF network.
        """
        return self._sdf_network

    def sample(self) -> torch.Tensor:
        """
        Sample a set of points.
        :params network: The implicit network.
        :returns: The sampled points
        """

        # Sample the uniform points.
        points = self.uniform_sampler.sample()

        # Compute the SDF values.
        sdf_values: torch.Tensor = self._sdf_network(points)[:, 0]

        # Compute areas to upsample.
        centroids_to_upsample = torch.where(torch.abs(sdf_values) < self.sdf_epsilon)[0]

        # Upsample in each region.
        for i in range(min(centroids_to_upsample.shape[0], int(self.max_n_samples / self.n_upsample))):
            # Get the centroid.
            centroid = points[centroids_to_upsample[i]]

            # Sample the points.
            points_to_upsample = self.__upsample_region(centroid, self.n_upsample)

            # Add the points to the list.
            points = torch.cat((points, points_to_upsample), dim=0)

        # If there were too many centroids, sample the remaining allowed points in the next centroid.
        if centroids_to_upsample.shape[0] > int(self.max_n_samples / self.n_upsample) and \
                points.shape[0] < self.max_n_samples + self.number_of_samples:
            # Get the centroid.
            centroid = points[centroids_to_upsample[int(self.max_n_samples / self.n_upsample)]]

            # Sample the points.
            points_to_upsample = self.__upsample_region(centroid, self.max_n_samples + self.number_of_samples -
                                                        points.shape[0])

            # Add the points to the list.
            points = torch.cat((points, points_to_upsample), dim=0)

        return points

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


class UnitVectorSampler(Sampler):
    def __init__(self,
                 number_of_samples: int,
                 device: torch.device):
        """
        Unit vector sampler class.
        :params number_of_samples: The number of samples to generate.
        :params device: The device to use.
        """

        # Initialize the super class.
        super().__init__(number_of_samples)

        # Store the device.
        self.device = device

    def sample(self) -> torch.Tensor:
        """
        Sample a set of unit vectors.
        :returns: The sampled points.
        """

        # Sample the unit vectors.
        vectors = torch.rand((self.number_of_samples, 3), device=self.device)

        # Normalize the vectors.
        unit_vectors = F.normalize(vectors, p=2, dim=1)

        return unit_vectors


class VectorFieldSampler(Sampler):
    def __init__(self,
                 min_bounds: torch.Tensor,
                 max_bounds: torch.Tensor,
                 number_of_samples: int,
                 n_upsample: int,
                 max_n_samples: int,
                 sdf_epsilon: float,
                 upsampling_radius: float,
                 network: ImplicitNetwork,
                 delta: float
                 ) -> None:
        """
        Vector field sampler class in a 3D rectangular box. Sample paris of points,
        with emphasis on regions where the SDF is small.
        :params min_bounds: The minimum bounds of the box.
        :params max_bounds: The maximum bounds of the box.
        :params number_of_samples: The number of uniform samples to generate.
        :params n_upsample: The number of samples to upsample in regions where the SDF is small.
        :params max_n_samples: The maximum number of samples to upsample.
        :params sdf_epsilon: The epsilon to use for the values. When the abs(SDF) is smaller than this value, upsample.
        :params upsampling_radius: The radius to use for the upsampling.
        :params network: The SDF network.
        :params delta: delta to generate two points from the sampled point.
        :returns: The sampled pairs of points, and their corresponding sdf values
        """

        # Initialize the super class.
        super().__init__(number_of_samples)

        # Create SDf sampler.
        self.sdf_sampler = SDFSampler(min_bounds, max_bounds, number_of_samples,
                                      n_upsample, max_n_samples, sdf_epsilon,
                                      upsampling_radius, network)

        # Create unit vector sampler.
        self.unit_vector_sampler = UnitVectorSampler(number_of_samples, min_bounds.device)

        # Store delta.
        self.delta = torch.tensor([delta]).to(min_bounds.device)

    @property
    def max_bounds(self) -> torch.Tensor:
        """
        :returns: The maximum bounds of the box.
        """

        return self.sdf_sampler.max_bounds

    @property
    def min_bounds(self) -> torch.Tensor:
        """
        :returns: The minimum bounds of the box.
        """

        return self.sdf_sampler.min_bounds

    def sample(self) -> torch.Tensor:
        """
        Sample a set of point pairs and their SDFs.
        :returns: The sampled points pairs and their SDFs. Tensor of shape (number_of_samples, 8).
        """

        # Sample the SDF points.
        sdf_points = self.sdf_sampler.sample()

        # Sample the unit vectors.
        self.unit_vector_sampler.number_of_samples = sdf_points.shape[0]
        unit_vectors = self.unit_vector_sampler.sample()

        # Compute the positive points.
        positive_points = sdf_points + unit_vectors * self.delta

        # Compute the negative points.
        negative_points = sdf_points - unit_vectors * self.delta

        # Compute the SDF values.
        positive_points_sdf = self.sdf_sampler.sdf_network(positive_points)[:, :1]
        negative_points_sdf = self.sdf_sampler.sdf_network(negative_points)[:, :1]

        # Concatenate the points and the SDF values.
        # TODO: Check if dimensions are correct.
        points_and_sdf_values = torch.cat((positive_points, negative_points,
                                           positive_points_sdf, negative_points_sdf), dim=1)

        return points_and_sdf_values
