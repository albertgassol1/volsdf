from pyhocon import ConfigFactory
from dataclasses import dataclass


@dataclass(frozen=True)
class VectorFieldNetworkConfig:
    # Hyperparameters
    input_dims: int = 256
    output_dims: int = 3
    feature_vector_dims: int = 0
    dimensions: list = ConfigFactory.parse_string('[256, 256, 256, 256, 256, 256, 256, 256]')
    embedder_multires: int = 0
    weight_norm: bool = True
    skip_connection_in: list = ConfigFactory.parse_string('[4]')
    bias_init: float = 0.0
    dropout: bool = True
    dropout_probability: float = 0.2


@dataclass(frozen=True)
class VectorFieldSamplerConfig:
    # Sampling
    number_of_samples: int = 1000
    n_upsample: int = 20
    max_n_samples: int = 1000
    sdf_epsilon: float = 0.5
    upsampling_radius: float = 1.0


@dataclass(frozen=True)
class VectorFieldConfig:
    learning_rate: float = 0.0001


@dataclass(frozen=True)
class Config:
    vector_field_network: VectorFieldNetworkConfig = VectorFieldNetworkConfig()
    vector_field_sampler: VectorFieldSamplerConfig = VectorFieldSamplerConfig()
    vector_field_config: VectorFieldConfig = VectorFieldConfig()

