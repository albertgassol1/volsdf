from dataclasses import dataclass
from typing import List, Union

from pyhocon import ConfigFactory


@dataclass(frozen=True)
class VectorFieldNetworkConfig:
    input_dims: int = 3
    output_dims: int = 3
    feature_vector_dims: int = 0
    dimensions: List[int] = ConfigFactory.parse_string('[256, 256, 256, 256, 256, 256, 256, 256]')
    embedder_multires: int = 0
    weight_norm: bool = True
    skip_connection_in: List[int] = ConfigFactory.parse_string('[4]')
    bias_init: float = 0.0
    dropout: bool = True
    dropout_probability: float = 0.2


@dataclass(frozen=True)
class VectorFieldSamplerConfig:
    number_of_samples: int = 1000
    n_upsample: int = 20
    max_n_samples: int = 1000
    sdf_epsilon: float = 0.5
    upsampling_radius: float = 1.0
    delta: float = 0.1


@dataclass
class VolSDFConfig:
    expname: str = "dtu"
    experiment_folder_name: str = "exps"
    scan_id: int = 65
    timestamp: str = "2021-05-18_15-00-00"
    check_point: str = "latest"


@dataclass
class VectorFieldConfig:
    learning_rate: float = 0.0001
    scheduler_decay_rate: float = 0.1
    gpu_index: Union[int, str] = "ignore"
    n_epochs: int = 100000
    expname: str = "dtu"
    experiment_folder_name: str = "exps_vector_field"
    is_continue: bool = False
    timestamp: str = "2021-05-18_15-00-00"
    experiment_dir: str = ""
    new_timestamp: str = ""
    check_points_folder: str = ""
    model_params_folder: str = "ModelParameters"
    optimizer_params_folder: str = "OptimizerParameters"
    scheduler_params_folder: str = "SchedulerParameters"
    config_path: str = "./conf/vector_field.conf"
    checkpoint: str = "latest"
    checkpoint_frequency: int = 100
    epsilon_border: float = 0.25


@dataclass(frozen=True)
class Config:
    vector_field_config: VectorFieldConfig = VectorFieldConfig()
    volsdf_config: VolSDFConfig = VolSDFConfig()
    vector_field_sampler: VectorFieldSamplerConfig = VectorFieldSamplerConfig()
    vector_field_network: VectorFieldNetworkConfig = VectorFieldNetworkConfig()
