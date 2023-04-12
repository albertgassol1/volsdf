import sys
from typing import Optional, Union

import fire
import GPUtil
from pyhocon import ConfigFactory
from training.vector_field_train import VectorFieldRunner
from utils.vector_field_config import Config, VectorFieldConfig, VectorFieldNetworkConfig, VectorFieldSamplerConfig, \
    VolSDFConfig

sys.path.append('../code')


def parse_config(n_epochs: Optional[int],
                 expname: Optional[str],
                 experiment_folder_name: Optional[str],
                 vol_sdf_experiment_folder_name: Optional[str],
                 vector_field_config_path: Optional[str],
                 vol_sdf_config_path: Optional[str],
                 scan_id: Optional[int],
                 gpu_index: Optional[Union[int, str]],
                 checkpoint: Optional[str],
                 timestamp: Optional[str],
                 vol_sdf_timestamp: Optional[str],
                 vol_sdf_checkpoint: Optional[str],
                 is_continue: Optional[bool]) -> Config:
    """
    Parse the configuration files.
    :param n_epochs: Number of epochs to train for.
    :param expname: Name of the experiment.
    :param experiment_folder_name: Name of the folder where experiments are stored.
    :param vol_sdf_experiment_folder_name: Name of the VolSDF experiment folder.
    :param vector_field_config_path: Path to vector field configuration file.
    :param vol_sdf_config_path: Path to VolSDF configuration file.
    :param scan_id: Scene to train the vector field network on.
    :param gpu_index: GPU to use.
    :param checkpoint: The checkpoint epoch of the run to be used in case of continuing from a previous run.
    :param timestamp: The timestamp of the run to be used in case of continuing from a previous run.
    :param vol_sdf_timestamp: The timestamp of the VolSDF run to be used.
    :param vol_sdf_checkpoint: The checkpoint epoch of the VolSDF run to be used.
    :param is_continue: If set, indicates continuing from a previous run.
    :return: Vector field configuration.
    """
    # Parse Vector Field configuration
    vector_field_raw_config = ConfigFactory.parse_file(vector_field_config_path)
    vector_field_config = VectorFieldConfig(**vector_field_raw_config.get_config('train'))
    vector_field_config.n_epochs = n_epochs
    vector_field_config.expname += expname
    vector_field_config.experiment_folder_name = experiment_folder_name
    vector_field_config.gpu_index = gpu_index
    vector_field_config.checkpoint = checkpoint
    vector_field_config.timestamp = timestamp
    vector_field_config.is_continue = is_continue
    vector_field_config.config_path = vector_field_config_path

    # Parse VolSDF configuration
    vol_sdf_raw_config = ConfigFactory.parse_file(vol_sdf_config_path)
    vol_sdf_config = VolSDFConfig(vol_sdf_raw_config.get_string('train.expname', default="dtu") + expname,
                                  vol_sdf_experiment_folder_name,
                                  scan_id if scan_id != -
                                  1 else vol_sdf_raw_config.get_string('dataset.scan_id', default=-1),
                                  vol_sdf_timestamp,
                                  vol_sdf_checkpoint)

    # Parse sampler configuration
    vector_field_sampler_config = VectorFieldSamplerConfig(**vector_field_raw_config.get_config('sampler'))

    # Parse vector field network configuration
    vector_field_network_config = VectorFieldNetworkConfig(**vector_field_raw_config.get_config('model'))

    # Return configuration
    return Config(vector_field_config, vol_sdf_config, vector_field_sampler_config, vector_field_network_config)


def train_runner(n_epoch: Optional[int] = 100000,
                 volsdf_conf: Optional[str] = "./confs/dtu.conf",
                 vector_field_conf: Optional[str] = "./confs/vector_field.conf",
                 vol_sdf_timestamp: Optional[str] = "latest",
                 vol_sdf_checkpoint: Optional[str] = "latest",
                 experiment_name: Optional[str] = "",
                 experiment_folder: Optional[str] = "exps_vector_field",
                 vol_sdf_experiment_folder: Optional[str] = "exps",
                 gpu: Optional[str] = "auto",
                 is_continue: Optional[bool] = False,
                 timestamp: Optional[str] = "latest",
                 checkpoint: Optional[str] = "latest",
                 scan_id: Optional[int] = -1) -> None:
    """
    Vector field network training runner.
    :param n_epoch: Number of epochs to train for.
    :param volsdf_conf: Path to VolSDF configuration file.
    :param vector_field_conf: Path to vector field configuration file.
    :param vol_sdf_timestamp: The timestamp of the VolSDF run to be used.
    :param vol_sdf_checkpoint: The checkpoint epoch of the VolSDF run to be used.
    :param experiment_name: Name of the experiment.
    :param experiment_folder: Name of the folder where experiments are stored.
    :param vol_sdf_experiment_folder: Name of the VolSDF experiment folder.
    :param gpu: GPU to use.
    :param is_continue: If set, indicates continuing from a previous run.
    :param timestamp: The timestamp of the run to be used in case of continuing from a previous run.
    :param checkpoint: The checkpoint epoch of the run to be used in case of continuing from a previous run.
    :param scan_id: Scene to train the vector field network on.
    """

    # Set GPU
    if gpu == "auto":
        device_ids = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False,
                                         excludeID=[], excludeUUID=[])
        gpu = device_ids[0]
    else:
        gpu = gpu

    # Parse configuration
    config = parse_config(n_epoch, experiment_name, experiment_folder,
                          vol_sdf_experiment_folder, vector_field_conf,
                          volsdf_conf, scan_id, gpu, checkpoint, timestamp,
                          vol_sdf_timestamp, vol_sdf_checkpoint, is_continue)

    # Create runner
    runner = VectorFieldRunner(config, ConfigFactory.parse_file(volsdf_conf).get_config('model'))

    # Train
    runner.train()


if __name__ == '__main__':
    fire.Fire(train_runner)
