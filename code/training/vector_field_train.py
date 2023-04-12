import os
import shutil
import sys
import time
from dataclasses import asdict
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import utils.general as utils
from model.network import VolSDFNetwork
from model.vector_field import VectorField
from model.vector_field_loss import VectorFieldLoss
from model.vector_field_sampler import VectorFieldSampler
from pyhocon.config_tree import ConfigTree
from utils.vector_field_config import Config


class VectorFieldRunner:
    def __init__(self, config: Config, volsdf_config: ConfigTree) -> None:
        """
        VectorFieldRunner class.
        :params config: The configuration.
        :params volsdf_config: volsdf configuration.
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

        # Run checks.
        self.__run_checks()

        # Load volSDF network and get the sdf network.
        volSDF_network = VolSDFNetwork(conf=volsdf_config)

        # Load volSDF network parameters.
        expdir = os.path.join('../', self.config.volsdf_config.experiment_folder_name,
                              self.config.volsdf_config.expname)
        checkpoints_dir = os.path.join(expdir, self.config.volsdf_config.timestamp)
        saved_model_state = torch.load(os.path.join(checkpoints_dir, 'checkpoints', 'ModelParameters',
                                                    self.config.volsdf_config.check_point + ".pth"))
        volSDF_network.load_state_dict(saved_model_state["model_state_dict"])
        self.sdf_network = volSDF_network.implicit_network

        # Create vector field folders.
        self.__create_vector_field_folders()

        # Set the device.
        if self.config.vector_field_config.gpu_index == "ignore":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{self.config.vector_field_config.gpu_index}"
                                       if torch.cuda.is_available() else "cpu")

        # Set the device for the networks.
        self.network.to(self.device)
        self.sdf_network.to(self.device)

        # Freeze the volSDF network.
        for param in self.sdf_network.parameters():
            param.requires_grad = False
        self.sdf_network.eval()

        # Get the min and max bounds of the sampler
        min_bounds, max_bounds = self.__get_sampler_bounds()

        # Create the sampler.
        sampler_args = asdict(self.config.vector_field_sampler)
        sampler_args["network"] = self.sdf_network
        sampler_args["min_bounds"] = min_bounds
        sampler_args["max_bounds"] = max_bounds
        self.sampler = VectorFieldSampler(**sampler_args)

        # Exponential learning rate decay.
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                self.config.vector_field_config.scheduler_decay_rate **
                                                                (1. / self.config.vector_field_config.n_epochs))

        # Get start epoch, vector field model parameters, optimizer parameters and scheduler state.
        self.start_epoch = 0
        if self.config.vector_field_config.is_continue:
            self.start_epoch = self.__load_checkpoint(os.path.join(self.config.vector_field_config.experiment_dir,
                                                                   self.config.vector_field_config.timestamp,
                                                                   'checkpoints'))

    def __run_checks(self) -> None:
        """
        Run checks.
        """
        # Check if the experiment folder exists.
        if self.config.volsdf_config.scan_id != -1:
            self.config.volsdf_config.expname += f"_{self.config.volsdf_config.scan_id}"
        if self.config.volsdf_config.timestamp == "latest":
            if os.path.exists(os.path.join("../", self.config.volsdf_config.experiment_folder_name,
                                           self.config.volsdf_config.expname)):
                timestamps = os.listdir(os.path.join(
                    "../", self.config.volsdf_config.experiment_folder_name, self.config.volsdf_config.expname))
                if len(timestamps) == 0:
                    raise Exception("Wrong experiment folder.")
                self.config.volsdf_config.timestamp = None
                for t in sorted(timestamps):
                    if os.path.exists(os.path.join("../", self.config.volsdf_config.experiment_folder_name,
                                                   self.config.volsdf_config.expname, t, "checkpoints",
                                                   "ModelParameters", str(self.config.volsdf_config.check_point) +
                                                   ".pth")):
                        self.config.volsdf_config.timestamp = t
                if self.config.volsdf_config.timestamp is None:
                    raise Exception("No good timestamp.")
            else:
                raise Exception("Wrong experiment folder.")

    def __create_vector_field_folders(self) -> None:
        """
        Create vector field folders.
        """
        # Run checks on previous experiments.
        if self.config.volsdf_config.scan_id != -1:
            self.config.vector_field_config.expname += f"_{self.config.volsdf_config.scan_id}"

        if self.config.vector_field_config.is_continue and self.config.vector_field_config.timestamp == "latest":
            if os.path.exists(os.path.join("../", self.config.vector_field_config.experiment_folder_name,
                                           self.config.vector_field_config.expname)):
                timestamps = os.listdir(os.path.join(
                    "../", self.config.vector_field_config.experiment_folder_name,
                    self.config.vector_field_config.expname))
                if len(timestamps) == 0:
                    self.config.vector_field_config.is_continue = False
                    self.config.vector_field_config.timestamp = None
                else:
                    self.config.vector_field_config.is_continue = True
                    self.config.vector_field_config.timestamp = sorted(timestamps)[-1]
            else:
                self.config.vector_field_config.is_continue = False
                self.config.vector_field_config.timestamp = None

        # Create the experiment folder.
        utils.mkdir_ifnotexists(os.path.join("../", self.config.vector_field_config.experiment_folder_name))
        self.config.vector_field_config.experiment_dir = \
            os.path.join("../", self.config.vector_field_config.experiment_folder_name,
                         self.config.vector_field_config.expname)
        utils.mkdir_ifnotexists(self.config.vector_field_config.experiment_dir)
        self.config.vector_field_config.new_timestamp = utils.get_timestamp()
        utils.mkdir_ifnotexists(os.path.join(self.config.vector_field_config.experiment_dir,
                                             self.config.vector_field_config.new_timestamp))

        # Create the checkpoints folder.
        self.config.vector_field_config.check_points_folder = \
            os.path.join(self.config.vector_field_config.experiment_dir,
                         self.config.vector_field_config.new_timestamp,
                         "checkpoints")
        utils.mkdir_ifnotexists(self.config.vector_field_config.check_points_folder)

        # Create folders for the model parameters, the optimizer parameters, and the scheduler parameters.
        utils.mkdir_ifnotexists(os.path.join(self.config.vector_field_config.check_points_folder,
                                             self.config.vector_field_config.model_params_folder))
        utils.mkdir_ifnotexists(os.path.join(self.config.vector_field_config.check_points_folder,
                                             self.config.vector_field_config.optimizer_params_folder))
        utils.mkdir_ifnotexists(os.path.join(self.config.vector_field_config.check_points_folder,
                                             self.config.vector_field_config.scheduler_params_folder))

        # Copy the configuration folder.
        shutil.copy2(self.config.vector_field_config.config_path,
                     os.path.join(self.config.vector_field_config.experiment_dir,
                                  self.config.vector_field_config.new_timestamp, 'runconf.conf'))

        # Copy the shell command.
        print(f"shell command : {sys.argv}")

    def __load_checkpoint(self, cehckpoint_path: str) -> int:
        """
        Load checkpoint.
        :param checkpoint_path: Path to the checkpoint.
        :return: Start epoch.
        """

        # Load the model parameters.
        saved_model = torch.load(os.path.join(cehckpoint_path,
                                              self.config.vector_field_config.model_params_folder,
                                              str(self.config.vector_field_config.checkpoint) + ".pth"))
        self.network.load_state_dict(saved_model['model_state_dict'])

        # Load the optimizer parameters.
        self.optimizer.load_state_dict(
            torch.load(os.path.join(cehckpoint_path,
                       self.config.vector_field_config.optimizer_params_folder,
                       str(self.config.vector_field_config.checkpoint) + ".pth"))
            ['optimizer_state_dict'])

        # Load the scheduler parameters.
        self.scheduler.load_state_dict(
            torch.load(os.path.join(cehckpoint_path,
                       self.config.vector_field_config.scheduler_params_folder,
                       str(self.config.vector_field_config.checkpoint) + ".pth"))
            ['scheduler_state_dict'])

        return int(saved_model['epoch'])

    def __get_sampler_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the bounds of the sampler from the trained volsdf.
        :return: Lower and upper bounds of the sampler.
        """
        # Blended MVS dataset.
        if self.config.volsdf_config.scan_id < 24:
            lower_bound = torch.tensor([-2.0, -2.0, -2.0]).to(self.device)
            upper_bound = torch.tensor([2.0, 2.0, 2.0]).to(self.device)
        # DTU dataset.
        else:
            # Load the bounding boxes.
            bb_dict = np.load('../data/DTU/bbs.npz')
            grid_params = bb_dict[str(self.config.volsdf_config.scan_id)]
            # Scale the bounding boxes.
            grid_params = grid_params * [[1.5], [1.0]]
            # Convert to torch tensors.
            lower_bound = torch.from_numpy(grid_params[0]).to(self.device).float()
            upper_bound = torch.from_numpy(grid_params[1]).to(self.device).float()

        return lower_bound, upper_bound

    def __save_checkpoints(self, epoch: int) -> None:
        """
        Save network state.
        :param epoch: current epoch
        """
        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.config.vector_field_config.check_points_folder,
                         self.config.vector_field_config.model_params_folder, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.config.vector_field_config.check_points_folder,
                         self.config.vector_field_config.model_params_folder, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.config.vector_field_config.check_points_folder,
                         self.config.vector_field_config.optimizer_params_folder, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.config.vector_field_config.check_points_folder,
                         self.config.vector_field_config.optimizer_params_folder, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.config.vector_field_config.check_points_folder,
                         self.config.vector_field_config.scheduler_params_folder, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.config.vector_field_config.check_points_folder,
                         self.config.vector_field_config.scheduler_params_folder, "latest.pth"))

    def train(self) -> None:
        """
        Train the network.
        """

        # Start training.
        for epoch in range(self.start_epoch, self.config.vector_field_config.n_epochs):

            # Set the start time.
            start_time = time.time()

            # Train the network.
            self.__train_epoch(epoch)

            # Save the network if the epoch is a multiple of the save frequency.
            if epoch % self.config.vector_field_config.checkpoint_frequency == 0:
                self.__save_checkpoints(epoch)

            # Print the time.
            print(f"Epoch {epoch} took {(time.time() - start_time)*1e3} miliseconds.")

        # Save the network.
        self.__save_checkpoints(self.config.vector_field_config.n_epochs)

    def __train_epoch(self, epoch: int) -> None:
        """
        Train the network for one epoch.
        :param epoch: Epoch number.
        """

        # Set the network to train mode.
        self.network.train()

        # Get the inputs using the sampler.
        inputs = self.sampler.sample()
        # Get ground truth from samples.
        cosine_gt, vetor_1_gt, vector_1_gt_weights = self.__get_ground_truth(inputs)

        # Zero the parameter gradients.
        self.optimizer.zero_grad()
        # Forward pass.
        positive_vector_fields = self.network(inputs[:, :3])
        negative_vector_fields = self.network(inputs[:, 3:6])
        # Compute the loss.
        loss = self.loss(positive_vector_fields, negative_vector_fields,
                         cosine_gt, vetor_1_gt, vector_1_gt_weights)
        # Backward pass.
        loss.backward()
        # Update the weights.
        self.optimizer.step()

        # Print the loss.
        print(f"Epoch {epoch} loss: {loss.item()}")

        # Take a scheduler step
        self.scheduler.step()

    def __get_ground_truth(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the ground truth from the inputs.
        :param inputs: Tensor with positive points, negative points and sdf values.
        :return: cosine similarity ground truth, vector 1 ground truth and weights on the
        vectors loss.
        """

        # Get the positive points.
        positive_points = inputs[:, :3]
        # Get the negative points.
        negative_points = inputs[:, 3:6]
        # Get the sdf values.
        positive_sdf_values = inputs[:, 6]
        negative_sdf_values = inputs[:, 7]

        # Get indices of points whose positive and negative sdf values have different signs.
        indices = torch.where(positive_sdf_values * negative_sdf_values < 0)[0]

        # Generate cosine similarity ground truth.
        # -1 for points with different signs and 1 for points with same signs.
        cosine_gt = torch.ones(inputs.shape[0]).to(self.device)
        cosine_gt[indices] = -1

        # If positive points are far from the center, then the vector 1 ground truth is
        # the normal vector pointing towards the center.
        # Get scene bounds.
        max_bounds = self.sampler.max_bounds
        min_bounds = self.sampler.min_bounds
        # Get indices of positive points close the bounds.
        condition = \
            torch.logical_or(positive_points[:, 0] > max_bounds[0] - self.config.vector_field_config.epsilon_border,
                             positive_points[:, 1] > max_bounds[1] - self.config.vector_field_config.epsilon_border)
        condition = \
            torch.logical_or(condition,
                             positive_points[:, 2] > max_bounds[2] - self.config.vector_field_config.epsilon_border)
        condition = \
            torch.logical_or(condition,
                             positive_points[:, 0] < min_bounds[0] + self.config.vector_field_config.epsilon_border)
        condition = \
            torch.logical_or(condition,
                             positive_points[:, 1] < min_bounds[1] + self.config.vector_field_config.epsilon_border)
        condition = \
            torch.logical_or(condition,
                             positive_points[:, 2] < min_bounds[2] + self.config.vector_field_config.epsilon_border)
        indices_close_to_bounds = torch.where(condition)[0]

        # Generate vector 1 ground truth.
        # Generate normal unit vectors pointing towards the center (considered to be (0,0,0)).
        vetor_1_gt = torch.zeros(inputs.shape[0], 3).to(self.device)
        vector_1_gt_weights = torch.zeros(inputs.shape[0]).to(self.device)
        vetor_1_gt[indices_close_to_bounds] = F.normalize(-positive_points[indices_close_to_bounds], p=2, dim=1)
        vector_1_gt_weights[indices_close_to_bounds] = 1

        return cosine_gt, vetor_1_gt, vector_1_gt_weights
