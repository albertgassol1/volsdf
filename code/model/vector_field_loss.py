import torch
from torch import nn


class VectorFieldLoss(nn.Module):
    def __init__(self) -> None:
        """
        VectorFieldLoss class for the vector field loss.
        """
        # Initialize the super class.
        super().__init__()

        # Create the cosine similarity
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

        # Create the cosine loss.
        self.cosine_loss = nn.MSELoss()

        # Create the vector field loss.
        self.vector_field_loss = nn.L1Loss()

    def forward(self,
                vector_field_1: torch.Tensor,
                vector_field_2: torch.Tensor,
                cosine_ground_truth: torch.Tensor,
                vector_field_ground_truth: torch.Tensor,
                vector_field_weight: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the vector field loss.
        :params vector_field_1: The first vector field.
        :params vector_field_2: The second vector field,
        associated to a position close to the first vector field position.
        :params cosine_ground_truth: The ground truth cosine similarity.
        :params vector_field_ground_truth: The ground truth vector field.
        :params vector_field_weight: The weight of the vector field.
        """

        # Compute the cosine similarity.
        cosine_similarity = self.cosine_similarity(vector_field_1, vector_field_2)

        # Compute the cosine loss.
        cosine_loss = self.cosine_loss(cosine_similarity, cosine_ground_truth)

        # Compute the vector field loss.
        vector_field_loss = self.vector_field_loss(vector_field_weight[:, None] * vector_field_1,
                                                   vector_field_weight[:, None] * vector_field_ground_truth)

        # Compute the total loss.
        total_loss = cosine_loss + vector_field_loss

        return total_loss
