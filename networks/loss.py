import torch.nn.functional as F
import torch
import torch.nn as nn
import torch_explain as te
from torch.nn.functional import one_hot

from utils.util import create_group_indices


def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)

class CELoss(nn.Module):
    def __init__(self, reduction='mean', weight=None):
        super(CELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction,
                                             weight=weight)

    def forward(self, input, target):
        input = input["prediction_out"]
        loss = self.criterion(input, target)
        return {"target_loss": loss}

class CELossWithEntropy(nn.Module):
    def __init__(self, lm, model, reduction='mean', weight=None):
        super(CELossWithEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction,
                                             weight=weight)
        self.lm = lm
        self.model = model
    def forward(self, input, target):
        input = input["prediction_out"]
        loss = self.criterion(input, target) + self.lm * te.nn.functional.entropy_logic_loss(self.model)
        return {"target_loss": loss}

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean', weight=None):
        super(BCEWithLogitsLoss, self).__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        input = input["prediction_out"]
        loss = self.criterion(input.squeeze(), target.float())
        return {"target_loss": loss}

class BCEWithLogitsLossWithEntropy(nn.Module):
    def __init__(self, lm, model, reduction='mean', weight=None):
        super(BCEWithLogitsLossWithEntropy, self).__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.lm = lm
        self.model = model

    def forward(self, input, target):
        input = input["prediction_out"]
        loss = self.criterion(input.squeeze(), one_hot(target).float()) + self.lm * te.nn.functional.entropy_logic_loss(self.model)
        return {"target_loss": loss}

class CustomConceptLoss(nn.Module):
    def __init__(self, concept_names):
        """
        Initialize the custom loss function.

        Parameters:
        - concept_names (dict): A dictionary specifying the concept groups and their indices.
        """
        super(CustomConceptLoss, self).__init__()
        self.concept_names = concept_names
        self.group_indices = create_group_indices(concept_names)
        self.criterion_ce = nn.NLLLoss()  # Cross-entropy loss for groups with multiple concepts
        self.criterion_bce = nn.BCELoss()  # Binary cross-entropy loss for groups with a single concept

    def forward(self, predictions, targets):
        """
        Compute the custom loss.

        Parameters:
        - predictions (torch.Tensor): The model output tensor containing predictions for all concepts.
        - targets (torch.Tensor): The ground truth labels tensor for all concepts.

        Returns:
        - loss (torch.Tensor): The total loss as a sum of all cross-entropy and binary cross-entropy losses.
        """
        total_loss = 0.0
        individual_losses = []

        for group, indices in self.group_indices.items():

            assert len(indices) != 2, f"Group of two concepts should be handled as one concept."
            if len(indices) > 2:
                # For groups with multiple concepts, use Cross-Entropy Loss
                group_predictions = predictions[:, indices]  # Select predictions for this group
                group_predictions = torch.log(group_predictions + 1e-12)  # Log probabilities
                group_targets = torch.argmax(targets[:, indices],dim=1)  # Convert to class indices
                group_loss = self.criterion_ce(group_predictions, group_targets)
            else:
                # For groups with a single concept, use Binary Cross-Entropy Loss
                group_index = indices[0]
                group_predictions = predictions[:, group_index].unsqueeze(1)  # Select predictions for this concept
                group_targets = targets[:, group_index].unsqueeze(1)  # Select targets for this concept
                group_loss = self.criterion_bce(group_predictions, group_targets)

            # Accumulate the loss
            total_loss += group_loss
            individual_losses.append(group_loss.detach().cpu().numpy())

        return total_loss, individual_losses

class SelectiveNetLoss(torch.nn.Module):
    def __init__(
            self, iteration, CE, selection_threshold, lm, alpha, coverage: float,
            dataset="cub", device='cpu', arch=None
    ):
        """
        Based on the implementation of SelectiveNet
        Args:
            loss_func: base loss function. the shape of loss_func(x, target) shoud be (B).
                       e.g.) torch.nn.CrossEntropyLoss(reduction=none) : classification
            coverage: target coverage.
            lm: Lagrange multiplier for coverage constraint. original experiment's value is 32.
        """
        super(SelectiveNetLoss, self).__init__()
        #assert 0.0 < coverage <= 1.0

        self.CE = CE
        self.coverage = coverage
        self.iteration = iteration
        self.selection_threshold = selection_threshold
        self.dataset = dataset
        self.arch = arch
        self.device = device
        self.lm = lm
        self.alpha = alpha

    def forward(self, outputs, target, prev_selection_outs=None):

        prediction_out = outputs["prediction_out"]
        selection_out = outputs["selection_out"]
        out_aux = outputs["out_aux"]
        target = target

        if prev_selection_outs is None:
            prev_selection_outs = []

        if self.iteration == 1:
            weights = selection_out
        else:
            pi = 1
            for prev_selection_out in prev_selection_outs:
                pi *= (1 - prev_selection_out)
            weights = pi * selection_out

        emp_coverage = torch.mean(weights)

        CE_risk = torch.mean(self.CE(prediction_out, target) * weights.view(-1))
        emp_risk = (CE_risk) / (emp_coverage + 1e-12)

        # compute penalty (=psi)
        coverage = torch.tensor([self.coverage], dtype=torch.float32, requires_grad=True, device=self.device)
        penalty = (torch.max(
            coverage - emp_coverage,
            torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device=self.device),
        ) ** 2)
        penalty *= self.lm

        selective_loss = emp_risk + penalty

        # auxillary loss
        aux_loss = torch.mean(self.CE(out_aux, target))
        total_loss = self.alpha * selective_loss + (1 - self.alpha) * aux_loss
        return {
            "selective_loss": selective_loss,
            "emp_coverage": emp_coverage,
            "CE_risk": CE_risk,
            "emp_risk": emp_risk,
            "cov_penalty": penalty,
            "aux_loss": aux_loss,
            "target_loss": total_loss
        }
        # return total_loss
