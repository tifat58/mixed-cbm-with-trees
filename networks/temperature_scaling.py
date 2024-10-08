import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from torch import nn, optim
from torch.nn import functional as F

from utils import create_group_indices, flatten_dict_to_list

"""
Adjusted from: https://github.com/gpleiss/temperature_scaling 
"""

class PlattScaling(nn.Module):
    def __init__(self):
        super(PlattScaling, self).__init__()
        # Initialize parameters 'a' and 'b' for the scaling transformation
        self.a = nn.Parameter(torch.tensor(1.0))  # Slope 'a'
        self.b = nn.Parameter(torch.tensor(0.0))  # Bias 'b'

    def forward(self, logits):
        # Apply the transformation ax + b followed by a sigmoid
        return self.a * logits + self.b

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, concept_names, device):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.concept_names = concept_names
        self.n_concepts = len(flatten_dict_to_list(self.concept_names))
        self.group_indices = create_group_indices(concept_names)
        self.n_groups = len(list(self.concept_names.keys()))
        self.device = device
        self.group_temperatures = nn.ParameterDict()
        bin_boundaries = torch.linspace(0, 1, 15 + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.group_temperatures_scaled = np.array([1.0 for _ in range(self.n_concepts)])
        self.group_platters = nn.ModuleDict()

    def forward(self, logits):

        for group, indices in self.group_indices.items():
            if len(indices) == 1:
                logits[:, indices] = self.group_platters[str(group)](logits[:, indices])
            else:
                logits[:, indices] = (logits[:, indices] / self.group_temperatures_scaled[indices]).float()
        return logits.detach()

    def temperature_scale(self, temperature, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature_vector = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature_vector

    def set_temperature(self, valid_loader):
        """
        Tune the temperature of the model (using the validation set).
        Use Platt scaling for binary classification (when len(indices) == 1),
        sigmoid and thresholding for binary targets, and temperature scaling for
        multi-class (when len(indices) >= 3).

        valid_loader (DataLoader): validation set loader
        """
        self.to(self.device)

        # First: collect all the logits and labels for the validation set
        for group, indices in self.group_indices.items():

            # For binary classification, we'll use BCEWithLogitsLoss
            bce_criterion = nn.BCEWithLogitsLoss().to(self.device)
            ece_criterion = _ECELoss().to(self.device)
            nll_criterion = nn.CrossEntropyLoss().to(self.device)

            assert len(indices) != 2, f"Group of two concepts should be handled as one concept."
            logits_list = []
            labels_list = []
            with torch.no_grad():
                for input, label, _ in valid_loader:
                    input = input.to(self.device)
                    logits = self.model.concept_predictor(input)
                    group_logits = logits[:, indices]

                    if len(indices) == 1:
                        # For binary case, sigmoid and thresholding to 0.5
                        group_targets = label[:, indices].to(
                            self.device).float()
                    else:
                        group_targets = torch.argmax(label[:, indices],
                                                     dim=1).to(self.device)

                    logits_list.append(group_logits)
                    labels_list.append(group_targets)

                logits = torch.cat(logits_list).to(self.device)
                labels = torch.cat(labels_list).to(self.device)

            # If len(indices) == 1, apply Platt scaling (binary classification)
            if len(indices) == 1:
                probs = torch.sigmoid(logits)
                before_bce = bce_criterion(probs.view(-1), labels.view(-1)).item()
                before_ece = ece_criterion(logits, labels.view(-1)).item()
                print('Before Platt scaling - BCE: %.3f, ECE: %.3f' % (
                before_bce, before_ece))

                # Initialize Platt scaling module
                platt_scaler = PlattScaling().to(self.device)

                # Optimize 'a' and 'b' for Platt scaling
                optimizer = optim.LBFGS(platt_scaler.parameters(), lr=0.01,
                                        max_iter=50)

                def eval_platt():
                    optimizer.zero_grad()
                    # Apply Platt scaling to the logits
                    scaled_probs = platt_scaler(logits)
                    # Calculate the binary cross-entropy loss
                    loss = bce_criterion(scaled_probs.view(-1), labels.view(-1))
                    loss.backward()
                    return loss

                optimizer.step(eval_platt)

                # Calculate BCE and ECE after Platt scaling
                scaled_logits = platt_scaler(logits)
                scaled_probs = torch.sigmoid(scaled_logits)
                after_bce = bce_criterion(scaled_probs.view(-1),
                                          labels.view(-1)).item()
                after_ece = ece_criterion(scaled_probs,
                                          labels.view(-1)).item()
                print('After Platt scaling - BCE: %.3f, ECE: %.3f' % (
                after_bce, after_ece))

                self.group_platters[str(group)] = platt_scaler

            # If len(indices) >= 3, apply temperature scaling (multi-class classification)
            elif len(indices) >= 3:

                self.group_temperatures[str(group)] = nn.Parameter(torch.ones(1) * 1.5)
                before_nll = nll_criterion(logits, labels).item()
                before_ece = ece_criterion(logits, labels).item()

                # Temperature scaling
                optimizer = optim.LBFGS([self.group_temperatures[str(group)]],
                                        lr=0.01, max_iter=50)

                def eval_temp():
                    optimizer.zero_grad()
                    loss = nll_criterion(
                        self.temperature_scale(self.group_temperatures[str(group)],
                                               logits), labels)
                    loss.backward()
                    return loss

                optimizer.step(eval_temp)

                # Calculate NLL and ECE after temperature scaling
                after_nll = nll_criterion(
                    self.temperature_scale(self.group_temperatures[str(group)],
                                           logits), labels).item()
                after_ece = ece_criterion(
                    self.temperature_scale(self.group_temperatures[str(group)],
                                           logits), labels).item()
                print('Optimal temperature: %.3f' % self.group_temperatures[
                    str(group)].item())
                print('Before temperature scaling - NLL: %.3f, ECE: %.3f' % (
                before_nll, before_ece))
                print('After temperature scaling - NLL: %.3f, ECE: %.3f' % (
                after_nll, after_ece))

                self.group_temperatures_scaled[indices] = self.group_temperatures[str(group)].item()

        return self

    def count_num_samples_per_bin(self, group, logits, labels, num_classes, temp_scaling=False):
        if temp_scaling:
            logits = self(group, logits)
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        num_samples_per_bin = {}
        labels_per_bin = {i: 0 for i in range(num_classes)}
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            key = f'{bin_lower.item()} - {bin_upper.item()}'
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            labels = labels[in_bin]
            for i in range(num_classes):
                labels_per_bin[i] = labels.eq(i).sum().item()
            num = in_bin.float().sum().item()
            num_samples_per_bin[key] = num
        return num_samples_per_bin, labels_per_bin

    def collect_confidences_and_predictions(self, group, logits, temp_scaling=False):
        if temp_scaling:
            logits = self(group, logits)
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        return confidences, predictions

    def get_bin_keys(self):
        bin_keys = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            key = f'{bin_lower.item()} - {bin_upper.item()}'
            bin_keys.append(key)
        return bin_keys


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model for both binary and multi-class classification.

    The input to this loss is the logits of a model (not softmax/sigmoid scores).

    It divides the confidence outputs into equally-sized interval bins. In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number of samples in each bin.

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        if logits.shape[1] == 1:  # Binary classification case
            # For binary classification, apply sigmoid to logits
            probs = torch.sigmoid(logits)
            confidences = probs.squeeze()  # Remove singleton dimension
            predictions = (confidences >= 0.5).long()  # Threshold at 0.5
        else:  # Multi-class classification case
            # For multi-class, apply softmax
            softmaxes = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(softmaxes, dim=1)

        accuracies = predictions.eq(labels)

        # Initialize the ECE
        ece = torch.zeros(1, device=logits.device)

        # Calculate |confidence - accuracy| in each bin
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(
                bin_upper.item())
            num_in_bin = in_bin.float().sum().item()

            if num_in_bin > 0:
                # Proportion of samples in this bin
                prop_in_bin = num_in_bin / logits.size(0)

                # Accuracy and average confidence for samples in this bin
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                # Calculate the weighted gap and accumulate
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece