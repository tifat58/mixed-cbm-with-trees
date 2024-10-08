from pathlib import Path

from torch import nn
import torch
import torch_explain as te

from data_loaders import find_class_imbalance
from networks.custom_chaid_tree import CHAIDTree
from networks.custom_dt_gini_with_entropy_metrics import \
    CustomDecisionTree
from networks.loss import SelectiveNetLoss, CELoss, CustomConceptLoss, \
    CELossWithEntropy
from networks.model_cub import get_model


class CUBCBMwithDTaslabelPredictorArchitecture:
    def __init__(self, config, device, hard_concepts=None, data_loader=None):

        if hard_concepts is None:
            self.hard_concepts = []
        else:
            self.hard_concepts = hard_concepts

        train_data_path = Path("datasets/CUB/class_attr_data_10/train.pkl")
        if "use_attribute_imbalance" in config["dataset"]:
            if config["dataset"]["use_attribute_imbalance"]:
                self.imbalance = torch.FloatTensor(find_class_imbalance(train_data_path, True)).to(device)
            else:
                self.imbalance = None
        else:
            self.imbalance = None

        if "pretrained_concept_predictor" in config["model"]:
            if config["model"]["use_all_concepts"] == False:
                self.selected_concepts = config["dataset"]["indices_to_keep_from_or_concept_list"]
                if "use_attribute_imbalance" in config["dataset"]:
                    if config["dataset"]["use_attribute_imbalance"]:
                        self.imbalance = self.imbalance[self.selected_concepts]
            else:
                self.selected_concepts = [i for i in range(config["dataset"]["num_concepts"])]
        else:
            if "train_with_all_concepts" in config["model"]:
                if config["model"]["train_with_all_concepts"] == False:
                    self.selected_concepts = config["dataset"]["indices_to_keep_from_or_concept_list"]
                    if "use_attribute_imbalance" in config["dataset"]:
                        if config["dataset"]["use_attribute_imbalance"]:
                            self.imbalance = self.imbalance[self.selected_concepts]
            else:
                self.selected_concepts = [i for i in range(config["dataset"]["num_concepts"])]

        self.concept_predictor, _, _ = get_model("/Users/gouse/PycharmProjects/AR-Imperial-Thesis/networks")
        if 'tree_type' in config["model"]:
            if config["model"]["tree_type"] == "binary":
                self.label_predictor = CustomDecisionTree(
                    min_samples_leaf=config["regularisation"]["min_samples_leaf"],
                    n_classes=config["dataset"]["num_classes"],
                    precision_round=False
                )
        else:
            self.label_predictor = CHAIDTree(min_child_size=config["regularisation"]["min_samples_leaf"],
                                             n_classes=config["dataset"]["num_classes"],
                                             concept_names=config["dataset"]["concept_names"])

        self.model = MainNetwork(self.concept_predictor, self.label_predictor)

        if "pretrained_concept_predictor" in config["model"]:
            state_dict = torch.load(config["model"]["pretrained_concept_predictor"])["state_dict"]
            # Create a new state dictionary for the concept predictor layers
            concept_predictor_state_dict = {}

            # Iterate through the original state dictionary and isolate concept predictor layers
            for key, value in state_dict.items():
                if key.startswith('concept_predictor'):
                    # Remove the prefix "concept_predictor."
                    new_key = key.replace('concept_predictor.', '')
                    concept_predictor_state_dict[new_key] = value

            self.model.concept_predictor.load_state_dict(concept_predictor_state_dict)
            print("Loaded pretrained concept predictor from ", config["model"]["pretrained_concept_predictor"])

            self.temperatures = None
            if "temperature_scaling" in config["model"]:
                if config["model"]["temperature_scaling"]:
                    self.temperatures = torch.load(config["model"]["pretrained_concept_predictor"])["whole_model"]

        if "pretrained_concept_predictor_joint" in config["model"]:
            self.concept_predictor_joint, _, _ = get_model("/Users/gouse/PycharmProjects/AR-Imperial-Thesis/networks")
            self.concept_predictor_joint = self.concept_predictor_joint.to(device)
            state_dict = torch.load(config["model"]["pretrained_concept_predictor_joint"])["state_dict"]
            # Create a new state dictionary for the concept predictor layers
            concept_predictor_state_dict = {}

            # Iterate through the original state dictionary and isolate concept predictor layers
            for key, value in state_dict.items():
                if key.startswith('concept_predictor'):
                    # Remove the prefix "concept_predictor."
                    new_key = key.replace('concept_predictor.', '')
                    concept_predictor_state_dict[new_key] = value

            self.concept_predictor_joint.load_state_dict(concept_predictor_state_dict)
            print("Loaded pretrained concept predictor (joint training) from ", config["model"]["pretrained_concept_predictor_joint"])
        else:
            self.concept_predictor_joint = None

        # Define loss functions and optimizers
        concept_names = config["dataset"]["concept_names"]
        self.criterion_concept = CustomConceptLoss(concept_names=concept_names)

        params_to_update = [
            {'params': self.model.concept_predictor.parameters(),
             'weight_decay': 0.000004, 'momentum': 0.9},
        ]
        self.optimizer = torch.optim.Adam(params_to_update,
                                          lr=config["model"]['xc_lr'])
        self.xc_optimizer = torch.optim.SGD(params_to_update, lr=config["model"]['xc_lr'])
        # if "pretrained_concept_predictor" in config["model"]:
        #     self.xc_optimizer.load_state_dict(torch.load(config["model"]["pretrained_concept_predictor"])["optimizer"])
        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.xc_optimizer, verbose=True)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.xc_optimizer, step_size=30, gamma=0.1)
        self.cy_optimizer = None

class CUBCBMArchitecture:
    def __init__(self, config, device, hard_concepts=None, data_loader=None):

        if hard_concepts is None:
            self.hard_concepts = []
        else:
            self.hard_concepts = hard_concepts

        train_data_path = Path("datasets/CUB/class_attr_data_10/train.pkl")
        if "use_attribute_imbalance" in config["dataset"]:
            if config["dataset"]["use_attribute_imbalance"]:
                self.imbalance = torch.FloatTensor(find_class_imbalance(train_data_path, True)).to(device)
            else:
                self.imbalance = None
        else:
            self.imbalance = None

        if "pretrained_concept_predictor" in config["model"]:
            if config["model"]["use_all_concepts"] == False:
                self.selected_concepts = config["dataset"]["indices_to_keep_from_or_concept_list"]
                if "use_attribute_imbalance" in config["dataset"]:
                    if config["dataset"]["use_attribute_imbalance"]:
                        self.imbalance = self.imbalance[self.selected_concepts]
            else:
                self.selected_concepts = [i for i in range(config["dataset"]["num_concepts"])]
        else:
            if "train_with_all_concepts" in config["model"]:
                if config["model"]["train_with_all_concepts"] == False:
                    self.selected_concepts = config["dataset"]["indices_to_keep_from_or_concept_list"]
                    if "use_attribute_imbalance" in config["dataset"]:
                        if config["dataset"]["use_attribute_imbalance"]:
                            self.imbalance = self.imbalance[self.selected_concepts]
            else:
                self.selected_concepts = [i for i in range(config["dataset"]["num_concepts"])]

        self.concept_predictor, _, _ = get_model("/Users/gouse/PycharmProjects/AR-Imperial-Thesis/networks")
        if 'entropy_layer' in config["model"]:
            self.label_predictor = LabelPredictorEntropyNN(concept_size=config["dataset"]["num_concepts"],
                                                            num_classes=config["dataset"]["num_classes"],
                                                            tau=config["model"]["tau"],
                                                            lm=config["model"]["lm"])
        elif 'psi_layer' in config["model"]:
            pass
        else:
            self.label_predictor = LabelPredictor(concept_size=config["dataset"]["num_concepts"],
                                                  num_classes=config["dataset"]["num_classes"])
        self.model = MainNetwork(self.concept_predictor, self.label_predictor)

        if "pretrained_concept_predictor" in config["model"]:
            state_dict = torch.load(config["model"]["pretrained_concept_predictor"])["state_dict"]
            # Create a new state dictionary for the concept predictor layers
            concept_predictor_state_dict = {}

            # Iterate through the original state dictionary and isolate concept predictor layers
            for key, value in state_dict.items():
                if key.startswith('concept_predictor'):
                    # Remove the prefix "concept_predictor."
                    new_key = key.replace('concept_predictor.', '')
                    concept_predictor_state_dict[new_key] = value

            self.model.concept_predictor.load_state_dict(concept_predictor_state_dict)
            print("Loaded pretrained concept predictor from ", config["model"]["pretrained_concept_predictor"])

            self.temperatures = None
            if "temperature_scaling" in config["model"]:
                if config["model"]["temperature_scaling"]:
                    self.temperatures = torch.load(config["model"]["pretrained_concept_predictor"])["whole_model"]

        # Define loss functions and optimizers
        concept_names = config["dataset"]["concept_names"]
        self.criterion_concept = CustomConceptLoss(concept_names=concept_names)
        if 'entropy_layer' in config["model"]:
            self.criterion_label = CELossWithEntropy(
                lm=config["model"]["lm"],
                model=self.model.label_predictor)
        else:
            self.criterion_label = CELoss()

        if "weight_decay" not in config["model"]:
            xc_params_to_update = [
                {'params': self.model.concept_predictor.parameters(), 'weight_decay': config["model"]['xc_weight_decay']},
            ]
            self.xc_optimizer = torch.optim.Adam(xc_params_to_update, lr=config["model"]['xc_lr'])
            # self.xc_optimizer = torch.optim.SGD(
            #     xc_params_to_update,
            #     lr=config["model"]['xc_lr'],
            #     momentum=config["model"]['xc_momentum'],
            # )
            # if "pretrained_concept_predictor" in config["model"]:
            #     self.xc_optimizer.load_state_dict(torch.load(config["model"]["pretrained_concept_predictor"])["optimizer"])
            cy_params_to_update = [
                {'params': self.model.label_predictor.parameters(), 'weight_decay': config["model"]['cy_weight_decay']},
            ]
            self.cy_optimizer = torch.optim.Adam(cy_params_to_update, lr=config["model"]['cy_lr'])
            # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.xc_optimizer, verbose=True)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.xc_optimizer, step_size=30, gamma=0.1)
        else:
            params_to_update = [
                {'params': self.model.concept_predictor.parameters(),
                 'weight_decay': 0.000004},
                {'params': self.model.label_predictor.parameters(),
                 'weight_decay': config["model"]['weight_decay']},
            ]
            self.optimizer = torch.optim.Adam(params_to_update,
                                              lr=config["model"]['lr'])
            # self.optimizer = torch.optim.SGD(params_to_update, lr=config["model"]['lr'])
            # if "pretrained_concept_predictor" in config["model"]:
            #     self.optimizer.load_state_dict(torch.load(config["model"]["pretrained_concept_predictor"])["optimizer"])
            # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.xc_optimizer, verbose=True)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

class MainNetwork(nn.Module):
    def __init__(self, concept_predictor, label_predictor):
        super(MainNetwork, self).__init__()

        self.concept_predictor = concept_predictor
        self.label_predictor = label_predictor

    def unfreeze_model(self):
        """
        Enable model updates by gradient-descent by unfreezing the model parameters.
        """
        for param in self.parameters():
            param.requires_grad = True

    def freeze_model(self):
        """
        Disable model updates by gradient-descent by freezing the model parameters.
        """
        for param in self.parameters():
            param.requires_grad = False


class LabelPredictorEntropyNN(nn.Module):
    def __init__(self, concept_size, num_classes, tau, lm):
        super(LabelPredictorEntropyNN, self).__init__()
        layers = [te.nn.EntropyLinear(concept_size, 10,
                                      temperature = tau, n_classes=num_classes),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(10, 4),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(4, 1)]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(-1)
        return x

class LabelPredictor(nn.Module):
    def __init__(self, concept_size, num_classes):
        super(LabelPredictor, self).__init__()
        # self.fc1 = nn.Linear(concept_size, 100)
        # self.fc2 = nn.Linear(100, 100)
        # self.fc3 = nn.Linear(100, 100)
        # self.fc4 = nn.Linear(100, num_classes)
        self.fc1 = nn.Linear(concept_size, num_classes)

    def forward(self, c):
        # c = torch.relu(self.fc1(c))
        # c = torch.relu(self.fc2(c))
        # c = torch.relu(self.fc3(c))
        # c = self.fc4(c)
        c = self.fc1(c)
        return c