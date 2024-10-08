from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from torch import nn
import torch
import torch_explain as te

from data_loaders import find_class_imbalance_mnist
from networks.custom_chaid_tree import CHAIDTree
from networks.custom_dt_gini_with_entropy_metrics import \
    CustomDecisionTree
from networks.loss import CELoss, CustomConceptLoss, CELossWithEntropy
from utils import flatten_dict_to_list


class MNISTBlackBoxArchitecture:
    def __init__(self, config, device, hard_concepts=None, data_loader=None):

        self.model = ConceptPredictor(config["dataset"]["num_classes"])

        # Define loss functions and optimizers
        self.criterion_label = CELoss()
        params_to_update = [
            {'params': self.model.parameters(),
             'weight_decay': config["model"]['weight_decay']},
        ]
        self.optimizer = torch.optim.Adam(params_to_update,
                                          lr=config["model"]['lr'])
        self.lr_scheduler = None
        self.selected_concepts = [i for i in range(config["dataset"]["num_concepts"])]

class MNISTCBMwithSLRaslabelPredictorArchitecture:
    def __init__(self, config, device, hard_concepts=None, data_loader=None):

        if hard_concepts is None:
            self.hard_concepts = []
        else:
            self.hard_concepts = hard_concepts

        C_train = data_loader.dataset[:][1]
        if "use_attribute_imbalance" in config["dataset"]:
            if config["dataset"]["use_attribute_imbalance"]:
                self.imbalance = torch.FloatTensor(find_class_imbalance_mnist(C_train)).to(device)
            else:
                self.imbalance = None
        else:
            self.imbalance = None

        self.concept_predictor = ConceptPredictor(config["dataset"]["num_concepts"])
        self.label_predictor = SGDClassifier(random_state=42, loss="log_loss",
                                             alpha=1e-3, l1_ratio=0.99,
                                             verbose=0,
                                             penalty="elasticnet", max_iter=10000)
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
        self.criterion_concept = torch.nn.BCELoss(weight=self.imbalance)
        self.criterion_per_concept = nn.BCELoss(reduction='none')

        xc_params_to_update = [
            {'params': self.model.concept_predictor.parameters(), 'weight_decay': config["model"]['xc_weight_decay']},
        ]
        self.xc_optimizer = torch.optim.Adam(xc_params_to_update, lr=config["model"]['xc_lr'])
        self.cy_optimizer = None
        self.lr_scheduler = None
        self.selected_concepts = [i for i in range(config["dataset"]["num_concepts"])]

class MNISTCBMwithDTaslabelPredictorArchitecture:
    def __init__(self, config, device, hard_concepts=None, data_loader=None):

        if hard_concepts is None:
            self.hard_concepts = []
        else:
            self.hard_concepts = hard_concepts

        C_train = data_loader.dataset[:][1]
        if "use_attribute_imbalance" in config["dataset"]:
            if config["dataset"]["use_attribute_imbalance"]:
                self.imbalance = torch.FloatTensor(find_class_imbalance_mnist(C_train)).to(device)
            else:
                self.imbalance = None
        else:
            self.imbalance = None

        self.concept_predictor = ConceptPredictor(config["dataset"]["num_concepts"])
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
            self.concept_predictor_joint = ConceptPredictor(config["dataset"]["num_concepts"])
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

        xc_params_to_update = [
            {'params': self.model.concept_predictor.parameters(), 'weight_decay': config["model"]['xc_weight_decay']},
        ]
        self.xc_optimizer = torch.optim.Adam(xc_params_to_update, lr=config["model"]['xc_lr'])
        self.cy_optimizer = None
        self.lr_scheduler = None
        self.selected_concepts = [i for i in range(config["dataset"]["num_concepts"])]

class MNISTCBMArchitecture:
    def __init__(self, config, device, hard_concepts=None, data_loader=None):

        if hard_concepts is None:
            self.hard_concepts = []
        else:
            self.hard_concepts = hard_concepts

        C_train = data_loader.dataset[:][1]
        if "use_attribute_imbalance" in config["dataset"]:
            if config["dataset"]["use_attribute_imbalance"]:
                self.imbalance = torch.FloatTensor(find_class_imbalance_mnist(C_train)).to(device)
            else:
                self.imbalance = None
        else:
            self.imbalance = None

        concept_size = config["dataset"]["num_concepts"] - len(self.hard_concepts)
        self.concept_predictor = ConceptPredictor(concept_size)
        if 'entropy_layer' in config["model"]:
            self.label_predictor = LabelPredictorEntropyNN(concept_size=concept_size,
                                                              num_classes=config["dataset"]["num_classes"],
                                                              tau=config["model"]["tau"],
                                                              lm=config["model"]["lm"])
        elif 'psi_layer' in config["model"]:
            pass
        elif 'lr_layer' in config["model"]:
            self.label_predictor = LabelPredictorLR(concept_size=concept_size,
                                                    num_classes=config["dataset"]["num_classes"])
        else:
            self.label_predictor = LabelPredictorNN(concept_size=config["dataset"]["num_concepts"],
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
        # self.criterion_concept = torch.nn.BCELoss(weight=self.imbalance)
        concept_names = config["dataset"]["concept_names"]
        self.criterion_concept = CustomConceptLoss(concept_names=concept_names)
        if 'entropy_layer' in config["model"]:
            self.criterion_label = CELossWithEntropy(lm=config["model"]["lm"],
                                                     model=self.model.label_predictor.layers)
        else:
            self.criterion_label = CELoss()

        if "weight_decay" not in config["model"]:
            xc_params_to_update = [
                {'params': self.model.concept_predictor.parameters(), 'weight_decay': config["model"]['xc_weight_decay']},
            ]
            self.xc_optimizer = torch.optim.Adam(xc_params_to_update, lr=config["model"]['xc_lr'])
            cy_params_to_update = [
                {'params': self.model.label_predictor.parameters(), 'weight_decay': config["model"]['cy_weight_decay']},
            ]
            self.cy_optimizer = torch.optim.Adam(cy_params_to_update, lr=config["model"]['cy_lr'])
        else:
            # only apply regularisation (if any) to the label predictor,
            # for a fair comparison with Tree Regularisation
            params_to_update = [
                {'params': self.model.concept_predictor.parameters(),
                 'weight_decay': 0},
                {'params': self.model.label_predictor.parameters(),
                 'weight_decay': config["model"]['weight_decay']},
            ]
            self.optimizer = torch.optim.Adam(params_to_update,
                                              lr=config["model"]['lr'])

        self.lr_scheduler = None
        self.selected_concepts = [i for i in range(config["dataset"]["num_concepts"])]

class MNISTCYArchitecture:
    def __init__(self, config):

        self.model = LabelPredictor(concept_size=config["dataset"]["num_features"],
                                    num_classes=config["dataset"]["num_classes"])

        # Define loss functions and optimizers
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config["model"]['lr'],
                                          weight_decay=config["model"]['weight_decay'])

# Define the models
# class ConceptPredictor(nn.Module):
#     def __init__(self, num_concepts):
#         super(ConceptPredictor, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, num_concepts)
#
#     def forward(self, x):
#         x = x.view(-1, 28 * 28)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         #x = self.fc3(x)
#         x = self.fc3(x)  # Sigmoid activation for binary concepts
#         return x

class ConceptPredictor(nn.Module):
    def __init__(self, num_concepts):
        super(ConceptPredictor, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # Convolutional layer
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(256, 120),
            # Adjust the input size based on your CNN output
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_concepts)  # Output 6 regression values
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.fc_model(x)
        #x = torch.sigmoid(self.fc_model(x))  # Sigmoid activation for binary concepts
        return x

class LabelPredictorNN(nn.Module):
    def __init__(self, concept_size, num_classes):
        super(LabelPredictorNN, self).__init__()
        self.fc1 = nn.Linear(concept_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, num_classes)

    def forward(self, c):
        c = torch.relu(self.fc1(c))
        c = torch.relu(self.fc2(c))
        c = torch.relu(self.fc3(c))
        c = self.fc4(c)
        return c

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

class LabelPredictorLR(nn.Module):
    def __init__(self, concept_size, num_classes):
        super(LabelPredictorLR, self).__init__()
        self.fc1 = nn.Linear(concept_size, num_classes)

    def forward(self, c):
        c = self.fc1(c)
        return c

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
