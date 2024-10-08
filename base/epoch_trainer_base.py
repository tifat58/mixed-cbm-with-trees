import os

import numpy as np
import graphviz
import torch
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from base import TrainerBase
from utils.tree_utils import get_light_colors, replace_splits, \
    modify_dot_with_colors, weighted_node_count, prune_tree


class EpochTrainerBase(TrainerBase):
    def __init__(self, arch, config, expert=None):

        super(EpochTrainerBase, self).__init__(arch, config, expert)
        self.or_class_names = config['dataset']['class_names']
        self.class_mapping = [i for i in range(len(self.or_class_names))]
        self.reduced_class_names = self.or_class_names
        self.config = config
        self.expert = expert
        self.or_colors = get_light_colors(len(self.or_class_names))

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _calculate_APL(self, min_samples_leaf, inputs, outputs):

        tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, random_state=42)
        inputs = inputs.detach().cpu().numpy()

        if outputs.shape[1] == 1:
            outputs = torch.sigmoid(outputs)
            outputs = outputs.detach().cpu().numpy()
            preds = np.where(outputs > 0.5, 1, 0).reshape(-1)
        else:
            outputs = outputs.detach().cpu().numpy()
            preds = np.argmax(outputs, axis=1)

        self.reduced_class_names = [self.or_class_names[i] for i in self.class_mapping if i in preds]
        self.reduced_colors = [self.or_colors[i] for i in self.class_mapping if i in preds]
        self.reduced_colors_dict = {i: self.reduced_colors[i] for i in range(len(self.reduced_class_names))}

        tree.fit(inputs, preds)
        y_pred = tree.predict(inputs)
        fid = accuracy_score(preds, y_pred)

        if "tree_apl_type" in self.config['regularisation']:
            if self.config['regularisation']['tree_apl_type'] == 'weighted_node_count':
                APL = weighted_node_count(tree, inputs)
            else:
                APL = tree.tree_.node_count
        else:
            APL = tree.tree_.node_count

        return APL, fid, list(tree.feature_importances_), tree