import torch.utils.data
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score
import graphviz
import numpy as np
from torch.nn.functional import one_hot
import os
from torch_explain.logic.nn import entropy
from utils.baseline_utils import test_explanation

from sklearn.tree import export_graphviz
from torch import nn

from epoch_trainers.xc_epoch_trainer import XC_Epoch_Trainer
from epoch_trainers.cy_epoch_trainer import CY_Epoch_Trainer
from utils import convert_to_index_name_mapping, flatten_dict_to_list, \
    update_pickle_dict, onehot_to_categorical
from utils.export_decision_paths_with_subtrees import \
    export_decision_paths_with_subtrees
from utils.tree_utils import get_light_colors, prune_tree_chaid


class IndependentCBMTrainer:

    def __init__(self, arch, config, device, data_loader, valid_data_loader,
                 reg=None, expert=None):

        self.arch = arch
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.expert = expert
        self.acc_metrics_location = self.config.dir + "/accumulated_metrics.pkl"

        self.num_concepts = config['dataset']['num_concepts']
        self.concept_names = config['dataset']['concept_names']
        self.class_names = config['dataset']['class_names']

        # concept pre-processing
        self.concept_group_names = list(self.concept_names.keys())
        self.n_concept_groups = len(list(self.concept_names.keys()))
        self.concept_idx_to_name_map = convert_to_index_name_mapping(self.concept_names)
        self.concept_names_flattened = flatten_dict_to_list(self.concept_names)

        # define the x->c model
        self.xc_epoch_trainer = XC_Epoch_Trainer(
            self.arch, self.config,
            self.device, self.data_loader,
            self.valid_data_loader)

        # create a new dataloader for the c->y model
        train_data_loader, val_data_loader = self.create_cy_dataloaders()

        # define the c->y model
        # check if the label predictor is a tree or not
        self.find_label_predictor_type()
        if self.sklearn_label_predictor == False:
            self.cy_epoch_trainer = CY_Epoch_Trainer(
                self.arch, self.config,
                self.device, train_data_loader,
                val_data_loader, expert=self.expert)

    def train(self):

        logger = self.config.get_logger('train')
        if "pretrained_concept_predictor" not in self.config["model"]:
            # train the x->c model
            print("\nTraining x->c")
            logger.info("Training x->c")
            self.xc_epoch_trainer._training_loop(self.config['trainer']['xc_epochs'])
            self.plot_xc()

        # train the c->y model
        print("\nTraining c->y")
        logger.info("Training c->y")

        # if the label predictor is a tree, train the tree
        if self.sklearn_label_predictor:
            # create a new dataloader for the c->y model
            all_C, all_y, all_C_val, all_y_val = self.extract_cy_data()

            # train the c->y model
            print("\nTraining hard c->y DT label predictor")
            logger.info("\nTraining hard c->y DT label predictor")
            self.arch.label_predictor.fit(all_C, all_y)
            # Prune the tree
            if self.chaid_tree:
                self.arch.label_predictor = prune_tree_chaid(self.arch.label_predictor)

            y_pred = self.arch.label_predictor.predict(all_C)
            print(f'Training Accuracy: {accuracy_score(all_y, y_pred)}')
            y_pred = self.arch.label_predictor.predict(all_C_val)
            print(f'Validation Accuracy: {accuracy_score(all_y_val, y_pred)}')
            if self.tree_label_predictor:
                self._visualize_DT_label_predictor(self.arch.label_predictor, X=all_C, path='')
            else:
                print(self.analyze_classifier(self.arch.label_predictor))

        # if the label predictor is not a tree, train the differentiable network
        else:
            self.cy_epoch_trainer._training_loop(self.config['trainer']['cy_epochs'])
            self.plot_cy()

        # if the model has an entropy layer, generate explanations
        if 'entropy_layer' in self.config["model"]:
            self.generate_entropy_explanations()

    def test(self, test_data_loader, hard_cbm=True):

        logger = self.config.get_logger('train')

        # load the best concept predictor if it was trained from scratch
        if self.config["trainer"]["monitor"] != 'off' and "pretrained_concept_predictor" not in self.config["model"]:
            self.load_best_concept_predictor()

        # load the best label predictor if not a tree
        if self.config["trainer"]["monitor"] != 'off':
            if self.sklearn_label_predictor == False:
                self.load_best_label_predictor()

        # evaluate x->c
        if self.chaid_tree:
            tensor_C_pred, tensor_y = self.xc_epoch_trainer._test(test_data_loader, hard_cbm=True, categorise=True)
        else:
            tensor_C_pred, tensor_y = self.xc_epoch_trainer._test(test_data_loader, hard_cbm=hard_cbm)

        # evaluate c->y
        if self.sklearn_label_predictor:
            #tensor_C_pred = tensor_C_pred.detach().cpu().numpy()
            tensor_y = tensor_y.detach().cpu().numpy()
            y_pred = self.arch.label_predictor.predict(tensor_C_pred)
            task_acc = accuracy_score(tensor_y, y_pred)
            print(f'\nTest Accuracy using the Hard CBM: {task_acc}')
            logger.info(f'\nTest Accuracy using the Hard CBM: {task_acc}')
            update_pickle_dict(self.acc_metrics_location,
                               self.config.exper_name, self.config.run_id,
                               'task_acc', task_acc)
            if self.tree_label_predictor:
                update_pickle_dict(self.acc_metrics_location,
                                   self.config.exper_name, self.config.run_id,
                                   'num_tree_nodes',
                                   self.arch.label_predictor.node_count)
        else:
            if torch.is_tensor(tensor_C_pred) == False:
                tensor_C_pred = torch.tensor(tensor_C_pred, dtype=torch.float32).to(self.device)
            # create a new dataloader for the c->y model
            test_data_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(tensor_C_pred, tensor_y),
                batch_size=self.config["data_loader"]["args"]["batch_size"], shuffle=False
            )
            self.cy_epoch_trainer._test(test_data_loader)

        if 'entropy_layer' in self.config["model"]:
            self.assess_entropy_explanations(logger, tensor_C_pred, tensor_y, test_data_loader)

    def load_best_label_predictor(self):
        path = str(self.config.save_dir) + '/model_best.pth'
        state_dict = torch.load(path)["state_dict"]
        # Create a new state dictionary for the concept predictor layers
        cy_state_dict = {}
        # Iterate through the original state dictionary and isolate concept predictor layers
        for key, value in state_dict.items():
            if key.startswith('label_predictor'):
                # Remove the prefix "cy_model."
                new_key = key.replace('label_predictor.', '')
                cy_state_dict[new_key] = value
        self.cy_epoch_trainer.model.label_predictor.load_state_dict(cy_state_dict)
        print("Loaded best label predictor from ", path)

    def load_best_concept_predictor(self):
        path = str(self.config.save_dir) + '/model_best.pth'
        state_dict = torch.load(path)["state_dict"]
        # Create a new state dictionary for the concept predictor layers
        concept_predictor_state_dict = {}

        # Iterate through the original state dictionary and isolate concept predictor layers
        for key, value in state_dict.items():
            if key.startswith('concept_predictor'):
                # Remove the prefix "concept_predictor."
                new_key = key.replace('concept_predictor.', '')
                concept_predictor_state_dict[new_key] = value

        self.xc_epoch_trainer.model.concept_predictor.load_state_dict(
            concept_predictor_state_dict)
        print("Loaded best concept predictor from ", path)
        # logger.info("Loaded best model from ", path)

    def plot_xc(self):
        results_trainer = self.xc_epoch_trainer.metrics_tracker.result()
        train_bce_losses = results_trainer['train_loss_per_concept']
        val_bce_losses = results_trainer['val_loss_per_concept']
        train_total_bce_losses = results_trainer['train_concept_loss']
        val_total_bce_losses = results_trainer['val_concept_loss']

        # Plotting the results
        epochs = range(1, self.config['trainer']['xc_epochs'] + 1)
        epochs_less = range(11, self.config['trainer']['xc_epochs'] + 1)
        plt.figure(figsize=(18, 30))

        plt.subplot(5, 2, 1)
        for i in range(self.n_concept_groups):
            plt.plot(epochs_less,
                     [train_bce_losses[j][i] for j in range(10, self.config['trainer']['xc_epochs'])],
                     label=f'Train BCE Loss {self.concept_group_names[i]}')
        plt.title('Training BCE Loss per Concept')
        plt.xlabel('Epochs')
        plt.ylabel('BCE Loss')
        plt.legend(loc='upper left')

        plt.subplot(5, 2, 2)
        for i in range(self.n_concept_groups):
            plt.plot(epochs_less,
                     [val_bce_losses[j][i] for j in range(10, self.config['trainer']['xc_epochs'])],
                     label=f'Val BCE Loss {self.concept_group_names[i]}')
        plt.title('Validation BCE Loss per Concept')
        plt.xlabel('Epochs')
        plt.ylabel('BCE Loss')
        plt.legend(loc='upper left')

        plt.subplot(5, 2, 3)
        plt.plot(epochs_less, train_total_bce_losses[10:], 'b',
                 label='Total Train BCE loss')
        plt.plot(epochs_less, val_total_bce_losses[10:], 'r',
                 label='Total Val BCE loss')
        plt.title('Total BCE Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Total BCE Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(str(self.config.log_dir) + '/xc_plots.png')
        #plt.show()

    def plot_cy(self):
        results_trainer = self.cy_epoch_trainer.metrics_tracker.result()
        train_target_losses = results_trainer['train_target_loss']
        val_target_losses = results_trainer['val_target_loss']
        train_accuracies = results_trainer['train_accuracy']
        val_accuracies = results_trainer['val_accuracy']
        APLs_train = results_trainer['train_APL']
        APLs_test = results_trainer['val_APL']
        fidelities_train = results_trainer['train_fidelity']
        fidelities_test = results_trainer['val_fidelity']
        FI_train = results_trainer['train_feature_importance']
        FI_test = results_trainer['val_feature_importance']

        # Plotting the results
        epochs = range(1, self.config['trainer']['cy_epochs'] + 1)
        plt.figure(figsize=(18, 30))

        plt.subplot(3, 2, 1)
        plt.plot(epochs, train_target_losses, 'b', label='Training Target loss')
        plt.plot(epochs, val_target_losses, 'r', label='Validation Target loss')
        plt.title('Training and Validation Target Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Target Loss')
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
        plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(epochs, APLs_train, 'b', label='Train')
        plt.plot(epochs, APLs_test, 'r', label='Test')
        plt.title('APL')
        plt.xlabel('Epochs')
        plt.ylabel('APL')
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(epochs, fidelities_train, 'b', label='Train')
        plt.plot(epochs, fidelities_test, 'r', label='Test')
        plt.title('Fidelity')
        plt.xlabel('Epochs')
        plt.ylabel('Fidelity')
        plt.legend()

        plt.subplot(3, 2, 5)
        for i in range(self.n_concept_groups):
            plt.plot(epochs, [fi[i] for fi in FI_train],
                     label=f'{self.concept_group_names[i]}')
        plt.title('Feature Importances (Train)')
        plt.xlabel('Epochs')
        plt.ylabel('Concept Importances')
        plt.legend(loc='upper left')

        plt.subplot(3, 2, 6)
        for i in range(self.n_concept_groups):
            plt.plot(epochs, [fi[i] for fi in FI_test],
                     label=f'{self.concept_group_names[i]}')
        plt.title('Feature Importances (Test)')
        plt.xlabel('Epochs')
        plt.ylabel('Concept Importances')
        plt.legend(loc='upper left')

        plt.tight_layout()
        if self.expert is not None:
            plt.savefig(str(self.config.log_dir) + '/cy_plots_expert_' + str(self.expert) + '.png')
        else:
            plt.savefig(str(self.config.log_dir) + '/cy_plots.png')
        #plt.show()

    def create_cy_dataloaders(self):

        if isinstance(self.data_loader.dataset, torch.utils.data.TensorDataset):
            all_C = self.data_loader.dataset[:][1]
            all_y = self.data_loader.dataset[:][2]
            all_C_val = self.valid_data_loader.dataset[:][1]
            all_y_val = self.valid_data_loader.dataset[:][2]
            train_dataset = torch.utils.data.TensorDataset(all_C, all_y)
            val_dataset = torch.utils.data.TensorDataset(all_C_val, all_y_val)
            train_data_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.config["data_loader"]["args"]["batch_size"], shuffle=True
            )
            val_data_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.config["data_loader"]["args"]["batch_size"], shuffle=False
            )
        else:
            train_data_loader = self.data_loader.dataset.get_all_data_in_tensors(
                batch_size=self.config["data_loader"]["args"]["batch_size"], shuffle=True
            )
            val_data_loader = self.valid_data_loader.dataset.get_all_data_in_tensors(
                batch_size=self.config["data_loader"]["args"]["batch_size"], shuffle=False
            )

        return train_data_loader, val_data_loader

    def extract_cy_data(self):

        if isinstance(self.data_loader.dataset, torch.utils.data.TensorDataset):
            all_C = self.data_loader.dataset[:][1]
            all_y = self.data_loader.dataset[:][2]
            all_C_val = self.valid_data_loader.dataset[:][1]
            all_y_val = self.valid_data_loader.dataset[:][2]
        else:
            all_C, all_y = self.data_loader.dataset.get_all_data_in_tensors(
                batch_size=self.config["data_loader"]["args"]["batch_size"], shuffle=True,
                return_loader=False
            )
            all_C_val, all_y_val = self.valid_data_loader.dataset.get_all_data_in_tensors(
                batch_size=self.config["data_loader"]["args"]["batch_size"], shuffle=False,
                return_loader=False
            )

        all_C = all_C.detach().cpu().numpy()
        all_y = all_y.detach().cpu().numpy()
        all_C_val = all_C_val.detach().cpu().numpy()
        all_y_val = all_y_val.detach().cpu().numpy()

        if self.chaid_tree:
            all_C = onehot_to_categorical(all_C, self.concept_names)
            all_C_val = onehot_to_categorical(all_C_val, self.concept_names)

        return all_C, all_y, all_C_val, all_y_val

    def _visualize_DT_label_predictor(self, tree, X=None, path=None, hard_tree=None):

        colors = get_light_colors(len(self.config['dataset']['class_names']))
        fig_path = str(self.config.log_dir) + '/trees'
        if path is not None:
            fig_path = fig_path + path
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        if self.sklearn_standard_tree:
            dot_data = export_graphviz(tree, out_file=None,
                                       feature_names=self.concept_group_names,
                                       class_names=self.config['dataset']['class_names'],
                                       filled=True, rounded=True,
                                       special_characters=True)
            # Render the graph
            graph = graphviz.Source(dot_data, directory=fig_path)
            name = f'dt_label_predictor_nodes'
            graph.render(name, format="pdf", cleanup=True)

        elif self.chaid_tree:
            APL = tree.node_count
            dot_data = tree.export_tree(feature_names = self.concept_group_names,
                                        class_names = self.config['dataset']['class_names'],
                                        class_colors = colors,
                                        feature_value_names = self.concept_idx_to_name_map)

            # Render the graph
            graph = graphviz.Source(dot_data, directory=fig_path)
            name = f'dt_label_predictor_nodes_{APL}'
            graph.render(name, format="pdf", cleanup=True)

            if X is not None:
                if hard_tree is not None:
                    tree_used_for_paths = hard_tree
                else:
                    tree_used_for_paths = tree
                leaf_indices = tree_used_for_paths.apply(X)

                for leaf in np.unique(leaf_indices):
                    sample_indices = np.where(leaf_indices == leaf)[0]
                    decision_paths = tree_used_for_paths.decision_path(X[sample_indices])
                    export_decision_paths_with_subtrees(
                        tree,
                        decision_paths,
                        feature_names_main=self.concept_group_names,
                        feature_names_subtree=self.concept_names_flattened,
                        class_colors=colors,
                        class_names = self.config['dataset']['class_names'],
                        feature_value_names=self.concept_idx_to_name_map,
                        output_dir = fig_path + '/decision_paths',
                        leaf_id = leaf)

        elif tree.__class__.__name__ == 'CustomDecisionTree':
            APL = tree.node_count
            dot_data = tree.export_tree(
                feature_names=self.concept_names_flattened,
                class_names=self.config['dataset']['class_names'],
                class_colors=colors)

            # Render the graph
            graph = graphviz.Source(dot_data, directory=fig_path)
            name = f'dt_label_predictor_nodes_{APL}'
            graph.render(name, format="pdf", cleanup=True)

            if X is not None:
                if hard_tree is not None:
                    tree_used_for_paths = self.arch.label_predictor
                else:
                    tree_used_for_paths = tree
                leaf_indices = tree_used_for_paths.apply(X)

                for leaf in np.unique(leaf_indices):
                    sample_indices = np.where(leaf_indices == leaf)[0]
                    decision_paths = tree_used_for_paths.decision_path(
                        X[sample_indices])
                    tree.export_decision_paths_with_subtree(
                        decision_paths,
                        feature_names=self.concept_group_names,
                        class_colors=colors,
                        class_names=self.config['dataset']['class_names'],
                        output_dir=fig_path + '/decision_paths',
                        leaf_id=leaf)

    def analyze_classifier(self, sklearn_classifier, k=10, print_lows=False):

        classifier = nn.Linear(
            self.config["dataset"]["num_concepts"], self.config["dataset"]["num_classes"]
        )
        classifier.weight.data = torch.tensor(sklearn_classifier.coef_)
        classifier.bias.data = torch.tensor(sklearn_classifier.intercept_)

        weights = classifier.weight.clone().detach()
        output = []

        if len(self.class_names) == 2:
            weights = [weights.squeeze(), weights.squeeze()]

        for idx in range(self.config["dataset"]["num_classes"]):
            cls_weights = weights[idx]
            topk_vals, topk_indices = torch.topk(cls_weights, k=k)
            topk_indices = topk_indices.detach().cpu().numpy()
            topk_concepts = [self.concept_names_flattened[j] for j in topk_indices]
            analysis_str = [f"Class : {self.class_names[idx]}"]
            for j, c in enumerate(topk_concepts):
                analysis_str.append(f"\t {j + 1} - {c}: {topk_vals[j]:.3f}")
            analysis_str = "\n".join(analysis_str)
            output.append(analysis_str)

            if print_lows:
                topk_vals, topk_indices = torch.topk(-cls_weights, k=k)
                topk_indices = topk_indices.detach().cpu().numpy()
                topk_concepts = [self.concept_names_flattened[j] for j in topk_indices]
                analysis_str = [f"Class : {self.class_names[idx]}"]
                for j, c in enumerate(topk_concepts):
                    analysis_str.append(
                        f"\t {j + 1} - {c}: {-topk_vals[j]:.3f}")
                analysis_str = "\n".join(analysis_str)
                output.append(analysis_str)

        analysis = "\n".join(output)
        return analysis

    def generate_entropy_explanations(self):
        self.cy_epoch_trainer.model.to('cpu')
        all_C, all_y, _, _ = self.extract_cy_data()
        all_C = torch.tensor(all_C, dtype=torch.float32)
        all_y = one_hot(torch.tensor(all_y))
        self.explanations, self.local_explanations = entropy.explain_classes(
            self.cy_epoch_trainer.model.label_predictor.layers, all_C, all_y,
            c_threshold=0.5,
            y_threshold=0.
        )
        extracted_concepts = {}
        for j in range(self.config["dataset"]["num_classes"]):
            n_used_concepts = sum(
                self.cy_epoch_trainer.model.label_predictor.layers[
                    0].concept_mask[j] > 0.5)
            print(f"Extracted concepts: {n_used_concepts}")
            extracted_concepts[j] = n_used_concepts
        # export the explanations in pickle
        update_pickle_dict(self.acc_metrics_location,
                           self.config.exper_name, self.config.run_id,
                           'extracted_concepts', extracted_concepts)
        # # export the explanations in pickle
        # update_pickle_dict(self.acc_metrics_location,
        #                    self.config.exper_name, self.config.run_id,
        #                    'local_explanations', self.local_explanations)
        self.cy_epoch_trainer.model.to(self.device)

    def assess_entropy_explanations(self, logger, tensor_C_pred, tensor_y,
                                       test_data_loader):
        tensor_y = one_hot(tensor_y)
        predictions_model = self.cy_epoch_trainer.predict(test_data_loader)
        predictions_model = one_hot(predictions_model)
        self.cy_epoch_trainer.model.to('cpu')
        self.test_explanation_accuracies_per_class = {}
        predictions_all = []
        targets_all = []
        predictions_model_all = []
        for class_idx, explanation_dict in self.explanations.items():
            class_int = int(class_idx)
            explanation = explanation_dict["explanation"]
            test_mask = torch.arange(len(tensor_C_pred))
            explanation_accuracy, predictions, targets = \
                test_explanation(explanation, tensor_C_pred.cpu(),
                                 tensor_y.cpu(),
                                 class_int, mask=test_mask)
            predictions_model_class = predictions_model[:, class_int]
            predictions_all.extend(predictions)
            targets_all.extend(targets)
            predictions_model_all.extend(predictions_model_class)
            self.test_explanation_accuracies_per_class[
                class_int] = explanation_accuracy
        self.test_explanation_accuracy_total = accuracy_score(targets_all,
                                                              predictions_all)
        print(f'\nExplanation Accuracy: {self.test_explanation_accuracy_total}')
        logger.info(
            f'\nExplanation Accuracy: {self.test_explanation_accuracy_total}')
        self.fidelity_total = accuracy_score(predictions_model_all,
                                             predictions_all)
        print(f'\nExplanation Fidelity: {self.fidelity_total}')
        logger.info(f'\nExplanation Fidelity: {self.fidelity_total}')
        self.test_explanation_f1_score = f1_score(targets_all, predictions_all)
        print(f'\nExplanation F1 Score: {self.test_explanation_f1_score}')
        logger.info(f'\nExplanation F1 Score: {self.test_explanation_f1_score}')
        self.precision_score = precision_score(predictions_model_all,
                                               predictions_all)
        self.recall_score = recall_score(predictions_model_all, predictions_all)
        print(f'\nExplanation Precision (fidelity): {self.precision_score}')
        logger.info(
            f'\nExplanation Precision (fidelity): {self.precision_score}')
        print(f'\nExplanation Recall (fidelity): {self.recall_score}')
        logger.info(f'\nExplanation Recall (fidelity): {self.recall_score}')
        self.fidelity_f1_score = f1_score(predictions_model_all,
                                          predictions_all)
        print(f'\nExplanation Fidelity F1 Score: {self.fidelity_f1_score}')
        logger.info(
            f'\nExplanation Fidelity F1 Score: {self.fidelity_f1_score}')
        update_pickle_dict(self.acc_metrics_location,
                           self.config.exper_name, self.config.run_id,
                           'test_explanation_accuracy_total',
                           self.test_explanation_accuracy_total)
        update_pickle_dict(self.acc_metrics_location,
                           self.config.exper_name, self.config.run_id,
                           'test_explanation_f1_score',
                           self.test_explanation_f1_score)
        update_pickle_dict(self.acc_metrics_location,
                           self.config.exper_name, self.config.run_id,
                           'fidelity_total',
                           self.fidelity_total)
        update_pickle_dict(self.acc_metrics_location,
                           self.config.exper_name, self.config.run_id,
                           'precision_score',
                           self.precision_score)
        update_pickle_dict(self.acc_metrics_location,
                           self.config.exper_name, self.config.run_id,
                           'recall_score',
                           self.recall_score)
        update_pickle_dict(self.acc_metrics_location,
                           self.config.exper_name, self.config.run_id,
                           'fidelity_f1_score',
                           self.fidelity_f1_score)
        # export the explanations in pickle
        update_pickle_dict(self.acc_metrics_location,
                           self.config.exper_name, self.config.run_id,
                           'test_explanation_accuracies_per_class',
                           self.test_explanation_accuracies_per_class)
        print("Explanation test accuracies saved at: ",
              self.acc_metrics_location)
        self.cy_epoch_trainer.model.to(self.device)

    def find_label_predictor_type(self):
        self.chaid_tree = False
        self.sklearn_label_predictor = False
        self.sklearn_standard_tree = False
        if self.arch.model.label_predictor.__class__.__name__ in [
            'DecisionTreeClassifier', 'CustomDecisionTree', 'CHAIDTree',
            'SGDClassifier']:
            self.sklearn_label_predictor = True
            if self.arch.model.label_predictor.__class__.__name__ in [
                'DecisionTreeClassifier', 'CHAIDTree', 'CustomDecisionTree']:
                self.tree_label_predictor = True
                if self.arch.model.label_predictor.__class__.__name__ == 'CHAIDTree':
                    self.chaid_tree = True
                elif self.arch.model.label_predictor.__class__.__name__ == 'DecisionTreeClassifier':
                    self.sklearn_standard_tree = True
            else:
                self.tree_label_predictor = False
        else:
            self.sklearn_label_predictor = False