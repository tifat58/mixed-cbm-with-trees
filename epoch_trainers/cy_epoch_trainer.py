import os
import pickle
import sys
from matplotlib import pyplot as plt
import graphviz

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from tqdm import tqdm

from loggers.cy_logger import CYLogger
from utils.util import compute_AUC, get_correct, update_pickle_dict
from base.epoch_trainer_base import EpochTrainerBase


class CY_Epoch_Trainer(EpochTrainerBase):
    """
    Trainer Epoch class using Tree Regularization
    """

    def __init__(self, arch, config, device,
                 data_loader, valid_data_loader=None,
                 lr_scheduler=None, expert=None):

        super(CY_Epoch_Trainer, self).__init__(arch, config, expert)

        # Extract the configuration parameters
        self.config = config
        self.device = device
        self.train_loader = data_loader
        self.val_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.arch = arch
        self.model = arch.model.to(self.device)
        self.criterion = arch.criterion_label
        self.min_samples_leaf = config['regularisation']['min_samples_leaf']
        self.epochs = config['trainer']['cy_epochs']
        self.expert = expert
        self.do_validation = self.val_loader is not None
        self.acc_metrics_location = self.config.dir + "/accumulated_metrics.pkl"

        # check if selective net is used
        if "selectivenet" in config.config.keys():
            self.selective_net = True
        else:
            self.selective_net = False

        self.metrics_tracker = CYLogger(config, expert=expert,
                                        tb_path=str(self.config.log_dir),
                                        output_path=str(self.config.save_dir),
                                        train_loader=self.train_loader,
                                        val_loader=self.val_loader,
                                        selectivenet=self.selective_net,
                                        device=self.device)

        self.metrics_tracker.begin_run()
        print("Device: ", self.device)

        self.optimizer = arch.cy_optimizer
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def _train_epoch(self, epoch):

        print(f"Training epoch {epoch}")
        self.metrics_tracker.begin_epoch()
        self.model.label_predictor.train()
        if self.selective_net:
            self.arch.selector.train()
            self.arch.aux_model.train()

        tensor_C = torch.FloatTensor().to(self.device)
        tensor_y_pred = torch.FloatTensor().to(self.device)

        with tqdm(total=len(self.train_loader), file=sys.stdout) as t:
            for batch_idx, (C_batch, y_batch) in enumerate(self.train_loader):
                batch_size = C_batch.size(0)
                C_batch = C_batch.to(self.device)
                tensor_C = torch.cat((tensor_C, C_batch), dim=0)
                y_batch = y_batch.to(self.device)

                # Forward pass
                y_pred = self.model.label_predictor(C_batch)
                tensor_y_pred = torch.cat((tensor_y_pred, y_pred), dim=0)
                outputs = {"prediction_out": y_pred}

                if self.selective_net:
                    out_selector = self.arch.selector(C_batch)
                    out_aux = self.arch.aux_model(C_batch)
                    outputs = {"prediction_out": y_pred, "selection_out": out_selector, "out_aux": out_aux}

                # Calculate Label losses
                loss_label = self.criterion(outputs, y_batch)
                self.metrics_tracker.update_batch(
                    update_dict_or_key=loss_label,
                    batch_size=batch_size,
                    mode='train')

                # Track target training loss and accuracy
                self.metrics_tracker.track_total_train_correct_per_epoch(
                    preds=outputs["prediction_out"], labels=y_batch
                )
                self.metrics_tracker.track_total_train_correct_per_epoch_per_class(
                    preds=outputs["prediction_out"], labels=y_batch
                )

                # if we operate in SGD mode, then X_batch + X_rest = X
                # We still need the complete dataset to compute the APL
                # In full-batch GD, X_batch = X and X_rest = None
                if (batch_idx == len(self.train_loader) - 1):

                    # Calculate the APL
                    APL, fid, fi, tree = self._calculate_APL(self.min_samples_leaf,
                                                             tensor_C, tensor_y_pred)
                    self.metrics_tracker.update_batch(update_dict_or_key='APL',
                                                      value=APL,
                                                      batch_size=len(self.train_loader.dataset),
                                                      mode='train')
                    self.metrics_tracker.update_batch(update_dict_or_key='fidelity',
                                                      value=fid,
                                                      batch_size=len(self.train_loader.dataset),
                                                      mode='train')
                    self.metrics_tracker.update_batch(
                        update_dict_or_key='feature_importance',
                        value=fi,
                        batch_size=len(self.train_loader.dataset),
                        mode='train')

                loss = loss_label["target_loss"]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.metrics_tracker.update_batch(update_dict_or_key='loss',
                                                  value=loss.detach().cpu().item(),
                                                  batch_size=batch_size,
                                                  mode='train')

                t.set_postfix(
                    batch_id='{0}'.format(batch_idx + 1))
                t.update()


        if self.do_validation:
            self._valid_epoch(epoch)

        # Update the epoch metrics
        self.metrics_tracker.end_epoch(selectivenet=self.selective_net)
        log = self.metrics_tracker.result_epoch()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log


    def _valid_epoch(self, epoch):

        print(f"Validation epoch {epoch}")
        self.model.label_predictor.eval()
        if self.selective_net:
            self.arch.selector.eval()
            self.arch.aux_model.eval()

        tensor_C = torch.FloatTensor().to(self.device)
        tensor_y_pred = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            with tqdm(total=len(self.val_loader), file=sys.stdout) as t:
                for batch_idx, (C_batch, y_batch) in enumerate(self.val_loader):
                    batch_size = C_batch.size(0)
                    C_batch = C_batch.to(self.device)
                    tensor_C = torch.cat((tensor_C, C_batch), dim=0)
                    y_batch = y_batch.to(self.device)

                    # Forward pass
                    y_pred = self.model.label_predictor(C_batch)
                    tensor_y_pred = torch.cat((tensor_y_pred, y_pred), dim=0)
                    outputs = {"prediction_out": y_pred}

                    if self.selective_net:
                        out_selector = self.arch.selector(C_batch)
                        out_aux = self.arch.aux_model(C_batch)
                        outputs = {"prediction_out": y_pred,
                                   "selection_out": out_selector,
                                   "out_aux": out_aux}

                    # save outputs for selectivenet
                    if self.selective_net:
                        self.metrics_tracker.update_batch(
                            update_dict_or_key='out_put_sel_proba',
                            value=out_selector,
                            batch_size=batch_size,
                            mode='val')
                        self.metrics_tracker.update_batch(
                            update_dict_or_key='out_put_class',
                            value=y_pred,
                            batch_size=batch_size,
                            mode='val')
                        self.metrics_tracker.update_batch(
                            update_dict_or_key='out_put_target',
                            value=y_batch,
                            batch_size=batch_size,
                            mode='val')

                    # Calculate Label losses
                    loss_label = self.criterion(outputs, y_batch)
                    self.metrics_tracker.update_batch(
                        update_dict_or_key=loss_label,
                        batch_size=batch_size,
                        mode='val')

                    # Track target training loss and accuracy
                    self.metrics_tracker.track_total_val_correct_per_epoch(
                        preds=outputs["prediction_out"], labels=y_batch
                    )
                    self.metrics_tracker.track_total_val_correct_per_epoch_per_class(
                        preds=outputs["prediction_out"], labels=y_batch
                    )

                    # if we operate in SGD mode, then X_batch + X_rest = X
                    # We still need the complete dataset to compute the APL
                    # In full-batch GD, X_batch = X and X_rest = None
                    if (batch_idx == len(self.val_loader) - 1):

                        # Calculate the APL
                        APL, fid, fi, tree = self._calculate_APL(
                            self.min_samples_leaf, tensor_C, tensor_y_pred)
                        self.metrics_tracker.update_batch(update_dict_or_key='APL',
                                                          value=APL,
                                                          batch_size=len(self.val_loader.dataset),
                                                          mode='val')
                        self.metrics_tracker.update_batch(
                            update_dict_or_key='fidelity',
                            value=fid,
                            batch_size=len(self.val_loader.dataset),
                            mode='val')
                        self.metrics_tracker.update_batch(
                            update_dict_or_key='feature_importance',
                            value=fi,
                            batch_size=len(self.val_loader.dataset),
                            mode='val')

                        # if (epoch == self.epochs - 1):
                        #     # visualize last tree
                        #     self._visualize_tree(tree, self.config, epoch, APL, 'None',
                        #                          'None', mode='val')

                        # self._visualize_tree(tree, self.config, epoch, APL, 'None',
                        #                      'None', mode='val')

                    loss = loss_label["target_loss"]
                    self.metrics_tracker.update_batch(update_dict_or_key='loss',
                                                      value=loss.detach().cpu().item(),
                                                      batch_size=batch_size,
                                                      mode='val')
                    t.set_postfix(
                        batch_id='{0}'.format(batch_idx + 1))
                    t.update()

        # Update the epoch metrics
        if self.selective_net:
            # evaluate g for correctly selected samples (pi >= 0.5)
            # should be higher
            self.metrics_tracker.evaluate_correctly(selection_threshold=self.config['selectivenet']['selection_threshold'])

            # evaluate g for correctly rejected samples (pi < 0.5)
            # should be lower
            self.metrics_tracker.evaluate_incorrectly(selection_threshold=self.config['selectivenet']['selection_threshold'])
            self.metrics_tracker.evaluate_coverage_stats(selection_threshold=self.config['selectivenet']['selection_threshold'])

    def predict(self, test_data_loader):

        self.model.label_predictor.eval()
        tensor_y_pred = torch.LongTensor().to(self.device)

        with torch.no_grad():
            with tqdm(total=len(test_data_loader), file=sys.stdout) as t:
                for batch_idx, (C_pred, y_batch) in enumerate(test_data_loader):

                    C_pred = C_pred.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # Forward pass
                    y_pred = self.model.label_predictor(C_pred)
                    preds = torch.argmax(y_pred, dim=1)
                    tensor_y_pred = torch.cat((tensor_y_pred, preds), dim=0)
                    t.set_postfix(
                        batch_id='{0}'.format(batch_idx + 1))
                    t.update()

        return tensor_y_pred.cpu()

    def _test(self, test_data_loader):

        self.model.label_predictor.eval()
        if self.selective_net:
            self.arch.selector.eval()
            self.arch.aux_model.eval()

        tensor_C_pred = torch.FloatTensor().to(self.device)
        tensor_y_pred = torch.FloatTensor().to(self.device)

        test_metrics = {"loss": 0, "target_loss": 0, "accuracy": 0, "APL": 0, "fidelity": 0,
                        "feature_importance": [], "APL_predictions": [], "total_correct": 0}

        with torch.no_grad():
            with tqdm(total=len(test_data_loader), file=sys.stdout) as t:
                for batch_idx, (C_pred, y_batch) in enumerate(test_data_loader):

                    batch_size = C_pred.size(0)
                    C_pred = C_pred.to(self.device)
                    tensor_C_pred = torch.cat((tensor_C_pred, C_pred), dim=0)
                    y_batch = y_batch.to(self.device)

                    # Forward pass
                    y_pred = self.model.label_predictor(C_pred)
                    tensor_y_pred = torch.cat((tensor_y_pred, y_pred), dim=0)
                    outputs = {"prediction_out": y_pred}

                    # Calculate Label losses
                    test_metrics["total_correct"] += get_correct(y_pred, y_batch, self.config["dataset"]["num_classes"])
                    loss_label = self.criterion(outputs, y_batch)
                    test_metrics["target_loss"] += loss_label["target_loss"].detach().cpu().item() * batch_size

                    # if we operate in SGD mode, then X_batch + X_rest = X
                    # We still need the complete dataset to compute the APL
                    # In full-batch GD, X_batch = X and X_rest = None
                    if (batch_idx == len(test_data_loader) - 1):
                        # Calculate the APL
                        APL, fid, fi, tree = self._calculate_APL(
                            self.min_samples_leaf, tensor_C_pred, tensor_y_pred)
                        test_metrics["APL"] = APL
                        test_metrics["fidelity"] = fid
                        test_metrics["feature_importance"] = fi

                        # if (epoch == self.epochs - 1) and self.selective_net == False:
                        #     # visualize last tree
                        #     self._visualize_tree(tree, self.config, epoch, APL,
                        #                          'None', 'None', mode='val',
                        #                          expert=str(self.expert) + '_joint')

                        # if (epoch == self.epochs - 1) and self.selective_net == False:
                        #     self._build_tree_with_fixed_roots(
                        #         self.min_samples_leaf, C_pred, y_pred,
                        #         self.gt_val_tree, 'val', None,
                        #         expert=str(self.expert) + '_joint'
                        #     )

                    loss = loss_label["target_loss"]
                    test_metrics["loss"] += loss.detach().cpu().item() * batch_size

                    t.set_postfix(
                        batch_id='{0}'.format(batch_idx + 1))
                    t.update()

        # Update the test metrics
        test_metrics["loss"] /= len(test_data_loader.dataset)
        test_metrics["accuracy"] = test_metrics["total_correct"] / len(test_data_loader.dataset)
        test_metrics["target_loss"] /= len(test_data_loader.dataset)

        # save test metrics in pickle
        with open(os.path.join(self.config.save_dir, f"test_metrics_ctoy.pkl"), "wb") as f:
            pickle.dump(test_metrics, f)

        task_acc = test_metrics['accuracy']
        # print test metrics
        print("Test Metrics:")
        print(f"Loss: {test_metrics['loss']}")
        print(f"Accuracy: {test_metrics['accuracy']}")
        print(f"Target Loss: {test_metrics['target_loss']}")
        print(f"APL: {test_metrics['APL']}")
        print(f"Fidelity: {test_metrics['fidelity']}")
        print(f"Feature Importance: {test_metrics['feature_importance']}")

        # put also in the logger info
        self.logger.info(f"Test Metrics:")
        self.logger.info(f"Loss: {test_metrics['loss']}")
        self.logger.info(f"Accuracy: {test_metrics['accuracy']}")
        self.logger.info(f"Target Loss: {test_metrics['target_loss']}")
        self.logger.info(f"APL: {test_metrics['APL']}")
        self.logger.info(f"Fidelity: {test_metrics['fidelity']}")
        self.logger.info(f"Feature Importance: {test_metrics['feature_importance']}")

        update_pickle_dict(self.acc_metrics_location, self.config.exper_name,
                           self.config.run_id, 'task_acc', task_acc)

    def _save_selected_results(self, loader, expert, mode, arch, min_samples_leaf_for_gt=None):
        print(f"\n------------------- Metrics ({mode}) ---------------------")
        print('Loading the best model and applying selectivenet ...')
        tensor_X_rej = torch.FloatTensor().to(self.device)
        tensor_X_acc = torch.FloatTensor().to(self.device)
        tensor_C_rej = torch.FloatTensor().to(self.device)
        tensor_C_acc = torch.FloatTensor().to(self.device)
        tensor_y_acc = torch.LongTensor().to(self.device)
        tensor_y_pred_acc = torch.FloatTensor().to(self.device)
        tensor_y_rej = torch.LongTensor().to(self.device)
        tensor_y_pred_rej = torch.FloatTensor().to(self.device)
        tensor_out_selector = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            with tqdm(total=len(loader), file=sys.stdout) as t:
                for batch_id, (X_batch, C_batch, y_batch) in enumerate(loader):
                    X_batch = X_batch.to(self.device)
                    C_batch = C_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    out_selector = arch.selector(C_batch)
                    tensor_out_selector = torch.cat((tensor_out_selector, out_selector), dim=0)

                    y_pred = arch.model.label_predictor(C_batch)
                    selection_threshold = self.config['selectivenet']['selection_threshold']
                    arr_rej_indices = torch.nonzero(out_selector < selection_threshold, as_tuple=True)[0]
                    arr_sel_indices = torch.nonzero(out_selector >= selection_threshold, as_tuple=True)[0]

                    if arr_rej_indices.size(0) > 0:
                        g_X = X_batch[arr_rej_indices, :, :, :]
                        g_concepts = C_batch[arr_rej_indices, :]
                        g_y = y_batch[arr_rej_indices]
                        g_ypred = y_pred[arr_rej_indices, :]

                        tensor_X_rej = torch.cat((tensor_X_rej, g_X), dim=0)
                        tensor_C_rej = torch.cat((tensor_C_rej, g_concepts), dim=0)
                        tensor_y_rej = torch.cat((tensor_y_rej, g_y), dim=0)
                        tensor_y_pred_rej = torch.cat((tensor_y_pred_rej, g_ypred), dim=0)

                    if arr_sel_indices.size(0) > 0:
                        g_X = X_batch[arr_sel_indices, :, :, :]
                        g_y = y_batch[arr_sel_indices]
                        g_ypred = y_pred[arr_sel_indices, :]
                        g_concepts = C_batch[arr_sel_indices, :]

                        tensor_X_acc = torch.cat((tensor_X_acc, g_X), dim=0)
                        tensor_y_acc = torch.cat((tensor_y_acc, g_y), dim=0)
                        tensor_C_acc = torch.cat((tensor_C_acc, g_concepts), dim=0)
                        tensor_y_pred_acc = torch.cat((tensor_y_pred_acc, g_ypred), dim=0)

                    # plot a bar plot with the number of concepts equal to 1 per class
                    # for i in range(3):
                    #     print(f'Class {i}')
                    #     class_digit = tensor_C_acc[tensor_y_acc == i]
                    #     for j in range(12):
                    #         print(
                    #             f'Concept {j}: {torch.sum((class_digit[:, j] == 1).int())}')


                    t.set_postfix(
                        batch_id='{0}'.format(batch_id + 1))
                    t.update()

        tensor_X_rej = tensor_X_rej.cpu()
        tensor_X_acc = tensor_X_acc.cpu()
        tensor_C_rej = tensor_C_rej.cpu()
        tensor_C_acc = tensor_C_acc.cpu()
        tensor_y_rej = tensor_y_rej.cpu()
        tensor_y_pred_rej = tensor_y_pred_rej.cpu()
        tensor_y_acc = tensor_y_acc.cpu()
        tensor_y_pred_acc = tensor_y_pred_acc.cpu()
        tensor_out_selector = tensor_out_selector.cpu()

        # plot a bar plot with the number of concepts equal to 1 per class
        # for i in range(3):
        #     print(f'Class {i}')
        #     class_digit = tensor_C_acc[tensor_y_acc == i]
        #     for j in range(12):
        #         print(
        #             f'Concept {j}: {torch.sum((class_digit[:, j] == 1).int())}')

        # Fit a tree
        # APL, fid, fi, tree = self._calculate_APL(self.min_samples_leaf,
        #                                          tensor_C_acc,
        #                                          tensor_y_pred_acc)
        #
        # self._visualize_tree(tree, self.config, None, APL, 'None', 'None',
        #                      mode=f'selected_{mode}_fid_{fid:.2f}_samples', expert=expert)
        #
        # tensor_C_all = torch.cat((tensor_C_acc, tensor_C_rej), dim=0)
        # tensor_y_pred_all = torch.cat((tensor_y_pred_acc, tensor_y_pred_rej), dim=0)
        # tensor_y_all = torch.cat((tensor_y_acc, tensor_y_rej), dim=0)
        # # isolate the indices of all ys = 1
        # tensor_C_all_8s = tensor_C_all[tensor_y_all == 1].numpy()
        # tensor_C_all_9s = tensor_C_all[tensor_y_all == 2].numpy()
        # tensor_C_all_6s = tensor_C_all[tensor_y_all == 0].numpy()
        #
        # APL_all, fid_all, fi_all, tree_all = self._calculate_APL(self.min_samples_leaf,
        #                                          tensor_C_all,
        #                                          tensor_y_pred_all)
        #
        # self._visualize_tree(tree_all, self.config, None, APL_all, 'None', 'None',
        #                      mode=f'all_{mode}_fid_{fid_all:.2f}_samples', expert=expert)
        #
        # APL_all_true, fid_all_true, fi_all_true, tree_all_true = self._calculate_APL_gt(300,
        #                                                                              tensor_C_all,
        #                                                                              tensor_y_all)
        #
        # self._visualize_tree(tree_all_true, self.config, None, APL_all_true, 'None', 'None',
        #                      mode=f'all_true_{mode}_fid_{fid_all_true:.2f}_samples', expert=expert)
        #
        # # Fit a tree
        # APL_truth, fid_truth, fi_truth, tree_truth = self._calculate_APL_gt(min_samples_leaf_for_gt,
        #                                                                     tensor_C_acc,
        #                                                                     tensor_y_acc)
        #
        # self._visualize_tree(tree_truth, self.config, None, APL_truth, 'None', 'None',
        #                      mode=f'selected_{mode}_fid_{fid_truth:.2f}_truth', expert=expert)
        #
        # APL_selector, fid_selector, fi_selector, tree_selector = self._calculate_APL(1,
        #                                                          tensor_C_all,
        #                                                          tensor_out_selector)
        #
        # # export tree
        # plt.figure(figsize=(20, 20))
        # dot_data = export_graphviz(
        #     decision_tree=tree_selector,
        #     out_file=None,
        #     filled=True,
        #     rounded=True,
        #     special_characters=True,
        #     feature_names=self.config['dataset']['concept_names'],
        #     class_names=["rejected", "accepted"],
        # )
        # name = f'selected_{mode}_fid_{fid_selector:.2f}_selector_expert_{expert}_nodes_{APL_selector}'
        #
        # # Render the graph
        # fig_path = str(self.config.log_dir) + '/trees'
        # graph = graphviz.Source(dot_data, directory=fig_path)
        # graph.render(name, format="png", cleanup=True)


        # print(f"APL: {APL}")
        # print(f"Fidelity: {fid}")
        # print(f"Feature Importance: {fi}")

        # print("Output sizes: ")
        # print(f"tensor_X_acc size: {tensor_X_acc.size()}")
        # print(f"tensor_X_rej size: {tensor_X_rej.size()}")
        # print(f"tensor_C_rej size: {tensor_C_rej.size()}")
        # print(f"tensor_y_rej size: {tensor_y_rej.size()}")
        # print(f"tensor_y_pred_rej size: {tensor_y_pred_rej.size()}")
        # print(f"tensor_y_acc size: {tensor_y_acc.size()}")
        # print(f"tensor_y_pred_acc size: {tensor_y_pred_acc.size()}")

        print(f"Expert: {expert}")
        print(f"Number of accepted {mode} samples: {tensor_X_acc.size(0)}")
        print(f"Number of rejected {mode} samples: {tensor_X_rej.size(0)}")
        # #proba = torch.nn.Softmax(dim=1)(tensor_y_pred_rej)[:, 1]
        # acc_rej = accuracy_score(tensor_y_rej.cpu().numpy(),
        #                         tensor_y_pred_rej.cpu().argmax(dim=1).numpy())
        # print(f"{mode} accuracy of the rejected samples: {acc_rej * 100} (%)")
        #
        # #proba = torch.nn.Softmax(dim=1)(tensor_y_pred_acc)[:, 1]
        # acc_acc = accuracy_score(tensor_y_acc.cpu().numpy(),
        #                         tensor_y_pred_acc.cpu().argmax(dim=1).numpy())
        # print(f"{mode} Accuracy of the accepted samples: {acc_acc * 100} (%)")

        # output_path = os.path.join(self.config.save_dir, "intermediate_tensors")
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)

        # torch.save(tensor_X_rej, os.path.join(output_path, f"expert_{expert}_{mode}_tensor_X_rej.pt"))
        # torch.save(tensor_C_rej, os.path.join(output_path, f"expert_{expert}_{mode}_tensor_C_rej.pt"))
        # torch.save(tensor_y_rej, os.path.join(output_path, f"expert_{expert}_{mode}_tensor_y.pt"))

        # return (tensor_X_rej, tensor_C_rej, tensor_y_rej, fi)


        # return (tensor_X_acc, tensor_C_acc, tensor_y_acc, fi_truth, tree_truth,
        #         tensor_X_rej, tensor_C_rej, tensor_y_rej)
        return (tensor_X_acc, tensor_C_acc, tensor_y_acc, None, None,
                tensor_X_rej, tensor_C_rej, tensor_y_rej)

    def _get_predictions_from_selector(self, loader, expert_idx, mode, expert):
        print(f'\nGet test samples selected by expert {expert_idx} ...')
        tensor_X_rej = torch.FloatTensor().to(self.device)
        tensor_X_acc = torch.FloatTensor().to(self.device)
        tensor_C_rej = torch.FloatTensor().to(self.device)
        tensor_C_acc = torch.FloatTensor().to(self.device)
        tensor_y_acc = torch.LongTensor().to(self.device)
        tensor_y_rej = torch.LongTensor().to(self.device)
        tensor_out_selector = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            with tqdm(total=len(loader), file=sys.stdout) as t:
                for batch_id, (X_batch, C_batch, y_batch) in enumerate(loader):
                    X_batch = X_batch.to(self.device)
                    C_batch = C_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # Forward pass
                    C_pred = expert.model.concept_predictor(X_batch)
                    C_pred = torch.sigmoid(C_pred)
                    C_pred = C_pred[:, expert.arch.selected_concepts]

                    out_selector = expert.arch.selector(C_pred)
                    tensor_out_selector = torch.cat((tensor_out_selector, out_selector), dim=0)

                    selection_threshold = self.config['selectivenet']['selection_threshold']
                    arr_rej_indices = torch.nonzero(out_selector < selection_threshold, as_tuple=True)[0]
                    arr_sel_indices = torch.nonzero(out_selector >= selection_threshold, as_tuple=True)[0]

                    if arr_rej_indices.size(0) > 0:
                        g_X = X_batch[arr_rej_indices, :, :, :]
                        g_concepts = C_batch[arr_rej_indices, :]
                        g_y = y_batch[arr_rej_indices]

                        tensor_X_rej = torch.cat((tensor_X_rej, g_X), dim=0)
                        tensor_C_rej = torch.cat((tensor_C_rej, g_concepts), dim=0)
                        tensor_y_rej = torch.cat((tensor_y_rej, g_y), dim=0)

                    if arr_sel_indices.size(0) > 0:
                        g_X = X_batch[arr_sel_indices, :, :, :]
                        g_y = y_batch[arr_sel_indices]
                        g_concepts = C_batch[arr_sel_indices, :]

                        tensor_X_acc = torch.cat((tensor_X_acc, g_X), dim=0)
                        tensor_y_acc = torch.cat((tensor_y_acc, g_y), dim=0)
                        tensor_C_acc = torch.cat((tensor_C_acc, g_concepts), dim=0)

                    t.set_postfix(
                        batch_id='{0}'.format(batch_id + 1))
                    t.update()

        tensor_X_rej = tensor_X_rej.cpu()
        tensor_X_acc = tensor_X_acc.cpu()
        tensor_C_rej = tensor_C_rej.cpu()
        tensor_C_acc = tensor_C_acc.cpu()
        tensor_y_rej = tensor_y_rej.cpu()
        tensor_y_acc = tensor_y_acc.cpu()

        print(f"Expert: {expert_idx}")
        print(f"Number of accepted {mode} samples: {tensor_X_acc.size(0)}")
        print(f"Number of rejected {mode} samples: {tensor_X_rej.size(0)}")

        return (tensor_X_acc, tensor_C_acc, tensor_y_acc, None, None,
                tensor_X_rej, tensor_C_rej, tensor_y_rej)
