import os
import pickle
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from loggers.joint_cbm_logger import JointCBMLogger

from base.epoch_trainer_base import EpochTrainerBase
from utils import compute_AUC, column_get_correct, get_correct, \
    sigmoid_or_softmax_with_groups
from utils.tree_utils import prune_tree


class XCY_Epoch_Trainer(EpochTrainerBase):
    """
    Trainer Epoch class using Tree Regularization
    """

    def __init__(self, arch, config, device, data_loader,
                 valid_data_loader=None, iteration=None):

        super(XCY_Epoch_Trainer, self).__init__(arch, config, iteration)

        # Extract the configuration parameters
        self.config = config
        self.device = device
        self.train_loader = data_loader
        self.val_loader = valid_data_loader
        self.arch = arch
        self.lr_scheduler = arch.lr_scheduler
        self.model = arch.model.to(self.device)
        self.alpha = config['model']['alpha']
        self.criterion_concept = arch.criterion_concept
        self.criterion_label = arch.criterion_label
        self.num_concepts = config['dataset']['num_concepts']
        self.min_samples_leaf = config['regularisation']['min_samples_leaf']
        self.epochs = config['trainer']['epochs']
        self.concept_names = config['dataset']['concept_names']
        self.n_concept_groups = len(list(self.concept_names.keys()))
        self.iteration = iteration

        self.do_validation = self.val_loader is not None
        self.acc_metrics_location = self.config.dir + "/accumulated_metrics.pkl"
        self.hard_concepts = self.arch.hard_concepts
        self.soft_concepts = [i for i in range(self.num_concepts) if i not in self.hard_concepts]

        combined_indices = self.hard_concepts + self.soft_concepts
        self.sorted_concept_indices = torch.argsort(torch.tensor(combined_indices))

        # check if selective net is used
        if "selectivenet" in config.config.keys():
            self.selective_net = True
        else:
            self.selective_net = False

        # Initialize the metrics tracker
        self.metrics_tracker = JointCBMLogger(config, iteration=iteration,
                                              tb_path=str(self.config.log_dir),
                                              output_path=str(self.config.save_dir),
                                              train_loader=self.train_loader,
                                              val_loader=self.val_loader,
                                              selectivenet=self.selective_net,
                                              device=self.device)
        self.metrics_tracker.begin_run()

        self.optimizer = arch.optimizer
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)


    def _train_epoch(self, epoch):

        print(f"Training Epoch {epoch}")
        self.metrics_tracker.begin_epoch()

        self.model.concept_predictor.train()
        self.model.label_predictor.train()
        if self.selective_net:
            self.arch.selector.train()
            self.arch.aux_model.train()

        tensor_C_pred= torch.FloatTensor().to(self.device)
        tensor_y_pred = torch.FloatTensor().to(self.device)

        with tqdm(total=len(self.train_loader), file=sys.stdout) as t:
            for batch_idx, (X_batch, C_batch, y_batch) in enumerate(self.train_loader):

                C_hard = C_batch[:, self.hard_concepts]

                batch_size = X_batch.size(0)
                X_batch = X_batch.to(self.device)
                C_hard = C_hard.to(self.device)
                C_batch = C_batch.to(self.device)

                y_batch = y_batch.to(self.device)

                # Forward pass
                C_pred_soft = self.model.concept_predictor(X_batch)
                C_pred_soft = C_pred_soft[:, self.arch.selected_concepts]

                # Track target training loss and accuracy
                self.metrics_tracker.track_total_train_correct_per_epoch_per_concept(
                    preds=C_pred_soft.detach().cpu(), labels=C_batch.detach().cpu()
                )
                # C_pred_soft = torch.sigmoid(C_pred_soft)
                C_pred_soft = sigmoid_or_softmax_with_groups(C_pred_soft, self.concept_names)

                C_pred_concat = torch.cat((C_hard, C_pred_soft), dim=1)
                C_pred = C_pred_concat[:, self.sorted_concept_indices]
                tensor_C_pred = torch.cat((tensor_C_pred, C_pred), dim=0)
                y_pred = self.model.label_predictor(C_pred)
                tensor_y_pred = torch.cat((tensor_y_pred, y_pred), dim=0)
                outputs = {"prediction_out": y_pred}

                if self.selective_net:
                    out_selector = self.arch.selector(C_pred)
                    out_aux = self.arch.aux_model(C_pred)
                    outputs = {"prediction_out": y_pred, "selection_out": out_selector, "out_aux": out_aux}

                # Calculate Concept losses
                loss_concept_total, loss_per_concept  = self.criterion_concept(C_pred, C_batch)
                self.metrics_tracker.update_batch(update_dict_or_key='concept_loss',
                                                  value=loss_concept_total.detach().cpu().item(),
                                                  batch_size=batch_size,
                                                  mode='train')
                self.metrics_tracker.update_batch(update_dict_or_key='loss_per_concept',
                                                  value=loss_per_concept,
                                                  batch_size=batch_size,
                                                  mode='train')

                # Calculate Label losses
                loss_label = self.criterion_label(outputs, y_batch)
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

                # use all predictions in the last batch
                if (batch_idx == len(self.train_loader) - 1):
                    # Calculate the APL
                    APL, fid, fi, tree = self._calculate_APL(self.min_samples_leaf,
                                                             tensor_C_pred, tensor_y_pred)
                    self.metrics_tracker.update_batch(update_dict_or_key='APL',
                                                      value=APL,
                                                      batch_size=len(self.train_loader.dataset),
                                                      mode='train')
                    self.metrics_tracker.update_batch(update_dict_or_key='fidelity',
                                                      value=fid,
                                                      batch_size=len(self.train_loader.dataset),
                                                      mode='train')
                    # self.metrics_tracker.update_batch(
                    #     update_dict_or_key='feature_importance',
                    #     value=fi,
                    #     batch_size=len(self.train_loader.dataset),
                    #     mode='train')

                loss = self.alpha * loss_concept_total + loss_label["target_loss"]

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

        print(f"Validation Epoch {epoch}")
        self.model.concept_predictor.eval()
        self.model.label_predictor.eval()
        if self.selective_net:
            self.arch.selector.eval()
            self.arch.aux_model.eval()

        tensor_C_pred = torch.FloatTensor().to(self.device)
        tensor_y_pred = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            with tqdm(total=len(self.val_loader), file=sys.stdout) as t:
                for batch_idx, (X_batch, C_batch, y_batch) in enumerate(self.val_loader):

                    C_hard = C_batch[:, self.hard_concepts]

                    batch_size = X_batch.size(0)
                    X_batch = X_batch.to(self.device)
                    C_batch = C_batch.to(self.device)

                    C_hard = C_hard.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # Forward pass
                    C_pred_soft = self.model.concept_predictor(X_batch)
                    C_pred_soft = C_pred_soft[:, self.arch.selected_concepts]

                    # Track target training loss and accuracy
                    self.metrics_tracker.track_total_val_correct_per_epoch_per_concept(
                        preds=C_pred_soft.detach().cpu(), labels=C_batch.detach().cpu()
                    )
                    # C_pred_soft = torch.sigmoid(C_pred_soft)
                    C_pred_soft = sigmoid_or_softmax_with_groups(C_pred_soft, self.concept_names)

                    C_pred_concat = torch.cat((C_hard, C_pred_soft), dim=1)
                    C_pred = C_pred_concat[:, self.sorted_concept_indices]
                    tensor_C_pred = torch.cat((tensor_C_pred, C_pred), dim=0)
                    y_pred = self.model.label_predictor(C_pred)
                    tensor_y_pred = torch.cat((tensor_y_pred, y_pred), dim=0)
                    outputs = {"prediction_out": y_pred}

                    if self.selective_net:
                        out_selector = self.arch.selector(C_pred)
                        out_aux = self.arch.aux_model(C_pred)
                        outputs = {"prediction_out": y_pred,
                                   "selection_out": out_selector,
                                   "out_aux": out_aux}

                    # save outputs for selectivenet
                    if self.selective_net:
                        self.metrics_tracker.update_batch(
                            update_dict_or_key='out_put_sel_proba',
                            value=out_selector.detach().cpu(),
                            batch_size=batch_size,
                            mode='val')
                        self.metrics_tracker.update_batch(
                            update_dict_or_key='out_put_class',
                            value=y_pred.detach().cpu(),
                            batch_size=batch_size,
                            mode='val')
                        self.metrics_tracker.update_batch(
                            update_dict_or_key='out_put_target',
                            value=y_batch.detach().cpu(),
                            batch_size=batch_size,
                            mode='val')

                    # Calculate Concept losses
                    loss_concept_total, loss_per_concept = self.criterion_concept(C_pred, C_batch)
                    self.metrics_tracker.update_batch(
                        update_dict_or_key='concept_loss',
                        value=loss_concept_total.detach().cpu().item(),
                        batch_size=batch_size,
                        mode='val')
                    self.metrics_tracker.update_batch(
                        update_dict_or_key='loss_per_concept',
                        value=loss_per_concept,
                        batch_size=batch_size,
                        mode='val')

                    # Calculate Label losses
                    loss_label = self.criterion_label(outputs, y_batch)
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

                    # use all predictions in the last batch
                    if (batch_idx == len(self.val_loader) - 1):
                        # Calculate the APL
                        APL, fid, fi, tree = self._calculate_APL(
                            self.min_samples_leaf, tensor_C_pred, tensor_y_pred)
                        self.metrics_tracker.update_batch(update_dict_or_key='APL',
                                                          value=APL,
                                                          batch_size=len(self.val_loader.dataset),
                                                          mode='val')
                        self.metrics_tracker.update_batch(
                            update_dict_or_key='fidelity',
                            value=fid,
                            batch_size=len(self.val_loader.dataset),
                            mode='val')
                        # self.metrics_tracker.update_batch(
                        #     update_dict_or_key='feature_importance',
                        #     value=fi,
                        #     batch_size=len(self.val_loader.dataset),
                        #     mode='val')

                        # if (epoch == self.epochs - 1) and self.selective_net == False:
                        #     # visualize last tree
                        #     self._visualize_tree(tree, self.config, epoch, APL,
                        #                          'None', 'None', mode='val',
                        #                          iteration=str(self.iteration) + '_joint')

                        # if (epoch == self.epochs - 1) and self.selective_net == False:
                        #     self._build_tree_with_fixed_roots(
                        #         self.min_samples_leaf, C_pred, y_pred,
                        #         self.gt_val_tree, 'val', None,
                        #         iteration=str(self.iteration) + '_joint'
                        #     )

                    loss = self.alpha * loss_concept_total + loss_label["target_loss"]
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

    def _test(self, test_data_loader):

        self.model.concept_predictor.eval()
        self.model.label_predictor.eval()
        if self.selective_net:
            self.arch.selector.eval()
            self.arch.aux_model.eval()

        tensor_C_pred = torch.FloatTensor().to(self.device)
        tensor_y_pred = torch.FloatTensor().to(self.device)

        test_metrics = {"concept_loss": 0, "loss_per_concept": np.zeros(self.n_concept_groups), "total_correct": 0,
                        "accuracy_per_concept": np.zeros(self.n_concept_groups),
                        "loss": 0, "target_loss": 0, "accuracy": 0, "APL": 0, "fidelity": 0,
                        "feature_importance": [], "APL_predictions": []}

        with torch.no_grad():
            with tqdm(total=len(test_data_loader), file=sys.stdout) as t:
                for batch_idx, (X_batch, C_batch, y_batch) in enumerate(test_data_loader):

                    batch_size = X_batch.size(0)
                    X_batch = X_batch.to(self.device)
                    C_batch = C_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # Forward pass
                    C_pred = self.model.concept_predictor(X_batch)
                    C_pred = C_pred[:, self.arch.selected_concepts]

                    # Track number of corrects per concept
                    correct_per_column = column_get_correct(C_pred, C_batch, self.concept_names)
                    correct_per_column = correct_per_column.detach().cpu().numpy()
                    test_metrics["accuracy_per_concept"] += np.array([x for x in correct_per_column])

                    # C_pred = torch.sigmoid(C_pred)
                    C_pred = sigmoid_or_softmax_with_groups(C_pred, self.concept_names)
                    tensor_C_pred = torch.cat((tensor_C_pred, C_pred), dim=0)
                    y_pred = self.model.label_predictor(C_pred)
                    tensor_y_pred = torch.cat((tensor_y_pred, y_pred), dim=0)
                    outputs = {"prediction_out": y_pred}

                    # Calculate Concept losses
                    loss_concept_total, loss_per_concept = self.criterion_concept(C_pred, C_batch)

                    test_metrics["concept_loss"] += loss_concept_total.detach().cpu().item() * batch_size
                    test_metrics["loss_per_concept"] += np.array([x * batch_size for x in loss_per_concept])

                    # Calculate Label losses
                    test_metrics["total_correct"] += get_correct(y_pred, y_batch, self.config["dataset"]["num_classes"])
                    loss_label = self.criterion_label(outputs, y_batch)
                    test_metrics["target_loss"] += loss_label["target_loss"].detach().cpu().item() * batch_size

                    # use all predictions in the last batch
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
                        #                          iteration=str(self.iteration) + '_joint')

                        # if (epoch == self.epochs - 1) and self.selective_net == False:
                        #     self._build_tree_with_fixed_roots(
                        #         self.min_samples_leaf, C_pred, y_pred,
                        #         self.gt_val_tree, 'val', None,
                        #         iteration=str(self.iteration) + '_joint'
                        #     )

                    loss = self.alpha * loss_concept_total + loss_label["target_loss"]
                    test_metrics["loss"] += loss.detach().cpu().item() * batch_size

                    t.set_postfix(
                        batch_id='{0}'.format(batch_idx + 1))
                    t.update()

        # Update the test metrics
        test_metrics["loss"] /= len(test_data_loader.dataset)
        test_metrics["accuracy"] = test_metrics["total_correct"] / len(test_data_loader.dataset)
        test_metrics["concept_loss"] /= len(test_data_loader.dataset)
        test_metrics["loss_per_concept"] = [x / len(test_data_loader.dataset) for x in test_metrics["loss_per_concept"]]
        test_metrics["accuracy_per_concept"] = [x / len(test_data_loader.dataset) for x in test_metrics["accuracy_per_concept"]]
        test_metrics["concept_accuracy"] = sum(test_metrics["accuracy_per_concept"]) / len(test_metrics["accuracy_per_concept"])
        test_metrics["target_loss"] /= len(test_data_loader.dataset)

        # save test metrics in pickle
        with open(os.path.join(self.config.save_dir, f"test_metrics.pkl"), "wb") as f:
            pickle.dump(test_metrics, f)

        # print test metrics
        print("Test Metrics:")
        print(f"Loss: {test_metrics['loss']}")
        print(f"Accuracy: {test_metrics['accuracy']}")
        print(f"Concept Loss: {test_metrics['concept_loss']}")
        print(f"Loss per Concept: {test_metrics['loss_per_concept']}")
        print(f"Concept Accuracy: {test_metrics['concept_accuracy']}")
        print(f"Accuracy per Concept: {test_metrics['accuracy_per_concept']}")
        print(f"Target Loss: {test_metrics['target_loss']}")
        print(f"APL: {test_metrics['APL']}")
        print(f"Fidelity: {test_metrics['fidelity']}")
        print(f"Feature Importance: {test_metrics['feature_importance']}")

        # put also in the logger info
        self.logger.info(f"Test Metrics:")
        self.logger.info(f"Loss: {test_metrics['loss']}")
        self.logger.info(f"Accuracy: {test_metrics['accuracy']}")
        self.logger.info(f"Concept Loss: {test_metrics['concept_loss']}")
        self.logger.info(f"Loss per Concept: {test_metrics['loss_per_concept']}")
        self.logger.info(f"Concept Accuracy: {test_metrics['concept_accuracy']}")
        self.logger.info(f"Accuracy per Concept: {test_metrics['accuracy_per_concept']}")
        self.logger.info(f"Target Loss: {test_metrics['target_loss']}")
        self.logger.info(f"APL: {test_metrics['APL']}")
        self.logger.info(f"Fidelity: {test_metrics['fidelity']}")
        self.logger.info(f"Feature Importance: {test_metrics['feature_importance']}")


    def _save_selected_results(self, loader, expert, mode):
        tensor_X_rej = torch.FloatTensor().to(self.device)
        tensor_X_acc = torch.FloatTensor().to(self.device)
        tensor_C_rej = torch.FloatTensor().to(self.device)
        tensor_C_pred_rej = torch.FloatTensor().to(self.device)
        tensor_C_acc = torch.FloatTensor().to(self.device)
        tensor_C_pred_acc = torch.FloatTensor().to(self.device)
        tensor_y_acc = torch.LongTensor().to(self.device)
        tensor_y_pred_acc = torch.FloatTensor().to(self.device)
        tensor_y_rej = torch.LongTensor().to(self.device)
        tensor_y_pred_rej = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            with tqdm(total=len(loader), file=sys.stdout) as t:
                for batch_id, (X_batch, C_batch, y_batch) in enumerate(loader):
                    X_batch = X_batch.to(self.device)
                    C_batch = C_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    C_pred = self.model.concept_predictor(X_batch)
                    out_selector = self.arch.selector(C_pred)
                    y_pred = self.model.label_predictor(C_pred)
                    selection_threshold = self.config['selectivenet']['selection_threshold']
                    arr_rej_indices = torch.nonzero(out_selector < selection_threshold, as_tuple=True)[0]
                    arr_sel_indices = torch.nonzero(out_selector >= selection_threshold, as_tuple=True)[0]

                    if arr_rej_indices.size(0) > 0:
                        g_X = X_batch[arr_rej_indices, :, :, :]
                        g_concepts = C_batch[arr_rej_indices, :]
                        g_y = y_batch[arr_rej_indices]
                        g_ypred = y_pred[arr_rej_indices, :]
                        g_pred_concepts = C_pred[arr_rej_indices, :]

                        tensor_X_rej = torch.cat((tensor_X_rej, g_X.cpu()), dim=0)
                        tensor_C_rej = torch.cat((tensor_C_rej, g_concepts), dim=0)
                        tensor_y_rej = torch.cat((tensor_y_rej, g_y), dim=0)
                        tensor_y_pred_rej = torch.cat((tensor_y_pred_rej, g_ypred), dim=0)
                        tensor_C_pred_rej = torch.cat((tensor_C_pred_rej, g_pred_concepts), dim=0)

                    if arr_sel_indices.size(0) > 0:
                        g_X = X_batch[arr_sel_indices, :, :, :]
                        g_y = y_batch[arr_sel_indices]
                        g_ypred = y_pred[arr_sel_indices, :]
                        g_concepts = C_batch[arr_sel_indices, :]
                        g_pred_concepts = C_pred[arr_sel_indices, :]

                        tensor_X_acc = torch.cat((tensor_X_acc, g_X.cpu()), dim=0)
                        tensor_y_acc = torch.cat((tensor_y_acc, g_y), dim=0)
                        tensor_C_acc = torch.cat((tensor_C_acc, g_concepts), dim=0)
                        tensor_y_pred_acc = torch.cat((tensor_y_pred_acc, g_ypred), dim=0)
                        tensor_C_pred_acc = torch.cat((tensor_C_pred_acc, g_pred_concepts), dim=0)

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
        tensor_C_pred_acc = tensor_C_pred_acc.cpu()
        tensor_C_pred_rej = tensor_C_pred_rej.cpu()

        # plot a bar plot with the number of concepts equal to 1 per class
        # for i in range(3):
        #     print(f'Class {i}')
        #     class_digit = tensor_C_pred_acc[tensor_y_acc == i]
        #     for j in range(12):
        #         print(
        #             f'Concept {j}: {torch.sum((class_digit[:, j] == 1).int())}')

        # Fit a tree
        APL, fid, fi, tree = self._calculate_APL(self.min_samples_leaf,
                                                 tensor_C_pred_acc,
                                                 tensor_y_pred_acc)

        print(f"APL: {APL}")
        print(f"Fidelity: {fid}")
        print(f"Feature Importance: {fi}")

        print("Output sizes: ")
        print(f"tensor_X size: {tensor_X_acc.size()}")
        print(f"tensor_C size: {tensor_X_rej.size()}")
        print(f"tensor_y_rej size: {tensor_y_rej.size()}")
        print(f"tensor_y_pred_rej size: {tensor_y_pred_rej.size()}")
        print(f"tensor_y_acc size: {tensor_y_acc.size()}")
        print(f"tensor_y_pred_acc size: {tensor_y_pred_acc.size()}")

        print("------------------- Metrics ---------------------")
        proba = torch.nn.Softmax(dim=1)(tensor_y_pred_rej)[:, 1]
        val_auroc, val_aurpc = compute_AUC(tensor_y_rej, pred=proba)
        acc_rej = accuracy_score(tensor_y_rej.cpu().numpy(),
                                tensor_y_pred_rej.cpu().argmax(dim=1).numpy())
        print(f"Accuracy of the rejected samples: {acc_rej * 100} (%)")
        print(f"Val AUROC of the rejected samples: {val_auroc} (0-1)")

        proba = torch.nn.Softmax(dim=1)(tensor_y_pred_acc)[:, 1]
        val_auroc, val_aurpc = compute_AUC(tensor_y_acc, pred=proba)
        acc_acc = accuracy_score(tensor_y_acc.cpu().numpy(),
                                tensor_y_pred_acc.cpu().argmax(dim=1).numpy())
        print(f"Accuracy of the accepted samples: {acc_acc * 100} (%)")
        print(f"Val AUROC of the accepted samples: {val_auroc} (0-1)")

        # output_path = os.path.join(self.config.save_dir, "intermediate_tensors")
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)

        # torch.save(tensor_X_rej, os.path.join(output_path, f"iteration_{iteration}_{mode}_tensor_X.pt"))
        # torch.save(tensor_C_rej, os.path.join(output_path, f"iteration_{iteration}_{mode}_tensor_C.pt"))
        # torch.save(tensor_y_rej, os.path.join(output_path, f"iteration_{iteration}_{mode}_tensor_y.pt"))

        return (tensor_X_acc, tensor_C_acc, tensor_y_acc, fi,
                tensor_X_rej, tensor_C_rej, tensor_y_rej)

    def load_gt_train_tree(self, tree):
        self.gt_train_tree = tree

    def load_gt_val_tree(self, tree):
        self.gt_val_tree = tree