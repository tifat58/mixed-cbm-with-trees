import os
import pickle

import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score
from tqdm import tqdm
import sys

from loggers.xc_logger import XCLogger

from base.epoch_trainer_base import EpochTrainerBase
from utils import column_get_correct, sigmoid_or_softmax_with_groups, \
    identify_misclassified_samples, convert_to_categorical, update_pickle_dict, \
    probs_to_binary


class XC_Epoch_Trainer(EpochTrainerBase):
    """
    Trainer Epoch class using Tree Regularization
    """

    def __init__(self, arch, config, device, data_loader, valid_data_loader=None):

        super(XC_Epoch_Trainer, self).__init__(arch, config)

        # Extract the configuration parameters
        self.config = config
        self.device = device
        self.train_loader = data_loader
        self.val_loader = valid_data_loader
        self.arch = arch
        self.lr_scheduler = arch.lr_scheduler
        self.model = arch.model.to(self.device)
        self.criterion = arch.criterion_concept
        self.min_samples_leaf = config['regularisation']['min_samples_leaf']
        self.concept_names = config['dataset']['concept_names']
        self.n_concept_groups = len(list(self.concept_names.keys()))

        self.do_validation = self.val_loader is not None
        self.acc_metrics_location = self.config.dir + "/accumulated_metrics.pkl"

        # Initialize the metrics tracker
        self.metrics_tracker = XCLogger(config, iteration=1,
                                       tb_path=str(self.config.log_dir),
                                       output_path=str(self.config.save_dir),
                                       train_loader=self.train_loader,
                                       val_loader=self.val_loader,
                                       device=self.device)
        self.metrics_tracker.begin_run()
        print("Device: ", self.device)

        self.optimizer = arch.xc_optimizer
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)


    def _train_epoch(self, epoch):

        print(f"Training Epoch {epoch}")
        self.metrics_tracker.begin_epoch()
        self.model.concept_predictor.train()

        with tqdm(total=len(self.train_loader), file=sys.stdout) as t:
            for batch_idx, (X_batch, C_batch, y_batch) in enumerate(self.train_loader):
                batch_size = X_batch.size(0)
                X_batch = X_batch.to(self.device)
                C_batch = C_batch.to(self.device)

                # Forward pass
                C_pred_soft = self.model.concept_predictor(X_batch)
                C_pred_soft = C_pred_soft[:, self.arch.selected_concepts]

                # Track target training loss and accuracy
                self.metrics_tracker.track_total_train_correct_per_epoch_per_concept(
                    preds=C_pred_soft.detach().cpu(), labels=C_batch.detach().cpu()
                )
                #C_pred = torch.sigmoid(C_pred_soft)
                C_pred = sigmoid_or_softmax_with_groups(C_pred_soft, self.concept_names)

                # Calculate Concept losses
                loss_concept_total, loss_per_concept = self.criterion(C_pred, C_batch)
                self.metrics_tracker.update_batch(update_dict_or_key='concept_loss',
                                                  value=loss_concept_total.detach().cpu().item(),
                                                  batch_size=batch_size,
                                                  mode='train')
                self.metrics_tracker.update_batch(update_dict_or_key='loss_per_concept',
                                                  value=loss_per_concept,
                                                  batch_size=batch_size,
                                                  mode='train')

                self.optimizer.zero_grad()
                loss_concept_total.backward()
                self.optimizer.step()
                self.metrics_tracker.update_batch(update_dict_or_key='loss',
                                                  value=loss_concept_total.detach().cpu().item(),
                                                  batch_size=batch_size,
                                                  mode='train')

                t.set_postfix(
                    batch_id='{0}'.format(batch_idx + 1))
                t.update()

        if self.do_validation:
            self._valid_epoch(epoch)

        # Update the epoch metrics
        self.metrics_tracker.end_epoch()
        log = self.metrics_tracker.result_epoch()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log


    def _valid_epoch(self, epoch):

        print(f"Validation Epoch {epoch}")
        self.model.concept_predictor.eval()

        with torch.no_grad():
            with tqdm(total=len(self.val_loader), file=sys.stdout) as t:
                for batch_idx, (X_batch, C_batch, y_batch) in enumerate(self.val_loader):
                    batch_size = X_batch.size(0)
                    X_batch = X_batch.to(self.device)
                    C_batch = C_batch.to(self.device)

                    # Forward pass
                    C_pred_soft = self.model.concept_predictor(X_batch)
                    C_pred_soft = C_pred_soft[:, self.arch.selected_concepts]

                    # Track target training loss and accuracy
                    self.metrics_tracker.track_total_val_correct_per_epoch_per_concept(
                        preds=C_pred_soft.detach().cpu(), labels=C_batch.detach().cpu()
                    )
                    # C_pred = torch.sigmoid(C_pred_soft)
                    C_pred = sigmoid_or_softmax_with_groups(C_pred_soft, self.concept_names)

                    # Calculate Concept losses
                    loss_concept_total, loss_per_concept = self.criterion(C_pred, C_batch)
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

                    # Track target training loss and accuracy
                    # self.metrics_tracker.track_total_val_correct_per_epoch(
                    #     preds=outputs["prediction_out"], labels=y_batch
                    # )

                    self.metrics_tracker.update_batch(update_dict_or_key='loss',
                                                      value=loss_concept_total.detach().cpu().item(),
                                                      batch_size=batch_size,
                                                      mode='val')

                    t.set_postfix(
                        batch_id='{0}'.format(batch_idx + 1))
                    t.update()

    def _test(self, test_data_loader, hard_cbm=False, categorise=False):

        self.model.concept_predictor.eval()
        #tensor_X = torch.FloatTensor().to(self.device)
        tensor_C_pred = torch.FloatTensor().to(self.device)
        tensor_C = torch.FloatTensor().to(self.device)
        tensor_y = torch.LongTensor().to(self.device)

        test_metrics = {"concept_loss": 0, "loss_per_concept": np.zeros(self.n_concept_groups), "total_correct": 0,
                        "accuracy_per_concept": np.zeros(self.n_concept_groups)}

        if self.config["trainer"]["save_test_tensors"] is True:
            misclassified_examples = {}
            correctly_classified_examples = {}

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

                    samples = identify_misclassified_samples(C_pred, C_batch, self.concept_names)
                    if self.config["trainer"]["save_test_tensors"] is True:
                        for i in range(X_batch.shape[0]):
                            if samples[i]:  # Check if any class is misclassified
                                sample_idx = batch_idx * test_data_loader.batch_size + i
                                misclassified_examples[sample_idx] = {
                                    'X': X_batch[i].cpu().numpy(),
                                    "C": C_batch[i].cpu().numpy(),
                                    'C_pred': C_pred[i].cpu().numpy()
                                }
                            else:
                                sample_idx = batch_idx * test_data_loader.batch_size + i
                                correctly_classified_examples[sample_idx] = {
                                    'X': X_batch[i].cpu().numpy(),
                                    "C": C_batch[i].cpu().numpy(),
                                    'C_pred': C_pred[i].cpu().numpy()
                                }

                    # C_pred = torch.sigmoid(C_pred_soft)
                    C_pred = sigmoid_or_softmax_with_groups(C_pred, self.concept_names)
                    #tensor_X = torch.cat((tensor_X, X_batch), dim=0)
                    tensor_C_pred = torch.cat((tensor_C_pred, C_pred), dim=0)
                    tensor_y = torch.cat((tensor_y, y_batch), dim=0)
                    tensor_C = torch.cat((tensor_C, C_batch), dim=0)

                    # Calculate Concept losses
                    loss_concept_total, loss_per_concept = self.criterion(C_pred, C_batch)

                    test_metrics["concept_loss"] += loss_concept_total.detach().cpu().item() * batch_size
                    test_metrics["loss_per_concept"] += np.array([x * batch_size for x in loss_per_concept])

                    t.set_postfix(
                        batch_id='{0}'.format(batch_idx + 1))
                    t.update()

        # Update the test metrics
        test_metrics["concept_loss"] /= len(test_data_loader.dataset)
        test_metrics["loss_per_concept"] = [x / len(test_data_loader.dataset) for x in test_metrics["loss_per_concept"]]
        test_metrics["accuracy_per_concept"] = [x / len(test_data_loader.dataset) for x in test_metrics["accuracy_per_concept"]]
        test_metrics["concept_accuracy"] = sum(test_metrics["accuracy_per_concept"]) / len(test_metrics["accuracy_per_concept"])

        # save test metrics in pickle
        with open(os.path.join(self.config.save_dir, f"test_metrics_xtoc.pkl"), "wb") as f:
            pickle.dump(test_metrics, f)

        if self.config["trainer"]["save_test_tensors"] is True:
            # Save the misclassified examples to a pickle file
            with open(os.path.join(self.config.save_dir, f"missclassified_test_samples.pkl"), 'wb') as f:
                pickle.dump(misclassified_examples, f)

            # Save the correctly classified examples to a pickle file
            with open(os.path.join(self.config.save_dir, f"correctly_classified_test_samples.pkl"), 'wb') as f:
                pickle.dump(correctly_classified_examples, f)

        # print test metrics
        print("Test Metrics:")
        print(f"Concept Loss: {test_metrics['concept_loss']}")
        print(f"Loss per Concept: {test_metrics['loss_per_concept']}")
        print(f"Concept Accuracy: {test_metrics['concept_accuracy']}")
        print(f"Accuracy per Concept: {test_metrics['accuracy_per_concept']}")

        # put also in the logger info
        self.logger.info(f"Test Metrics:")
        self.logger.info(f"Concept Loss: {test_metrics['concept_loss']}")
        self.logger.info(f"Loss per Concept: {test_metrics['loss_per_concept']}")
        self.logger.info(f"Concept Accuracy: {test_metrics['concept_accuracy']}")
        self.logger.info(f"Accuracy per Concept: {test_metrics['accuracy_per_concept']}")

        # update pickle dict
        update_pickle_dict(self.acc_metrics_location, self.config.exper_name,
                           self.config.run_id, 'concept_acc',
                           float(test_metrics['concept_accuracy']))

        # if we use a hard-cbm, convert the predictions to binary
        #tensor_C_pred = torch.sigmoid(tensor_C_pred)
        tensor_C_pred_categorical = convert_to_categorical(tensor_C_pred.detach().cpu().numpy(), self.concept_names)

        # output_path = os.path.join(self.config.save_dir, "test_tensors")
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)
        #
        # tensor_X = tensor_X.cpu()
        # tensor_C_pred = tensor_C_pred.cpu()
        # tensor_y = tensor_y.cpu()
        #
        # torch.save(tensor_X, os.path.join(output_path, f"test_tensor_X.pt"))
        # torch.save(tensor_C_pred, os.path.join(output_path, f"test_tensor_C_binarised.pt"))
        # torch.save(tensor_y, os.path.join(output_path, f"test_tensor_y.pt"))
        # print(f"\nSaved test tensors in {output_path}")

        if hard_cbm:
            if categorise:
                tensor_C_pred = tensor_C_pred_categorical
            else:
                tensor_C_pred = probs_to_binary(tensor_C_pred.detach().cpu().numpy(), self.concept_names)

        return tensor_C_pred, tensor_y

    def _predict(self, X=None, data_loader=None, use_data_loader=True,
                 categorise=False, temperatures=None):

        if use_data_loader:
            assert data_loader is not None and X is None
        else:
            assert data_loader is None and X is not None

        self.model.concept_predictor.eval()

        if use_data_loader:
            tensor_C_pred = torch.FloatTensor().to(self.device)
            tensor_y = torch.LongTensor().to(self.device)

            with torch.no_grad():
                with tqdm(total=len(data_loader), file=sys.stdout) as t:
                    for batch_idx, (X_batch, C_batch, y_batch) in enumerate(data_loader):

                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        # Forward pass
                        C_pred = self.model.concept_predictor(X_batch)
                        C_pred = C_pred[:, self.arch.selected_concepts]
                        tensor_C_pred = torch.cat((tensor_C_pred, C_pred), dim=0)
                        tensor_y = torch.cat((tensor_y, y_batch), dim=0)

                        t.set_postfix(
                            batch_id='{0}'.format(batch_idx + 1))
                        t.update()

            # if we use a hard-cbm, convert the predictions to binary
            #tensor_C_pred = torch.sigmoid(tensor_C_pred)
            tensor_C_pred = sigmoid_or_softmax_with_groups(tensor_C_pred, self.concept_names)
            if categorise:
                tensor_C_pred = convert_to_categorical(tensor_C_pred.detach().cpu().numpy(), self.concept_names)
            else:
                tensor_C_pred = tensor_C_pred.cpu().numpy()
            return tensor_C_pred, tensor_y
        else:
            X = X.to(self.device)
            with torch.no_grad():
                C_pred = self.model.concept_predictor(X)

            if temperatures is not None:
                C_pred = temperatures(C_pred)
                #C_pred = C_pred / temperatures

            C_pred = C_pred[:, self.arch.selected_concepts]
            # C_pred = torch.sigmoid(C_pred).cpu().numpy()
            C_pred = sigmoid_or_softmax_with_groups(C_pred, self.concept_names)
            if categorise:
                C_pred = convert_to_categorical(C_pred.detach().cpu().numpy(), self.concept_names)
            else:
                C_pred = C_pred.cpu().numpy()
            return C_pred
