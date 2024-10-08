import os
import pickle
import sys
import torch
from tqdm import tqdm

from loggers import XYLogger
from utils.util import get_correct
from base.epoch_trainer_base import EpochTrainerBase


class XY_Epoch_Trainer(EpochTrainerBase):
    """
    Trainer Epoch class using Tree Regularization
    """

    def __init__(self, arch, config, device,
                 data_loader, valid_data_loader=None,
                 lr_scheduler=None, expert=None):

        super(XY_Epoch_Trainer, self).__init__(arch, config, expert)

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
        self.epochs = config['trainer']['epochs']

        self.do_validation = self.val_loader is not None
        self.acc_metrics_location = self.config.dir + "/accumulated_metrics.pkl"
        # check if selective net is used
        if "selectivenet" in config.config.keys():
            self.selective_net = True
        else:
            self.selective_net = False

        self.metrics_tracker = XYLogger(config, expert=expert,
                                      tb_path=str(self.config.log_dir),
                                      output_path=str(self.config.save_dir),
                                      train_loader=self.train_loader,
                                      val_loader=self.val_loader,
                                      selectivenet=self.selective_net,
                                      device=self.device)
        self.metrics_tracker.begin_run()
        print("Device: ", self.device)

        self.optimizer = arch.optimizer
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def _train_epoch(self, epoch):

        print(f"Training epoch {epoch}")
        self.metrics_tracker.begin_epoch()
        self.model.train()
        if self.selective_net:
            self.arch.selector.train()
            self.arch.aux_model.train()

        tensor_X = torch.FloatTensor().to(self.device)
        tensor_y_pred = torch.FloatTensor().to(self.device)

        with tqdm(total=len(self.train_loader), file=sys.stdout) as t:
            for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
                batch_size = X_batch.size(0)
                X_batch = X_batch.to(self.device)
                tensor_X = torch.cat((tensor_X, X_batch), dim=0)
                y_batch = y_batch.to(self.device)

                # Forward pass
                y_pred = self.model(X_batch)
                tensor_y_pred = torch.cat((tensor_y_pred, y_pred), dim=0)
                outputs = {"prediction_out": y_pred}

                if self.selective_net:
                    out_selector = self.arch.selector(X_batch)
                    out_aux = self.arch.aux_model(X_batch)
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
        self.model.eval()
        if self.selective_net:
            self.arch.selector.eval()
            self.arch.aux_model.eval()

        tensor_X = torch.FloatTensor().to(self.device)
        tensor_y_pred = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            with tqdm(total=len(self.val_loader), file=sys.stdout) as t:
                for batch_idx, (X_batch, y_batch) in enumerate(self.val_loader):
                    batch_size = X_batch.size(0)
                    X_batch = X_batch.to(self.device)
                    tensor_X = torch.cat((tensor_X, X_batch), dim=0)
                    y_batch = y_batch.to(self.device)

                    # Forward pass
                    y_pred = self.model(X_batch)
                    tensor_y_pred = torch.cat((tensor_y_pred, y_pred), dim=0)
                    outputs = {"prediction_out": y_pred}

                    if self.selective_net:
                        out_selector = self.arch.selector(X_batch)
                        out_aux = self.arch.aux_model(X_batch)
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

    def _test(self, test_data_loader):

        self.model.eval()
        if self.selective_net:
            self.arch.selector.eval()
            self.arch.aux_model.eval()

        tensor_y_pred = torch.FloatTensor().to(self.device)
        tensor_y = torch.FloatTensor().to(self.device)
        test_metrics = {"loss": 0, "target_loss": 0, "accuracy": 0, "total_correct": 0}

        with torch.no_grad():
            with tqdm(total=len(test_data_loader), file=sys.stdout) as t:
                for batch_idx, (X_batch, y_batch) in enumerate(test_data_loader):

                    batch_size = X_batch.size(0)
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    tensor_y = torch.cat((tensor_y, y_batch), dim=0)

                    # Forward pass
                    y_pred = self.model(X_batch)
                    tensor_y_pred = torch.cat((tensor_y_pred, y_pred), dim=0)
                    outputs = {"prediction_out": y_pred}

                    # Calculate Label losses
                    test_metrics["total_correct"] += get_correct(y_pred, y_batch, self.config["dataset"]["num_classes"])
                    loss_label = self.criterion(outputs, y_batch)
                    test_metrics["target_loss"] += loss_label["target_loss"].detach().cpu().item() * batch_size

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
        with open(os.path.join(self.config.save_dir, f"test_metrics_xtoy.pkl"), "wb") as f:
            pickle.dump(test_metrics, f)

        # print test metrics
        print("Test Metrics:")
        print(f"Loss: {test_metrics['loss']}")
        print(f"Accuracy: {test_metrics['accuracy']}")
        print(f"Target Loss: {test_metrics['target_loss']}")

        # put also in the logger info
        self.logger.info(f"Test Metrics:")
        self.logger.info(f"Loss: {test_metrics['loss']}")
        self.logger.info(f"Accuracy: {test_metrics['accuracy']}")
        self.logger.info(f"Target Loss: {test_metrics['target_loss']}")

        # save the predictions
        tensor_y_pred = tensor_y_pred.cpu()
        tensor_y = tensor_y.cpu()
        if self.config["dataset"]["num_classes"] == 1:
            y_hat = torch.sigmoid(tensor_y_pred)
            y_hat = [1 if y_hat[i] >= 0.5 else 0 for i in range(len(y_hat))]
            correct = [1 if y_hat[i] == tensor_y[i] else 0 for i in range(len(y_hat))]
        else:
            correct = tensor_y_pred.argmax(dim=1).eq(tensor_y)
        with open(os.path.join(self.config.save_dir, f"test_pred_correct_or_not_xtoy.pkl"), "wb") as f:
            pickle.dump(correct, f)