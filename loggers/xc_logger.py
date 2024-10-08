import os.path
import time

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from utils import *


class XCLogger:
    def __init__(self, config, iteration, tb_path, output_path, train_loader,
                 val_loader, device=None):
        """
        Initialized each parameters of each run.
        """
        self.iteration = iteration
        self.tb_path = tb_path + '/xc_logger'
        self.output_path = output_path + '/xc_model'
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epoch_id = 0
        self.best_epoch_id = 0

        self.n_classes = config['dataset']['num_classes']
        self.concept_names = config['dataset']['concept_names']
        self.concept_group_names = list(self.concept_names.keys())
        self.n_concept_groups = len(list(self.concept_names.keys()))

        self.run_id = 0
        self.run_data = []
        self.run_start_time = None
        self.epoch_duration = None

        self.tb = None
        self.val_best_accuracy = 0
        self.best_val_loss = 1000000
        self.val_auroc = None

        self.attributes_per_epoch = {
            "train_loss": 0,
            "val_loss": 0,
            "train_concept_loss": 0,
            "val_concept_loss": 0,
            "train_total_correct": 0,
            "val_total_correct": 0,
        }
        self.train_accuracy = None
        self.val_accuracy = None

        self.list_attributes_per_epoch = {
            "train_loss_per_concept": np.zeros(self.n_concept_groups),
            "val_loss_per_concept": np.zeros(self.n_concept_groups),
            "train_accuracy_per_concept": np.zeros(self.n_concept_groups),
            "val_accuracy_per_concept": np.zeros(self.n_concept_groups),
        }

        self.all_epoch_attributes = {key: [] for key in
                                     self.attributes_per_epoch}
        self.all_epoch_attributes["train_accuracy"] = []
        self.all_epoch_attributes["val_accuracy"] = []

        self.all_epoch_list_attributes = {key: [] for key in
                                          self.list_attributes_per_epoch}

    def begin_run(self):
        """
        Records all the parameters at the start of each run.
        :return: none
        """
        self.run_start_time = time.time()

        self.run_id += 1
        self.tb = SummaryWriter(f"{self.tb_path}")
        # print("################## TB Log path ###################")
        # print(f"{self.tb_path}")
        # print("################## TB Log path ###################")

    def end_run(self):
        """
        Records all the parameters at the end of each run.

        :return: none
        """
        self.tb.close()
        self.epoch_id = 0

    def begin_epoch(self):
        for key in self.attributes_per_epoch:
            self.attributes_per_epoch[key] = 0
        for key in self.list_attributes_per_epoch:
            self.list_attributes_per_epoch[key].fill(0)

        self.train_accuracy = None
        self.val_accuracy = None

        self.epoch_id += 1

    def update_batch(self, update_dict_or_key, batch_size, value=None,
                     mode='train'):
        prefix = "train_" if mode == 'train' else "val_"
        if isinstance(update_dict_or_key, dict):
            for key, val in update_dict_or_key.items():
                full_key = prefix + key
                if full_key in self.attributes_per_epoch:
                    val = val.detach().item()
                    self.attributes_per_epoch[full_key] += val * batch_size
                elif full_key in self.list_attributes_per_epoch:
                    val = val.detach()
                    self.list_attributes_per_epoch[full_key] += np.array(
                        [x * batch_size for x in val])
        elif isinstance(update_dict_or_key, str) and value is not None:
            full_key = prefix + update_dict_or_key
            if full_key in self.attributes_per_epoch:
                self.attributes_per_epoch[full_key] += value * batch_size
            elif full_key in self.list_attributes_per_epoch:
                self.list_attributes_per_epoch[full_key] += np.array(
                    [x * batch_size for x in value])
        else:
            raise ValueError(
                "Invalid input: expected a dictionary or a key-value pair")

    def end_epoch(self):

        for key in self.list_attributes_per_epoch:
            if key.startswith("train_"):
                dataset_length = len(self.train_loader.dataset)
            elif key.startswith("val_"):
                dataset_length = len(self.val_loader.dataset)
            else:
                raise ValueError("Invalid key")
            self.list_attributes_per_epoch[key] = self.list_attributes_per_epoch[key] / dataset_length
            self.all_epoch_list_attributes[key].append((self.list_attributes_per_epoch[key]).tolist())

        for key in self.attributes_per_epoch:
            if key.startswith("train_"):
                dataset_length = len(self.train_loader.dataset)
            elif key.startswith("val_"):
                dataset_length = len(self.val_loader.dataset)
            else:
                raise ValueError("Invalid key: ", key)
            self.attributes_per_epoch[key] /= dataset_length
            self.all_epoch_attributes[key].append(
                self.attributes_per_epoch[key])

        # self.tb.add_scalar("Epoch_stats_model/Train_accuracy",
        #                    self.train_accuracy, self.epoch_id)
        # self.tb.add_scalar("Epoch_stats_model/Val_accuracy", self.val_accuracy,
        #                    self.epoch_id)

        self.tb.add_scalar("XC_Logger/Train Loss",
                           self.attributes_per_epoch['train_loss'],
                           self.epoch_id)
        self.tb.add_scalar("XC_Logger/Val Loss",
                           self.attributes_per_epoch['val_loss'], self.epoch_id)

        self.train_accuracy = self.list_attributes_per_epoch['train_accuracy_per_concept'].mean()
        self.val_accuracy = self.list_attributes_per_epoch['val_accuracy_per_concept'].mean()
        self.all_epoch_attributes["train_accuracy"].append(self.train_accuracy)
        self.all_epoch_attributes["val_accuracy"].append(self.val_accuracy)

        self.tb.add_scalar("XC_Logger/Train Accuracy",
                           self.train_accuracy,
                           self.epoch_id)
        self.tb.add_scalar("XC_Logger/Val Accuracy",
                           self.val_accuracy, self.epoch_id)
        # report concept losses and concept accuracies
        for i in range(self.n_concept_groups):
            self.tb.add_scalar(f"XC_Logger Train Loss Per Concept/Train Loss Concept {self.concept_group_names[i]}",
                               self.list_attributes_per_epoch['train_loss_per_concept'][i],
                               self.epoch_id)
            self.tb.add_scalar(f"XC_Logger Val Loss Per Concept/Val Loss Concept {self.concept_group_names[i]}",
                               self.list_attributes_per_epoch['val_loss_per_concept'][i],
                               self.epoch_id)
            self.tb.add_scalar(f"XC_Logger Train Accuracy Per Concept/Train Accuracy Concept {self.concept_group_names[i]}",
                               self.list_attributes_per_epoch['train_accuracy_per_concept'][i],
                               self.epoch_id)
            self.tb.add_scalar(f"XC_Logger Val Accuracy Per Concept/Val Accuracy Concept {self.concept_group_names[i]}",
                               self.list_attributes_per_epoch['val_accuracy_per_concept'][i],
                               self.epoch_id)

    def __str__(self):
        return f"Attributes per epoch: {self.attributes_per_epoch}\nList Attributes per epoch: {self.list_attributes_per_epoch}"


    def track_total_train_correct_per_epoch_per_concept(self, preds, labels):
        """
        Calculates the correct prediction at the each iteration of batch.

        :param preds: predicted labels
        :param labels: true labels

        :return: the totalcorrect prediction at the each iteration of batch
        """
        correct_per_column = column_get_correct(preds, labels, self.concept_names)
        self.list_attributes_per_epoch["train_accuracy_per_concept"] += np.array(
            [x for x in correct_per_column])

    def track_total_val_correct_per_epoch_per_concept(self, preds, labels):
        """
        Calculates the correct prediction at the each iteration of batch.

        :param preds: predicted labels
        :param labels: true labels

        :return: the totalcorrect prediction at the each iteration of batch
        """
        correct_per_column = column_get_correct(preds, labels, self.concept_names)
        self.list_attributes_per_epoch["val_accuracy_per_concept"] += np.array(
            [x for x in correct_per_column])

    def result(self):
        performance_dict = {**self.all_epoch_list_attributes,
                            **self.all_epoch_attributes}
        performance_df = pd.DataFrame(
            dict([(col_name, pd.Series(values)) for col_name, values in
                  performance_dict.items()])
        )
        performance_df.to_csv(
            os.path.join(self.tb_path, "train_val_stats") + ".csv")
        return performance_dict

    def result_epoch(self):
        performance_dict = {**self.attributes_per_epoch,
                            **self.list_attributes_per_epoch}
        # Add the values to the dictionary
        performance_dict['train_accuracy'] = self.train_accuracy
        performance_dict['val_accuracy'] = self.val_accuracy
        return performance_dict
