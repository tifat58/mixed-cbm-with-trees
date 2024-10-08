import torch.utils.data
from matplotlib import pyplot as plt

from epoch_trainers.xy_epoch_trainer import XY_Epoch_Trainer


class BlackBoxXYTrainer:

    def __init__(self, arch, config, device, data_loader, valid_data_loader,
                 reg=None, expert=None):

        self.arch = arch
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.expert = expert
        self.epochs = config['trainer']['epochs']
        self.acc_metrics_location = self.config.dir + "/accumulated_metrics.pkl"

        # create a new dataloader for the c->y model
        if self.config["arch"]["type"] == 'ParabolaArchitecture':
            train_data_loader = data_loader
            val_data_loader = valid_data_loader
        else:
            train_data_loader, val_data_loader = self.create_xy_dataloaders()

        # define the trainer
        self.reg = reg
        self.epoch_trainer = XY_Epoch_Trainer(
            self.arch, self.config,
            self.device, train_data_loader,
            val_data_loader)

    def train(self):

        logger = self.config.get_logger('train')

        # train the c->y model
        print("\nTraining x->y")
        logger.info("Training x->y")
        self.epoch_trainer._training_loop(self.epochs)
        self.plot()

    def test(self, test_data_loader, hard_cbm=None):

        # create a new dataloader only with x and y data
        test_data_loader = self.get_xy_test_dataloader(
            test_data_loader,
            batch_size=self.config["data_loader"]["args"]["batch_size"]
        )
        self.epoch_trainer._test(test_data_loader)

    def plot(self):
        results_trainer = self.epoch_trainer.metrics_tracker.result()
        train_target_losses = results_trainer['train_target_loss']
        val_target_losses = results_trainer['val_target_loss']
        train_accuracies = results_trainer['train_accuracy']
        val_accuracies = results_trainer['val_accuracy']

        # Plotting the results
        epochs = range(1, self.epochs + 1)
        plt.figure(figsize=(18, 30))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_target_losses, 'b', label='Training Target loss')
        plt.plot(epochs, val_target_losses, 'r', label='Validation Target loss')
        plt.title('Training and Validation Target Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Target Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
        plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        if self.expert is not None:
            plt.savefig(str(self.config.log_dir) + '/cy_plots_expert_' + str(self.expert) + '.png')
        else:
            plt.savefig(str(self.config.log_dir) + '/cy_plots.png')
        #plt.show()

    def create_xy_dataloaders(self):

        if isinstance(self.data_loader.dataset, torch.utils.data.TensorDataset):
            all_X = self.data_loader.dataset[:][0]
            all_y = self.data_loader.dataset[:][2]
            all_X_val = self.valid_data_loader.dataset[:][0]
            all_y_val = self.valid_data_loader.dataset[:][2]
            train_dataset = torch.utils.data.TensorDataset(all_X, all_y)
            val_dataset = torch.utils.data.TensorDataset(all_X_val, all_y_val)
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

    def get_xy_test_dataloader(self, test_data_loader, batch_size):

        if isinstance(test_data_loader.dataset, torch.utils.data.TensorDataset):
            X = test_data_loader.dataset[:][0]
            y = test_data_loader.dataset[:][2]
            test_data_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X, y),
                batch_size=batch_size,
                shuffle=False
            )
        else:
            raise ValueError("Only TensorDataset is supported for now")

        return test_data_loader


