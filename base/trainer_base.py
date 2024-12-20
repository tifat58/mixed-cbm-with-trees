import numpy as np
import torch
from abc import abstractmethod
from numpy import inf
import os


class TrainerBase:
    """
    Base class for all trainers
    """

    def __init__(self, arch, config, expert=None):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.arch = arch
        self.model = arch.model
        if hasattr(arch, "optimizer"):
            self.optimizer = arch.optimizer
        else:
            self.optimizer = arch.cy_optimizer
        self.expert = expert

        cfg_trainer = config['trainer']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 0
        if self.expert is not None:
            self.checkpoint_dir = str(
                config.save_dir) + f'/expert_{self.expert}'
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
        else:
            self.checkpoint_dir = str(config.save_dir)
        self.last_checkpoint_path = None  # Track the last checkpoint file

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        :return: A log that contains information about training
        """
        raise NotImplementedError

    def _training_loop(self, epochs):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, epochs):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            self.logger.info('\n')
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                self.logger.info('    ' + f'{str(key)}: ' + f'{value}')

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not,
                    # according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is "
                                        "disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn\'t improve for {} epochs. "
                        "Training stops.".format(self.early_stop))
                    break

                if (epoch % self.save_period == 0 or best) and (self.mnt_mode != 'off'):
                    self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to
        'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if hasattr(self.arch, "selector"):
            state['selector'] = self.arch.selector.state_dict()

        filename = self.checkpoint_dir + f'/checkpoint-epoch{epoch}.pth'
        torch.save(state, filename)
        self.logger.info(f"Saving checkpoint: {filename} ...")

        # Delete the previous checkpoint if it exists
        if self.last_checkpoint_path and os.path.exists(
                self.last_checkpoint_path):
            os.remove(self.last_checkpoint_path)
            self.logger.info(
                f"Deleted previous checkpoint: {self.last_checkpoint_path}")

        # Update the last checkpoint path
        self.last_checkpoint_path = filename

        if save_best:
            best_path = self.checkpoint_dir + '/model_best.pth'
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")
            print(
                f"Best model found at epoch: {epoch} with "
                f"{self.mnt_metric}: {self.mnt_best}")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is "
                "different from that of "
                "checkpoint. This may yield an exception while state_dict is "
                "being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is
        # not changed.
        if checkpoint['config']['optimizer']['type'] != \
                self.config['optimizer']['type']:
            self.logger.warning(
                "Warning: Optimizer type given in config file is different "
                "from that of checkpoint. "
                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(
                self.start_epoch))
