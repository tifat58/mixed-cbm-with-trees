import torch
from sklearn.metrics import accuracy_score

from trainers import IndependentCBMTrainer, McbmTrainer, \
    SequentialCBMTrainer
import importlib
import copy

class MoMcbmTrainer:

    def __init__(self, arch, config, device, data_loader,
                 valid_data_loader, reg=None):

        self.arch = arch
        self.config = config
        self.config_selectivenet = self.config._main_config
        self.config_leakage_inspection = self.config.rest_configs[0]
        self.device = device
        self.init_train_data_loader = data_loader
        self.init_valid_data_loader = valid_data_loader
        self.reg = reg
        self.acc_metrics_location = self.config.save_dir + "/accumulated_metrics.pkl"

        self.num_experts = self.config['trainer']['num_experts']
        self.coverage_per_expert = self.config['selectivenet']['coverage']
        self.experts_selectivenet = {}
        self.experts_leakage_inspection = {}

    def train(self):

        for expert in range(1, self.num_experts + 1):
            self.train_expert(expert)

    def test(self, test_data_loader, hard_cbm=False):

        logger = self.config.get_logger('trainer')
        y_all = []
        y_pred_all = []
        for expert in range(1, self.num_experts + 1):
            test_data_loader, y, y_pred = self.test_expert(expert, test_data_loader)
            y_all.extend(y)
            y_pred_all.extend(y_pred)

        print(f'\nTest Accuracy of the complete algorithm: {accuracy_score(y_all, y_pred_all)}')
        logger.info(f'\nTest Accuracy of the complete algorithm: {accuracy_score(y_all, y_pred_all)}')

    def train_expert(self, expert):

        if expert > 1:
            train_loader = self.train_data_loader
            valid_loader = self.valid_data_loader
            arch = self.init_selectivenet_module(train_loader)
        else:
            arch = self.arch
            train_loader = self.init_train_data_loader
            valid_loader = self.init_valid_data_loader

        # define first expert as independent cbm with selectivenet
        config = copy.deepcopy(self.config)
        config._main_config["selectivenet"]["coverage"] = self.coverage_per_expert[expert-1]
        arch.criterion_label.coverage = self.coverage_per_expert[expert-1]
        expert_selectivenet = SequentialCBMTrainer(
            arch, config, self.device,
            train_loader, valid_loader,
            reg=None, expert=expert
        )
        self.logger = self.config.get_logger('trainer')
        self.logger.info('\n')
        self.logger.info(f'Expert {expert} - Training Hard CBM with SelectiveNet ...')
        print(f'\nExpert {expert} - Training Hard CBM with SelectiveNet ...')
        expert_selectivenet.train()
        # best_model_path = self.first_expert.cy_epoch_trainer.checkpoint_dir + '/model_best.pth'
        # checkpoint = torch.load(best_model_path)
        # model_state_dict = checkpoint['state_dict']
        # selector_state_dict = checkpoint['selector']
        # arch.model.load_state_dict(model_state_dict)
        # arch.selector.load_state_dict(selector_state_dict)
        self.experts_selectivenet[expert] = expert_selectivenet

        # get the selected train and val data
        (tensor_X_train, tensor_C_train, tensor_y_acc_train, fi_train, tree_train,
         tensor_X_rej_train, tensor_C_rej_train, tensor_y_rej_train,)  \
            = expert_selectivenet.cy_epoch_trainer._save_selected_results(
            loader=train_loader, expert=expert, mode="train", arch=arch)

        (tensor_X, tensor_C, tensor_y_acc, fi, tree,
        tensor_X_rej_val, tensor_C_rej_val, tensor_y_rej_val,) \
            = expert_selectivenet.cy_epoch_trainer._save_selected_results(
            loader=valid_loader, expert=expert, mode="valid", arch=arch)

        # create a new dataloader with the selected samples
        new_dataset = torch.utils.data.TensorDataset(tensor_X_train, tensor_C_train, tensor_y_acc_train)
        new_train_data_loader = torch.utils.data.DataLoader(
            new_dataset, batch_size=self.config['data_loader']['args']['batch_size'], shuffle=True
        )

        # create a new dataloader with the selected samples
        new_dataset = torch.utils.data.TensorDataset(tensor_X, tensor_C, tensor_y_acc)
        new_valid_data_loader = torch.utils.data.DataLoader(
            new_dataset, batch_size=self.config['data_loader']['args']['batch_size'], shuffle=False)

        self.logger.info('\n')
        self.logger.info(f'Expert {expert} - Performing Leakage Inspection...')
        print(f'\nExpert {expert} - Performing Leakage Inspection...')

        # Get the models used for leakage inspection
        arch = self.init_leakage_inspection_module(new_train_data_loader)
        config = copy.deepcopy(self.config)
        expert_leakage_inspection = McbmTrainer(
            arch, config, self.device,
            new_train_data_loader, new_valid_data_loader,
            reg=None, expert=expert
        )
        expert_leakage_inspection.train()
        self.experts_leakage_inspection[expert] = expert_leakage_inspection

        # create a new dataloader with the rejected samples
        new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_train, tensor_C_rej_train, tensor_y_rej_train)
        if len(new_dataset) != 0:
            self.train_data_loader = torch.utils.data.DataLoader(
                new_dataset, batch_size=self.config['data_loader']['args']['batch_size'], shuffle=True
            )

        # create a new dataloader with the rejected samples
        new_dataset = torch.utils.data.TensorDataset(tensor_X_rej_val, tensor_C_rej_val, tensor_y_rej_val)
        if len(new_dataset) != 0:
            self.valid_data_loader = torch.utils.data.DataLoader(
                new_dataset, batch_size=self.config['data_loader']['args']['batch_size'], shuffle=False
            )

    def test_expert(self, expert, test_dataloader):

        self.logger = self.config.get_logger('trainer')

        # Get the accepted and rejected test samples of this expert
        self.logger.info('\n')
        self.logger.info(f'Expert {expert} - Testing Hard CBM with SelectiveNet ...')
        print(f'\nExpert {expert} - Testing Hard CBM with SelectiveNet ...')
        cy_epoch_trainer = self.experts_selectivenet[expert].cy_epoch_trainer

        (tensor_X, tensor_C, tensor_y_acc, fi, tree, tensor_X_rej, tensor_C_rej, tensor_y_rej)  \
            = cy_epoch_trainer._get_predictions_from_selector(
            loader=test_dataloader, expert_idx=expert, mode="test", expert=cy_epoch_trainer
        )

        # create a new dataloader with the selected samples
        new_dataset = torch.utils.data.TensorDataset(tensor_X, tensor_C, tensor_y_acc)
        new_data_loader = torch.utils.data.DataLoader(
            new_dataset, batch_size=self.config['data_loader']['args']['batch_size'], shuffle=False
        )

        # Do leakage Inspection
        self.logger.info('\n')
        self.logger.info(f'Expert {expert} - Performing Leakage Inspection...')
        print(f'\nExpert {expert} - Performing Leakage Inspection...')
        y, y_pred = self.experts_leakage_inspection[expert].test(new_data_loader)

        # create a new dataloader with the rejected samples
        new_dataset = torch.utils.data.TensorDataset(tensor_X_rej, tensor_C_rej, tensor_y_rej)
        if len(new_dataset) != 0:
            test_dataloader = torch.utils.data.DataLoader(
                new_dataset, batch_size=self.config['data_loader']['args']['batch_size'], shuffle=True
            )
        else:
            test_dataloader = None

        return test_dataloader, y, y_pred


    def init_selectivenet_module(self, train_loader):
        arch_module = importlib.import_module("architectures")
        self.config._main_config = self.config_selectivenet
        arch = self.config.init_obj('arch', arch_module, self.config,
                                    device=self.device, data_loader=train_loader)
        self.logger.info('\n')
        self.logger.info(arch.model)
        return arch

    def init_leakage_inspection_module(self, train_loader):
        arch_module = importlib.import_module("architectures")
        self.config._main_config = self.config_leakage_inspection
        arch = self.config.init_obj('arch', arch_module, self.config,
                                    device=self.device, data_loader=train_loader)
        self.logger.info('\n')
        self.logger.info(arch.model)
        return arch
        