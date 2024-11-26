import argparse
import collections
import torch
import numpy as np
from utils.parse_config import ConfigParser
from utils import prepare_device
import importlib


# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.random.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    dataloaders_module = importlib.import_module("data_loaders")
    data_loader, valid_data_loader, test_data_loader = config.init_obj(
        'data_loader', dataloaders_module, config=config
    )

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])

    # build model architecture, then print to console
    arch_module = importlib.import_module("architectures")
    arch = config.init_obj('arch', arch_module, config=config, device=device,
                           data_loader=data_loader)
    logger.info("\n")
    logger.info(arch.model)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    # lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainers = importlib.import_module("trainers")
    trainer = config.init_obj('trainer', trainers,
                              arch=arch,
                              config=config,
                              device=device,
                              data_loader=data_loader,
                              valid_data_loader=valid_data_loader
                              )

    trainer.train()
    print("\nTraining completed")
    print("Starting testing ...")
    logger.info("\n")
    logger.info("Training completed")

    if config["trainer"]['type'] == 'IndependentCBMTrainer':
        hard_cbm = config["trainer"]['hard_cbm']
        if isinstance(hard_cbm, int):
            hard_cbm = bool(hard_cbm)
    else:
        hard_cbm = False

    logger.info("Starting testing ...")
    trainer.test(test_data_loader, hard_cbm=hard_cbm)
    print("\nTesting completed")
    logger.info("\n")
    logger.info("Testing completed")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Imperial Diploma Project')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--n', '--name'], type=str, target='name'),
        CustomArgs(['--sd', '--save_dir'], type=str, target='trainer;save_dir'),
        CustomArgs(['--pretrained_concept_predictor'],
                     type=str, target='model;pretrained_concept_predictor'),
        CustomArgs(['--pretrained_concept_predictor_joint'],
                   type=str, target='model;pretrained_concept_predictor_joint'),
        CustomArgs(['--msl', '--min_samples_leaf'],
                   type=int, target='regularisation;min_samples_leaf'),
        CustomArgs(['--entropy_layer', '--entropy_layer'],
                     type=str, target='model;entropy_layer'),
        CustomArgs(['--tau', '--tau'],
                   type=float, target='model;tau'),
        CustomArgs(['--lm', '--lm'],
                   type=float, target='model;lm'),
        CustomArgs(['--hard_cbm', '--hard_cbm'],
                   type=int, target='trainer;hard_cbm'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
