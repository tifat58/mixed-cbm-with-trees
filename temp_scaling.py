import importlib
import os
from networks.temperature_scaling import ModelWithTemperature

import argparse
import collections
import torch
import numpy as np

from utils import prepare_device, create_group_indices
from utils.parse_config import ConfigParser


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

    # Now we're going to wrap the model with a decorator that adds temperature scaling
    concept_names = config['dataset']['concept_names']
    model = ModelWithTemperature(arch.model, concept_names, device)

    # Tune the model temperature, and save the results
    model.set_temperature(valid_data_loader)
    model_filename = os.path.join(config.save_dir, 'model_with_temperature.pth')
    state = {
        'state_dict': model.model.state_dict(),
        'whole_model': model,
    }
    torch.save(state, model_filename)
    print('Temperature scaled model sved to %s' % model_filename)
    print('Done!')


    # group_indices = create_group_indices(concept_names)
    # num_classes = config['dataset']['num_classes']
    # bin_keys = model.get_bin_keys()
    # samples_per_group_before = {}
    # samples_per_group_after = {}
    # confidences_and_predictions_before = {}
    # confidences_and_predictions_after = {}
    # for group, indices in group_indices.items():
    #     samples_per_group_before[group] = {key: 0 for key in bin_keys}
    #     samples_per_group_after[group] = {key: 0 for key in bin_keys}
    #     print(f"Group: {group}")
    #     with torch.no_grad():
    #         for input, label, _ in valid_data_loader:
    #             input = input.to(device)
    #             logits_or = arch.concept_predictor(input)
    #             logits_or = logits_or[:, indices]
    #             labels_or = label[:, indices]
    #             dict1 = model.count_num_samples_per_bin(group, logits_or, labels_or, num_classes, temp_scaling=False)
    #             dict2 = model.count_num_samples_per_bin(group, logits_or, labels_or, num_classes, temp_scaling=True)
    #             samples_per_group_before[group] = {key: dict1[key] + samples_per_group_before[group][key] for key in dict1}
    #             samples_per_group_after[group] = {key: dict2[key] + samples_per_group_after[group][key] for key in dict2}
    #
    #             confidences_and_predictions_before[group] = model.collect_confidences_and_predictions(group, logits_or, temp_scaling=False)
    #             confidences_and_predictions_after[group] = model.collect_confidences_and_predictions(group, logits_or, temp_scaling=True)

    # make a group bar chart where the x axis has the confidences per group before and after, the the y axis has the number of samples
    # import matplotlib.pyplot as plt
    # import pandas as pd
    # import seaborn as sns
    #
    # # Before
    # data = []
    # for group, confidences_and_predictions in confidences_and_predictions_before.items():
    #     for confidence, predictions in confidences_and_predictions.items():
    #         for prediction, count in predictions.items():
    #             data.append({'Group': group, 'Confidence': confidence, 'Prediction': prediction, 'Count': count})
    #
    # df = pd.DataFrame(data)
    # sns.barplot(x='Confidence', y='Count', hue='Prediction', data=df)
    # plt.title('Before temperature scaling')
    # plt.show()

    print("Done")


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
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
