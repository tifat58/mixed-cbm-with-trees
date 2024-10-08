import json
import logging.config
import pickle

import numpy as np
import torch
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

from sklearn.metrics import roc_auc_score, average_precision_score


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def get_correct(y_hat, y, num_classes):
    if y_hat.shape[1] == 1:
        y_hat = torch.sigmoid(y_hat)
        y_hat = [1 if y_hat[i] >= 0.5 else 0 for i in range(len(y_hat))]
        correct = [1 if y_hat[i] == y[i] else 0 for i in range(len(y_hat))]
        return np.sum(correct)
    else:
        return y_hat.argmax(dim=1).eq(y).sum().item()

def correct_predictions_per_class(logits, true_labels, num_classes,
                                  threshold=0.5):
    """
    Calculate the number of correct predictions per class.

    Parameters:
    logits (torch.Tensor): A tensor of predicted logits (shape: [batch_size, num_classes] or [batch_size]).
    true_labels (torch.Tensor): A tensor of true labels (shape: [batch_size]).
    num_classes (int): The number of classes.
    threshold (float): Threshold to convert logits to binary predictions for single class case.

    Returns:
    list: A list containing the number of correct predictions for each class.
    """
    if logits.shape[1] == 1:
        # Binary classification case
        probabilities = torch.sigmoid(logits)
        predicted_classes = torch.where(probabilities >= threshold, 1, 0)
    else:
        # Multi-class classification case
        predicted_classes = torch.argmax(logits, dim=1)

    # Initialize a list to store the number of correct predictions per class
    correct_counts = [0] * num_classes

    # Iterate over each class and count correct predictions
    for class_idx in range(num_classes):
        correct_counts[class_idx] = torch.sum(
            (predicted_classes == class_idx) & (
                        true_labels == class_idx)).item()

    return correct_counts

def column_get_correct(logits, labels, my_dict, threshold=0.5):
    """
    Calculate accuracy per column for predicted logits using categorical conversion.

    Parameters:
    logits (torch.Tensor): A tensor of predicted logits (shape: [batch_size, num_labels]).
    labels (torch.Tensor): A tensor of true labels (shape: [batch_size, num_labels]).
    my_dict (dict): A dictionary where keys represent group names and values are lists
                    of category names.
    threshold (float): Threshold to convert logits to binary predictions.

    Returns:
    torch.Tensor: A tensor containing the number of correct predictions for each column.
    """

    # Convert logits and labels to numpy for conversion
    logits_np = logits.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Convert logits and labels to categorical values
    logits_categorical = convert_to_categorical(logits_np, my_dict)
    labels_categorical = onehot_to_categorical(labels_np, my_dict)

    # Convert back to torch tensors
    logits_categorical = torch.tensor(logits_categorical, dtype=torch.int64)
    labels_categorical = torch.tensor(labels_categorical, dtype=torch.int64)

    # Calculate accuracy per column
    correct_predictions = (logits_categorical == labels_categorical).float()
    correct_predictions = correct_predictions.sum(dim=0)

    return correct_predictions

def count_labels_per_class(y):
    """
    Count the number of ground truth labels per class.

    Parameters:
    y (torch.Tensor): A tensor of ground truth class labels (shape: [batch_size]).

    Returns:
    dict: A dictionary with the class labels as keys and the number of ground truth labels as values.
    """
    # Get unique class labels and their counts
    unique_labels, counts = torch.unique(y, return_counts=True)

    # Create a dictionary with class labels as keys and counts as values
    label_counts = {int(label.item()): int(count.item()) for label, count in
                    zip(unique_labels, counts)}

    return label_counts

def compute_AUC(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs, AUPRCs of all classes.
    """
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    try:
        AUROCs = roc_auc_score(gt_np, pred_np)
        AUPRCs = average_precision_score(gt_np, pred_np)
    except:
        AUROCs = 0.5
        AUPRCs = 0.5

    return AUROCs, AUPRCs

def setup_logging(save_dir, log_config='loggers/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)

def compute_class_weights(
    config,
    train_dl,
    n_classes,
):

    task_class_weights = None

    if config.get('use_task_class_weights', False):
        logging.info(
            f"Computing task class weights in the training dataset with "
            f"size {len(train_dl)}..."
        )
        attribute_count = np.zeros((max(n_classes, 2),))
        samples_seen = 0
        for i, data in enumerate(train_dl):
            if len(data) == 2:
                (_, (y, _)) = data
            else:
                (_, y, _) = data
            if n_classes > 1:
                y = torch.nn.functional.one_hot(
                    y,
                    num_classes=n_classes,
                ).cpu().detach().numpy()
            else:
                y = torch.cat(
                    [torch.unsqueeze(1 - y, dim=-1), torch.unsqueeze(y, dim=-1)],
                    dim=-1,
                ).cpu().detach().numpy()
            attribute_count += np.sum(y, axis=0)
            samples_seen += y.shape[0]
        print("Class distribution is:", attribute_count / samples_seen)
        if n_classes > 1:
            task_class_weights = samples_seen / attribute_count - 1
        else:
            task_class_weights = np.array(
                [attribute_count[0]/attribute_count[1]]
            )
    return task_class_weights


def create_group_indices(concept_groups, squeeze_double=True):
    """
    Create a mapping from concept group names to indices.

    Parameters:
    - concept_groups (dict): A dictionary of concept groups.

    Returns:
    - group_indices (dict): A dictionary mapping group names to their concept indices.
    """
    group_indices = {}
    current_index = 0
    current_group_index = 0

    for group_name, concepts in concept_groups.items():
        num_concepts = len(concepts)
        if num_concepts == 2 and squeeze_double:
            group_indices[current_group_index] = [current_index]
            current_index += 1
        elif num_concepts == 2 and squeeze_double == False:
            group_indices[current_group_index] = [current_index, current_index]
            current_index += 1
        else:
            group_indices[current_group_index] = list(range(current_index, current_index + num_concepts))
            current_index += num_concepts
        current_group_index += 1

    return group_indices

def convert_to_index_name_mapping(my_dict):
    feature_value_names = {}

    # Iterate over the dictionary with an index
    for idx, (key, values) in enumerate(my_dict.items()):
        # Create a nested dictionary where the inner values are indexed
        feature_value_names[idx] = {i: value for i, value in enumerate(values)}

    return feature_value_names


def flatten_dict_to_list(my_dict):
    flattened_list = []

    # Iterate over the dictionary
    for key, values in my_dict.items():
        # Create the flattened elements with the key as prefix
        for value in values:
            if len(values) == 2:
                flattened_list.append(f"{key}")
            else:
                flattened_list.append(f"{key}::{value}")

    return flattened_list

def convert_to_categorical(matrix, my_dict):
    """
    Reverses a probabilistic-encoded matrix back to its original categorical values,
    where each group of columns may have a different number of probabilistic columns.

    Parameters:
    - matrix (numpy.ndarray): The probabilistic-encoded matrix.
    - my_dict (dict): A dictionary where keys represent group names and values are lists
                      of category names.

    Returns:
    - numpy.ndarray: The matrix of original categorical values.
    """

    # Calculate the number of original columns
    n_original_cols = len(my_dict)

    # Initialize the reversed matrix
    reversed_matrix = np.zeros((matrix.shape[0], n_original_cols), dtype=int)

    # Initialize the start index for each group's columns in the encoded matrix
    start_index = 0

    # Iterate over each group in the dictionary
    for col_idx, (group, categories) in enumerate(my_dict.items()):
        # Number of columns for this group
        q = len(categories)

        if q == 2:
            offset = 1
            # If there's only one element, apply sigmoid to the whole group at once
            sigmoid_values = 1 / (1 + np.exp(-matrix[:, start_index]))
            reversed_matrix[:, col_idx] = (sigmoid_values >= 0.5).astype(int)
        elif q == 1:
            # Raise an error if there are exactly two categories
            raise ValueError(f"Group '{group}' has 1 category, which is not supported.")
        else:
            offset = q
            # Otherwise, take the argmax across the group for each row
            reversed_matrix[:, col_idx] = np.argmax(matrix[:, start_index:start_index + q], axis=1)

        # Update the start index for the next group's columns
        start_index += offset

    return reversed_matrix

def onehot_to_categorical(matrix, my_dict):
    """
    Converts one-hot encoded matrix back to its original categorical values.
    When q > 3, it takes the argmax to get the categorical value.
    When q = 1, it leaves the input as it is (returns the binary value as is).

    Parameters:
    - matrix (numpy.ndarray): The one-hot encoded matrix.
    - my_dict (dict): A dictionary where keys represent group names and values are lists
                      of category names.

    Returns:
    - numpy.ndarray: The matrix of original categorical values.
    """

    # Calculate the number of original columns
    n_original_cols = len(my_dict)

    # Initialize the reversed matrix
    reversed_matrix = np.zeros((matrix.shape[0], n_original_cols), dtype=int)

    # Initialize the start index for each group's columns in the encoded matrix
    start_index = 0

    # Iterate over each group in the dictionary
    for col_idx, (group, categories) in enumerate(my_dict.items()):
        # Number of columns for this group
        q = len(categories)

        if q == 2:
            offset = 1
            # If there's only one category, simply take the value as is (0 or 1)
            reversed_matrix[:, col_idx] = matrix[:, start_index].astype(int)
        elif q == 1:
            # Raise an error if there are exactly two categories
            raise ValueError(f"Group '{group}' has 1 category, which is not supported.")
        else:
            offset = q
            # Otherwise, take the argmax across the group for each row if q > 3
            reversed_matrix[:, col_idx] = np.argmax(matrix[:, start_index:start_index + q], axis=1)

        # Update the start index for the next group's columns
        start_index += offset

    return reversed_matrix

def sigmoid_or_softmax_with_groups(logits, my_dict):
    """
    Applies softmax or sigmoid to each group of logits based on the number of categories specified in my_dict.

    Parameters:
    - logits (torch.Tensor): The input tensor with logits. Shape: (batch_size, total_categories).
    - my_dict (dict): A dictionary defining groups and the number of categories for each group.

    Returns:
    - torch.Tensor: The tensor with softmax or sigmoid applied to each group of logits.
    """
    # Initialize the list to store results for each group
    result = []

    # Keep track of the current index in the logits tensor
    current_index = 0

    # Apply appropriate activation for each group in my_dict
    for group, categories in my_dict.items():
        num_categories = len(categories)

        # Apply activation based on the number of categories
        if num_categories == 2:
            # Slice the logits tensor for this group
            offset = 1
            group_logits = logits[:, current_index:current_index + 1]
            # Apply sigmoid for a single category
            group_result = torch.sigmoid(group_logits)
        elif num_categories == 1:
            # Raise an error if there are exactly two categories
            raise ValueError(f"Group '{group}' has 1 category, which is not supported.")
        else:
            offset = num_categories
            # Slice the logits tensor for this group
            group_logits = logits[:, current_index:current_index + num_categories]
            # Apply softmax for more than two categories
            group_result = torch.softmax(group_logits, dim=1)

        result.append(group_result)

        # Update the index for the next group
        current_index += offset

    # Concatenate all results to get the final matrix
    result = torch.cat(result, dim=1)

    return result

def probs_to_binary(matrix, my_dict):
    """
    Converts a matrix of probabilities to binary format.
    - If q = 1, apply a threshold of 0.5 to the probabilities (assumed already sigmoid).
    - If q >= 3, use argmax to set "1" at the max position and "0" at the others.

    Parameters:
    - matrix (numpy.ndarray): The probabilistic matrix.
    - my_dict (dict): A dictionary where keys represent group names and values are lists
                      of category names.

    Returns:
    - numpy.ndarray: A binary matrix (same shape as input matrix) where values are 0 or 1.
    """

    # Initialize the binary matrix with the same shape as the input matrix
    binary_matrix = np.zeros_like(matrix, dtype=int)

    # Initialize the start index for each group's columns in the encoded matrix
    start_index = 0

    # Iterate over each group in the dictionary
    for col_idx, (group, categories) in enumerate(my_dict.items()):
        # Number of columns for this group
        q = len(categories)

        if q == 2:
            # If q == 2, apply threshold at 0.5 directly (values are assumed to already be sigmoid)
            offset = 1
            binary_matrix[:, start_index] = (matrix[:, start_index] >= 0.5).astype(int)
        elif q >= 3:
            offset = q
            # If q >= 3, use argmax to set "1" at the max position and "0" at others
            argmax_indices = np.argmax(matrix[:, start_index:start_index + q], axis=1)
            # Set "1" at the position of the argmax for each row
            binary_matrix[np.arange(matrix.shape[0]), start_index + argmax_indices] = 1
        elif q == 1:
            # Raise an error if there are exactly two categories
            raise ValueError(f"Group '{group}' has 1 category, which is not supported.")

        # Update the start index for the next group's columns
        start_index += offset

    return binary_matrix

def identify_misclassified_samples(logits, labels, my_dict, threshold=0.5):
    """
    Identify samples where any concept is misclassified.

    Parameters:
    logits (torch.Tensor): A tensor of predicted logits (shape: [batch_size, num_labels]).
    labels (torch.Tensor): A tensor of true labels (shape: [batch_size, num_labels]).
    my_dict (dict): A dictionary where keys represent group names and values are lists
                    of category names.
    threshold (float): Threshold to convert logits to binary predictions.

    Returns:
    torch.Tensor: A boolean tensor indicating `True` for samples with any misclassified concept, `False` otherwise.
    """

    # Convert logits and labels to numpy for conversion
    logits_np = logits.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Convert logits and labels to categorical values
    logits_categorical = convert_to_categorical(logits_np, my_dict)
    labels_categorical = onehot_to_categorical(labels_np, my_dict)

    # Convert back to torch tensors
    logits_categorical = torch.tensor(logits_categorical, dtype=torch.int64)
    labels_categorical = torch.tensor(labels_categorical, dtype=torch.int64)

    # Calculate a boolean mask for misclassification per sample
    misclassified = (logits_categorical != labels_categorical).any(dim=1)

    return misclassified

def round_to_nearest_steps(array):
    # Shift values down by 0.05
    shifted_array = array - 0.05
    # Scale by 10
    scaled_array = shifted_array * 10
    # Round to the nearest integer
    rounded_array = np.round(scaled_array)
    # Rescale back and shift back up by 0.05, then round to two decimal places
    result = np.round(rounded_array / 10 + 0.05, 2)
    return result

def precision_round(value, tol=1e-5):
    """Round values that have floating-point precision issues around .9999 or .00000."""
    if isinstance(value, np.floating):
        value = value.item()  # Convert NumPy float to Python float

    decimal_part = value - np.floor(value)  # Get the decimal part
    rounded_value = round(value, 6)  # Round to a set precision (6 decimal places)

    # Handle cases like 0.6999999 or 0.80000004
    if abs(value - rounded_value) < tol:  # If the value is close to its rounded version
        return rounded_value
    return value  # Otherwise, return the original value


def update_pickle_dict(pickle_file, exper_name, run_id, new_key, new_value):
    """
    Opens a pickled dictionary, adds a new key-value pair, and re-saves the dictionary.
    If the pickle file does not exist, it creates a new one.

    Args:
    - pickle_file (str): The path to the pickle file.
    - new_key: The key to be added to the dictionary.
    - new_value: The value associated with the new key.
    :param run_id:
    """
    try:
        # Try to open the pickle file and load the dictionary
        with open(pickle_file, 'rb') as file:
            my_dict = pickle.load(file)
    except FileNotFoundError:
        # If the file does not exist, create a new dictionary
        print(f"{pickle_file} not found. Creating a new dictionary.")
        my_dict = {}

    # Add the new key-value pair to the dictionary
    if exper_name not in my_dict:
        my_dict[exper_name] = {}

    if run_id not in my_dict[exper_name]:
        my_dict[exper_name][run_id] = {}

    my_dict[exper_name][run_id][new_key] = new_value

    # Save the updated or newly created dictionary back into the pickle file
    with open(pickle_file, 'wb') as file:
        pickle.dump(my_dict, file)
    print(f"Updated dictionary saved to {pickle_file}")