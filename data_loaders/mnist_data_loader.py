import pickle

import graphviz
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms


def get_mnist_dataLoader_original(data_dir='./datasets/parabola',
                        type='SGD', config=None,
                        batch_size=None):

    # Download training and test sets
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images
    ])

    train_dataset = datasets.MNIST(root='./datasets/MNIST/data', train=True,
                                   download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root='./datasets/MNIST/data', train=False,
                                  download=True,
                                  transform=transform)

    dict_of_lists = {6: [], 8: [], 9: []}
    for i, (_, label) in enumerate(train_dataset):
        if label in dict_of_lists.keys():
            dict_of_lists[label].append(
                train_dataset.data[i].reshape(1, 28, 28))

    for key in dict_of_lists.keys():
        dict_of_lists[key] = np.vstack(dict_of_lists[key]).reshape(-1, 1,
                                                                   28, 28)
        if key == 8:
            X = torch.cat((torch.tensor(dict_of_lists[6]),
                           torch.tensor(dict_of_lists[8])))
        elif key > 8:
            X = torch.cat((X, torch.tensor(dict_of_lists[key])))

    # import pickle files
    with open('./datasets/MNIST/mine_preprocessed/area_dict.pkl', 'rb') as f:
        area = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/length_dict.pkl', 'rb') as f:
        length = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/thickness_dict.pkl', 'rb') as f:
        thickness = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/slant_dict.pkl', 'rb') as f:
        slant = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/width_dict.pkl', 'rb') as f:
        width = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/height_dict.pkl', 'rb') as f:
        height = pickle.load(f)

    # load the targets test
    with open('./datasets/MNIST/mine_preprocessed/area_dict_test.pkl', 'rb') as f:
        area_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/length_dict_test.pkl', 'rb') as f:
        length_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/thickness_dict_test.pkl', 'rb') as f:
        thickness_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/slant_dict_test.pkl', 'rb') as f:
        slant_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/width_dict_test.pkl', 'rb') as f:
        width_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/height_dict_test.pkl', 'rb') as f:
        height_test = pickle.load(f)

    targets = []
    digits_size = 0
    labels = []
    # for i in range(4,10):
    for i in [6, 8, 9]:
        # targets += list(
        #     zip(thickness[i], width[i], slant[i], height[i]))
        targets += list(
            zip(thickness[i], width[i], length[i]))
        # targets += list(
        # zip(thickness[i], area[i], length[i],
        #                     width[i], height[i], slant[i]))
        if i == 6:
            k = 0
        elif i == 8:
            k = 1
        else:
            k = 2
        # labels.append([(i-4) for j in range(len(targets) - digits_size)])
        labels.append([k for j in range(len(targets) - digits_size)])
        digits_size += len(width[i])

    targets = np.array(targets)
    C_categorical = np.apply_along_axis(lambda col: pd.qcut(col, q=3, labels=False), axis=0, arr=targets)

    # Fit and transform the matrix C to one-hot encoded format
    encoder = OneHotEncoder(sparse_output=False)  # Set sparse_output to False to get a dense matrix
    C = encoder.fit_transform(C_categorical)

    y = np.array([item for sublist in labels for item in sublist])

    # Split the data
    X_train, X_val, C_train, C_val, y_train, y_val = train_test_split(X, C, y,
                                                                      test_size=0.5,
                                                                      random_state=42)
    X_val, X_test, C_val, C_test, y_val, y_test = train_test_split(X_val, C_val, y_val,
                                                                      test_size=0.5,
                                                                      random_state=42)
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    C_train = torch.tensor(C_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    C_val = torch.tensor(C_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    C_test = torch.tensor(C_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # plot a bar plot with the number of concepts equal to 1 per class
    # for i in range(3):
    #     print(f'Class {i}')
    #     class_digit = C_train[y_train == i]
    #     for j in range(12):
    #         print(f'Concept {j}: {torch.sum((class_digit[:, j] == 1).int())}')

    # Create DataLoader
    train_dataset = TensorDataset(X_train, C_train, y_train)
    val_dataset = TensorDataset(X_val, C_val, y_val)
    test_dataset = TensorDataset(X_test, C_test, y_test)

    if type == 'SGD':
        data_train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
        data_val_loader = DataLoader(dataset=val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)
        data_test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)
    else:
        NotImplementedError('ERROR: data type not supported!')

    return data_train_loader, data_val_loader, data_test_loader

def get_mnist_dataLoader_full(data_dir='.', type='SGD', config=None, batch_size=None):

    # Download training and test sets
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images
    ])

    train_dataset = datasets.MNIST(root='./datasets/MNIST/data', train=True,
                                   download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root='./datasets/MNIST/data', train=False,
                                  download=True,
                                  transform=transform)

    dict_of_lists = {i: [] for i in range(10)}
    for i, (_, label) in enumerate(train_dataset):
        dict_of_lists[label].append(
            train_dataset.data[i].reshape(1, 28, 28))

    for key in dict_of_lists.keys():
        dict_of_lists[key] = np.vstack(dict_of_lists[key]).reshape(-1, 1,
                                                                   28, 28)
        if key == 1:
            X = torch.cat((torch.tensor(dict_of_lists[0]),
                           torch.tensor(dict_of_lists[1])))
        elif key > 1:
            X = torch.cat((X, torch.tensor(dict_of_lists[key])))

    # import pickle files
    with open('./datasets/MNIST/mine_preprocessed/area_dict.pkl', 'rb') as f:
        area = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/length_dict.pkl', 'rb') as f:
        length = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/thickness_dict.pkl', 'rb') as f:
        thickness = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/slant_dict.pkl', 'rb') as f:
        slant = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/width_dict.pkl', 'rb') as f:
        width = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/height_dict.pkl', 'rb') as f:
        height = pickle.load(f)

    # load the targets test
    with open('./datasets/MNIST/mine_preprocessed/area_dict_test.pkl', 'rb') as f:
        area_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/length_dict_test.pkl', 'rb') as f:
        length_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/thickness_dict_test.pkl', 'rb') as f:
        thickness_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/slant_dict_test.pkl', 'rb') as f:
        slant_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/width_dict_test.pkl', 'rb') as f:
        width_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/height_dict_test.pkl', 'rb') as f:
        height_test = pickle.load(f)

    targets = []
    digits_size = 0
    labels = []
    for i in range(10):
        targets += list(
            # zip(thickness[i], width[i], length[i])
            zip(thickness[i], width[i], length[i], slant[i], area[i], height[i])
        )

        labels.append([i for j in range(len(targets) - digits_size)])
        digits_size += len(width[i])

    targets = np.array(targets)
    C_categorical = np.apply_along_axis(lambda col: pd.qcut(col, q=3, labels=False), axis=0, arr=targets)

    # Fit and transform the matrix C to one-hot encoded format
    encoder = OneHotEncoder(
        sparse_output=False)  # Set sparse_output to False to get a dense matrix
    C = encoder.fit_transform(C_categorical)

    y = np.array([item for sublist in labels for item in sublist])

    # Split the data
    x_train, x_val, c_train, c_val, y_train, y_val = train_test_split(X, C, y,
                                                                      test_size=0.5,
                                                                      random_state=42)
    x_val, x_test, c_val, c_test, y_val, y_test = train_test_split(x_val, c_val,
                                                                   y_val,
                                                                   test_size=0.5,
                                                                   random_state=42)
    # Convert to PyTorch tensors
    X_train = torch.tensor(x_train, dtype=torch.float32)
    C_train = torch.tensor(c_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_val = torch.tensor(x_val, dtype=torch.float32)
    C_val = torch.tensor(c_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    X_test = torch.tensor(x_test, dtype=torch.float32)
    C_test = torch.tensor(c_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # plot a bar plot with the number of concepts equal to 1 per class
    # for i in range(3):
    #     print(f'Class {i}')
    #     class_digit = C_train[y_train == i]
    #     for j in range(12):
    #         print(f'Concept {j}: {torch.sum((class_digit[:, j] == 1).int())}')

    # Create DataLoader
    train_dataset = TensorDataset(X_train, C_train, y_train)
    val_dataset = TensorDataset(X_val, C_val, y_val)
    test_dataset = TensorDataset(X_test, C_test, y_test)

    if type == 'SGD':
        data_train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
        data_val_loader = DataLoader(dataset=val_dataset,
                                     batch_size=batch_size,
                                     shuffle=False)
        data_test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)
    else:
        NotImplementedError('ERROR: data type not supported!')

    return data_train_loader, data_val_loader, data_test_loader


def get_mnist_cy_dataLoader(ratio=0.2,
                           data_dir='./datasets/parabola',
                           type='Full-GD',
                           batch_size=None):

    # import pickle files
    with open('./datasets/MNIST/mine_preprocessed/area_dict.pkl', 'rb') as f:
        area = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/length_dict.pkl', 'rb') as f:
        length = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/thickness_dict.pkl', 'rb') as f:
        thickness = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/slant_dict.pkl', 'rb') as f:
        slant = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/width_dict.pkl', 'rb') as f:
        width = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/height_dict.pkl', 'rb') as f:
        height = pickle.load(f)

    # load the targets test
    with open('./datasets/MNIST/mine_preprocessed/area_dict_test.pkl', 'rb') as f:
        area_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/length_dict_test.pkl', 'rb') as f:
        length_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/thickness_dict_test.pkl', 'rb') as f:
        thickness_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/slant_dict_test.pkl', 'rb') as f:
        slant_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/width_dict_test.pkl', 'rb') as f:
        width_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/height_dict_test.pkl', 'rb') as f:
        height_test = pickle.load(f)

    targets = []
    digits_size = 0
    labels = []
    # for i in range(4,10):
    for i in [6, 8, 9]:
        # targets += list(
        #     zip(thickness[i], width[i], slant[i], height[i]))
        targets += list(
            zip(thickness[i], width[i], length[i]))
        # targets += list(
        # zip(thickness[i], area[i], length[i],
        #                     width[i], height[i], slant[i]))
        if i == 6:
            k = 0
        elif i == 8:
            k = 1
        else:
            k = 2
        # labels.append([(i-4) for j in range(len(targets) - digits_size)])
        labels.append([k for j in range(len(targets) - digits_size)])
        digits_size += len(width[i])

    targets = np.array(targets)

    def assign_bins(data, bin_edges):
        return np.digitize(data, bins=bin_edges, right=True)

    # Convert bin numbers to one-hot encoded values
    def one_hot_encode(bin_numbers, num_bins):
        return np.eye(num_bins)[bin_numbers - 1]

    bins_data_all = []
    for i in range(targets.shape[1]):
        # Combine the two lists
        combined_data = list(targets[:, i])

        # Sort the combined data
        combined_sorted = np.sort(combined_data)

        # Determine the number of data points per bin
        num_bins = 4
        bin_size = len(combined_sorted) // num_bins

        # Calculate bin edges
        bin_edges = [combined_sorted[i * bin_size] for i in
                     range(1, num_bins)] + [
                        combined_sorted[-1]]
        bin_edges = [-np.inf] + bin_edges

        # Assign bins to the original data lists
        bins_data = assign_bins(targets[:, i], bin_edges)

        # do one-hot encoding in the bins
        bins_data = one_hot_encode(bins_data, num_bins)

        # flatten the matrix
        bins_data_all.append(bins_data)

    # stack in the second dimension
    C = np.stack(bins_data_all, axis=1).reshape(-1, 12)

    y = np.array([item for sublist in labels for item in sublist])

    # Create synthetic dataset
    np.random.seed(42)
    num_classes = 3

    # Creating continuous concept targets (e.g., 5 concepts)

    # Standardize the data
    # scaler_X = StandardScaler()
    # scaler_C = StandardScaler()
    #
    # X = scaler_X.fit_transform(X)
    # C = scaler_C.fit_transform(C)

    # Split the data
    C_train, C_val, y_train, y_val = train_test_split(
        C,y, test_size=0.5,random_state=42
    )

    # Convert to PyTorch tensors
    C_train = torch.tensor(C_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    C_val = torch.tensor(C_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(C_train, y_train)
    val_dataset = TensorDataset(C_val, y_val)

    if type == 'SGD':
        data_train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
    else:
        NotImplementedError('ERROR: data type not supported!')

    if batch_size is None:
        batch_size = C_val.shape[0]
    else:
        batch_size = batch_size

    data_test_loader = DataLoader(dataset=val_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)

    return data_train_loader, data_test_loader

def get_mnist_dataLoader_8_9(data_dir='.',
                        type='SGD', config=None,
                        batch_size=None):

    # Download training and test sets
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images
    ])

    train_dataset = datasets.MNIST(root='./datasets/MNIST/data', train=True,
                                   download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root='./datasets/MNIST/data', train=False,
                                  download=True,
                                  transform=transform)

    dict_of_lists = {8: [], 9: []}
    for i, (_, label) in enumerate(train_dataset):
        if label in dict_of_lists.keys():
            dict_of_lists[label].append(
                train_dataset.data[i].reshape(1, 28, 28))

    for key in dict_of_lists.keys():
        dict_of_lists[key] = np.vstack(dict_of_lists[key]).reshape(-1, 1,
                                                                   28, 28)
        if key == 9:
            X = torch.cat((torch.tensor(dict_of_lists[8]),
                           torch.tensor(dict_of_lists[9])))

    # import pickle files
    with open('./datasets/MNIST/mine_preprocessed/area_dict.pkl', 'rb') as f:
        area = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/length_dict.pkl', 'rb') as f:
        length = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/thickness_dict.pkl', 'rb') as f:
        thickness = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/slant_dict.pkl', 'rb') as f:
        slant = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/width_dict.pkl', 'rb') as f:
        width = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/height_dict.pkl', 'rb') as f:
        height = pickle.load(f)

    # load the targets test
    with open('./datasets/MNIST/mine_preprocessed/area_dict_test.pkl', 'rb') as f:
        area_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/length_dict_test.pkl', 'rb') as f:
        length_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/thickness_dict_test.pkl', 'rb') as f:
        thickness_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/slant_dict_test.pkl', 'rb') as f:
        slant_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/width_dict_test.pkl', 'rb') as f:
        width_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/height_dict_test.pkl', 'rb') as f:
        height_test = pickle.load(f)

    targets = []
    digits_size = 0
    labels = []
    for i in [8, 9]:
        targets += list(
            zip(thickness[i], width[i], length[i]))
        if i == 8:
            k = 0
        else:
            k = 1
        labels.append([k for j in range(len(targets) - digits_size)])
        digits_size += len(width[i])

    targets = np.array(targets)

    def assign_bins(data, bin_edges):
        return np.digitize(data, bins=bin_edges, right=True)

    # Convert bin numbers to one-hot encoded values
    def one_hot_encode(bin_numbers, num_bins):
        return np.eye(num_bins)[bin_numbers - 1]

    bins_data_all = []
    for i in range(targets.shape[1]):
        # Combine the two lists
        combined_data = list(targets[:, i])

        # Sort the combined data
        combined_sorted = np.sort(combined_data)

        # Determine the number of data points per bin
        num_bins = 4
        bin_size = len(combined_sorted) // num_bins

        # Calculate bin edges
        bin_edges = [combined_sorted[i * bin_size] for i in
                     range(1, num_bins)] + [
                        combined_sorted[-1]]
        bin_edges = [-np.inf] + bin_edges

        # Assign bins to the original data lists
        bins_data = assign_bins(targets[:, i], bin_edges)

        # do one-hot encoding in the bins
        bins_data = one_hot_encode(bins_data, num_bins)

        # flatten the matrix
        bins_data_all.append(bins_data)

    # stack in the second dimension
    C = np.stack(bins_data_all, axis=1).reshape(-1, 12)

    y = np.array([item for sublist in labels for item in sublist])

    # Create synthetic dataset
    np.random.seed(42)

    # Creating continuous concept targets (e.g., 5 concepts)

    # Standardize the data
    # scaler_X = StandardScaler()
    # scaler_C = StandardScaler()
    #
    # X = scaler_X.fit_transform(X)
    # C = scaler_C.fit_transform(C)

    # Split the data
    X_train, X_val, C_train, C_val, y_train, y_val = train_test_split(X, C, y,
                                                                      test_size=0.5,
                                                                      random_state=42)
    X_val, X_test, C_val, C_test, y_val, y_test = train_test_split(X_val, C_val, y_val,
                                                                      test_size=0.5,
                                                                      random_state=42)
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    C_train = torch.tensor(C_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    C_val = torch.tensor(C_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    C_test = torch.tensor(C_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # plot a bar plot with the number of concepts equal to 1 per class
    # for i in range(3):
    #     print(f'Class {i}')
    #     class_digit = C_train[y_train == i]
    #     for j in range(12):
    #         print(f'Concept {j}: {torch.sum((class_digit[:, j] == 1).int())}')

    # Create DataLoader
    train_dataset = TensorDataset(X_train, C_train, y_train)
    val_dataset = TensorDataset(X_val, C_val, y_val)
    test_dataset = TensorDataset(X_test, C_test, y_test)

    if type == 'SGD':
        data_train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
        data_val_loader = DataLoader(dataset=val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)
        data_test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)
    else:
        NotImplementedError('ERROR: data type not supported!')

    return data_train_loader, data_val_loader, data_test_loader

def collate_fn(batch):
    data, concepts, labels, indices = zip(*batch)
    data = torch.stack(data)
    concepts = torch.stack(concepts)
    labels = torch.stack(labels)
    indices = torch.tensor(indices)
    return data, concepts, labels, indices


def get_mnist_dataLoader_different_bins(data_dir='./datasets/parabola',
                        type='SGD', config=None,
                        batch_size=None):

    # Download training and test sets
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images
    ])

    train_dataset = datasets.MNIST(root='./datasets/MNIST/data', train=True,
                                   download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root='./datasets/MNIST/data', train=False,
                                  download=True,
                                  transform=transform)

    dict_of_lists = {7: [], 8: [], 9: []}
    for i, (_, label) in enumerate(train_dataset):
        if label in dict_of_lists.keys():
            dict_of_lists[label].append(
                train_dataset.data[i].reshape(1, 28, 28))

    for key in dict_of_lists.keys():
        dict_of_lists[key] = np.vstack(dict_of_lists[key]).reshape(-1, 1,
                                                                   28, 28)
        if key == 8:
            X = torch.cat((torch.tensor(dict_of_lists[7]),
                           torch.tensor(dict_of_lists[8])))
        elif key > 8:
            X = torch.cat((X, torch.tensor(dict_of_lists[key])))

    # import pickle files
    with open('./datasets/MNIST/mine_preprocessed/area_dict.pkl', 'rb') as f:
        area = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/length_dict.pkl', 'rb') as f:
        length = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/thickness_dict.pkl', 'rb') as f:
        thickness = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/slant_dict.pkl', 'rb') as f:
        slant = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/width_dict.pkl', 'rb') as f:
        width = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/height_dict.pkl', 'rb') as f:
        height = pickle.load(f)

    # load the targets test
    with open('./datasets/MNIST/mine_preprocessed/area_dict_test.pkl', 'rb') as f:
        area_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/length_dict_test.pkl', 'rb') as f:
        length_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/thickness_dict_test.pkl', 'rb') as f:
        thickness_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/slant_dict_test.pkl', 'rb') as f:
        slant_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/width_dict_test.pkl', 'rb') as f:
        width_test = pickle.load(f)
    with open('./datasets/MNIST/mine_preprocessed/height_dict_test.pkl', 'rb') as f:
        height_test = pickle.load(f)

    targets = []
    digits_size = 0
    labels = []
    # for i in range(4,10):
    for i in [7, 8, 9]:
        # targets += list(
        #     zip(thickness[i], width[i], slant[i], height[i]))
        targets += list(
            zip(thickness[i], width[i], length[i]))
        # targets += list(
        # zip(thickness[i], area[i], length[i],
        #                     width[i], height[i], slant[i]))
        if i == 7:
            k = 0
        elif i == 8:
            k = 1
        else:
            k = 2
        # labels.append([(i-4) for j in range(len(targets) - digits_size)])
        labels.append([k for j in range(len(targets) - digits_size)])
        digits_size += len(width[i])

    targets = np.array(targets)

    def assign_bins(data, bin_edges):
        return np.digitize(data, bins=bin_edges, right=True)

    # Convert bin numbers to one-hot encoded values
    def one_hot_encode(bin_numbers, num_bins):
        return np.eye(num_bins)[bin_numbers - 1]

    def process_data_not_equal_bins(targets, num_bins=4):
        bins_data_all_indices = {}
        bins_data_all = []
        min_max_values_all = []
        closest_images_all = []
        bin_counts = []

        for i in range(targets.shape[1]):
            # Calculate bin edges for equally distant features
            min_val, max_val = np.min(targets[:, i]), np.max(targets[:, i])
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)

            # Assign bins to the original data lists
            bins_data = assign_bins(targets[:, i], bin_edges)

            # Do one-hot encoding in the bins
            bins_data_encoded = one_hot_encode(bins_data, num_bins)

            # Get min and max values per bin
            min_max_values = []
            closest_images = []
            counts = []

            feature_bins_data = {}

            for bin_num in range(1, num_bins + 1):
                bin_indices = np.where(bins_data == bin_num)[0]
                bin_values = targets[bin_indices, i]
                counts.append(len(bin_indices))

                if len(bin_values) > 0:
                    min_val = np.min(bin_values)
                    max_val = np.max(bin_values)
                    min_max_values.append((min_val, max_val))

                    # Select 5 images closest to the minimum and 5 closest to the maximum
                    closest_min_indices = bin_indices[
                        np.argsort(np.abs(bin_values - min_val))[:5]]
                    closest_max_indices = bin_indices[
                        np.argsort(np.abs(bin_values - max_val))[:5]]
                    closest_images.append(
                        (closest_min_indices, closest_max_indices))
                else:
                    min_max_values.append((None, None))
                    closest_images.append(([], []))

                feature_bins_data[bin_num] = list(bin_indices)

            bins_data_all.append(bins_data_encoded)
            bins_data_all_indices[i] = feature_bins_data
            min_max_values_all.append(min_max_values)
            closest_images_all.append(closest_images)
            bin_counts.append(counts)

        return bins_data_all, bins_data_all_indices, min_max_values_all, closest_images_all, bin_counts

    bins_data_all, _, _, _, _ = process_data_not_equal_bins(targets, num_bins=3)
    # stack in the second dimension
    C = np.stack(bins_data_all, axis=1).reshape(-1, 9)
    y = np.array([item for sublist in labels for item in sublist])

    # Create synthetic dataset
    np.random.seed(42)
    num_classes = 3

    # Creating continuous concept targets (e.g., 5 concepts)

    # Standardize the data
    # scaler_X = StandardScaler()
    # scaler_C = StandardScaler()
    #
    # X = scaler_X.fit_transform(X)
    # C = scaler_C.fit_transform(C)

    # Split the data
    X_train, X_val, C_train, C_val, y_train, y_val = train_test_split(X, C, y,
                                                                      test_size=0.5,
                                                                      random_state=42)
    X_val, X_test, C_val, C_test, y_val, y_test = train_test_split(X_val, C_val, y_val,
                                                                      test_size=0.5,
                                                                      random_state=42)
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    C_train = torch.tensor(C_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    C_val = torch.tensor(C_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    C_test = torch.tensor(C_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # plot a bar plot with the number of concepts equal to 1 per class
    # for i in range(3):
    #     print(f'Class {i}')
    #     class_digit = C_train[y_train == i]
    #     for j in range(12):
    #         print(f'Concept {j}: {torch.sum((class_digit[:, j] == 1).int())}')

    # Create DataLoader
    train_dataset = TensorDataset(X_train, C_train, y_train)
    val_dataset = TensorDataset(X_val, C_val, y_val)
    test_dataset = TensorDataset(X_test, C_test, y_test)

    if type == 'SGD':
        data_train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
        data_val_loader = DataLoader(dataset=val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)
        data_test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)
    else:
        NotImplementedError('ERROR: data type not supported!')

    return data_train_loader, data_val_loader, data_test_loader