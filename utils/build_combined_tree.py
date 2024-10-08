import copy
import numpy as np
import graphviz
from networks.custom_dt_gini_with_entropy_metrics import CustomDecisionTree
import pandas as pd
from torchvision import datasets, transforms
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import torch
import pickle
from utils.export_combined_tree import export_combined_tree
from utils.export_decision_paths_with_subtrees import \
    export_decision_paths_with_subtrees
from utils.tree_utils import prune_tree_chaid, \
    get_light_colors

from networks.custom_chaid_tree import CHAIDTree

def build_combined_tree(original_tree, leaf_trees):
    """
    Construct a new tree where the leaves of the CHAID tree are replaced with custom decision trees.
    Also aggregates total_info_gain and info_gain_per_leaf for each subtree.
    """
    # Find the maximum node ID in the CHAID tree to start numbering from there
    max_chaid_id = max(node.id for node in traverse_chaid_nodes(original_tree.root))
    node_count = original_tree.node_count

    # Dictionaries to store the aggregated info
    total_info_gain = 0
    aggregated_info_gain_per_leaf = {}

    # Helper function to recursively copy the CHAID tree and replace leaves
    def copy_and_replace(node):
        nonlocal max_chaid_id
        nonlocal node_count
        nonlocal total_info_gain
        nonlocal aggregated_info_gain_per_leaf

        # If the node is a leaf and has a corresponding subtree in leaf_trees
        if node.is_leaf and node.id in leaf_trees:
            # Get the custom tree associated with this leaf
            subtree_root = leaf_trees[node.id].tree
            subtree_root.id = node.id  # Assign the CHAID leaf node ID to the custom tree root

            # Reassign IDs to the children of the custom subtree root
            def reassign_ids(subtree_node):
                nonlocal max_chaid_id
                nonlocal node_count
                if subtree_node.left:
                    max_chaid_id += 1
                    node_count += 1
                    subtree_node.left.id = max_chaid_id
                    reassign_ids(subtree_node.left)
                if subtree_node.right:
                    max_chaid_id += 1
                    node_count += 1
                    subtree_node.right.id = max_chaid_id
                    reassign_ids(subtree_node.right)

            reassign_ids(subtree_root)

            # Aggregate information gain from this subtree
            subtree_info_gain, subtree_info_per_leaf = leaf_trees[node.id].traverse_and_sum_info_gain(subtree_root)
            total_info_gain += subtree_info_gain
            aggregated_info_gain_per_leaf.update(subtree_info_per_leaf)

            return subtree_root  # Return the root of the custom tree as replacement
        else:
            # Recursively process child nodes
            for category, child in node.children.items():
                replaced_child = copy_and_replace(child)
                if replaced_child != child:
                    node.children[category] = replaced_child
            return node

    # Start the replacement process from the root of the original tree
    combined_tree_root = copy_and_replace(original_tree.root)

    # Update the original tree with the new root and structure
    combined_tree_obj = CHAIDTree(
        n_classes=original_tree.n_classes,
        concept_names=original_tree.concept_names,
        max_depth=original_tree.max_depth
    )
    combined_tree_obj.root = combined_tree_root
    combined_tree_obj.node_count = node_count

    # Print aggregated info gain and info per leaf
    print(f"\nTotal Information Gain: {total_info_gain}")
    #print(f"Information Gain per Leaf Node: {aggregated_info_gain_per_leaf}")

    return combined_tree_obj, total_info_gain, aggregated_info_gain_per_leaf

def traverse_chaid_nodes(node):
    """
    Generator to traverse nodes in a CHAID tree.
    """
    yield node
    for child in node.children.values():
        yield from traverse_chaid_nodes(child)


# Example usage
if __name__ == "__main__":
    # from sklearn.datasets import load_iris
    # iris = load_iris()
    # X, y = iris.data, iris.target

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
    y = np.array([item for sublist in labels for item in sublist])

    # Convert continuous features to categorical bins for CHAID
    X = np.apply_along_axis(lambda col: pd.qcut(col, q=3, labels=False), axis=0, arr=targets)

    # Define feature names and class names
    feature_names = ["thickness", "width", "length"]
    feature_names_flattened = ["thickness_small", "thickness_medium", "thickness_large",
                               "width_small", "width_medium", "width_large",
                               "length_small", "length_medium", "length_large"]
    feature_names_dict = {
            "thickness": ["small", "medium", "large"],
            "width": ["small", "medium", "large"],
            "length": ["small", "medium", "large"]
        },
    feature_value_names = {0: {0: "small", 1: "medium", 2: "large"},
                           1: {0: "small", 1: "medium", 2: "large"},
                           2: {0: "small", 1: "medium", 2: "large"}}
    class_names = ["6", "8", "9"]
    class_colors = ['#DDDDDD', '#F7F7F7', '#FFFFCC']

    # Fit the CHAID tree
    chaid_tree = CHAIDTree(n_classes=len(np.unique(y)),
                           concept_names=feature_names_dict,
                           max_depth=1000, min_child_size=3000, alpha=0.05)
    chaid_tree.fit(X, y)
    # Prune the tree
    chaid_tree = prune_tree_chaid(chaid_tree)

    binary_tree = DecisionTreeClassifier(max_depth=1000)
    binary_tree.fit(X, y)
    print(binary_tree.score(X, y))
    # print binary tree
    dot = tree.export_graphviz(binary_tree,
                               out_file=None,
                               feature_names=feature_names,
                               class_names=class_names,
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot)
    graph.render("binary_tree", format='pdf', cleanup=True)

    # Predict classes
    predictions = chaid_tree.predict(X)
    acc = np.mean(predictions == y)
    print(f"Accuracy: {acc}")

    # Apply method
    ids = chaid_tree.apply(X)

    # Export the tree to Graphviz format
    dot_data = chaid_tree.export_tree(feature_names, class_names,
                                      class_colors, feature_value_names)
    graph = graphviz.Source(dot_data)
    graph.render("chaid_tree", format='pdf', cleanup=True)

    # Decision path
    leaf_indices = chaid_tree.apply(X)

    for leaf in np.unique(leaf_indices):
        sample_indices = np.where(leaf_indices == leaf)[0]
        decision_paths = chaid_tree.decision_path(X[sample_indices])
        chaid_tree.export_decision_paths(
            decision_paths,
            feature_names=feature_names,
            class_colors=class_colors,
            class_names=class_names,
            feature_value_names=feature_value_names,
            output_dir='./decision_paths_chaid',
            leaf_id=leaf
        )
    chaid_tree_or = copy.deepcopy(chaid_tree)

    def get_leaf_nodes(node):
        """
        Get all leaf nodes from the CHAID tree.

        Parameters:
        - node: The root node of the CHAID tree.

        Returns:
        - A list of leaf nodes.
        """
        if node.is_leaf:
            return [node]

        leaves = []
        for child in node.children.values():
            leaves.extend(get_leaf_nodes(child))
        return leaves

    def get_samples_for_leaf(leaf, X, tree):
        """
        Return the indices of samples that end up in the given leaf node.

        Parameters:
        - leaf: The target leaf node for which to find samples.
        - X: The input dataset (numpy array).
        - tree: The root node of the CHAID tree.

        Returns:
        - A list of sample indices that end up in the given leaf node.
        """
        path = []

        def traverse(node, sample_indices):
            if node.is_leaf:
                if node.id == leaf.id:
                    path.extend(sample_indices)
                return

            # Traverse through each category/child node
            for category, child_node in node.children.items():
                child_indices = [i for i in sample_indices if
                                 X[i, node.feature] == category]
                traverse(child_node, child_indices)

        # Start traversal from the root
        traverse(tree, list(range(X.shape[0])))
        return path

    leaf_trees = {}
    leaf_nodes = get_leaf_nodes(chaid_tree.root)
    for leaf in leaf_nodes:
        indices = get_samples_for_leaf(leaf, X, chaid_tree.root)
        X_leaf, y_leaf = X[indices], y[indices]
        if len(np.unique(y_leaf)) > 1:
            leaf_tree = CustomDecisionTree(min_samples_leaf=1,
                                           n_classes=len(np.unique(y_leaf)))
            leaf_tree.fit(X_leaf, y_leaf)
            leaf_trees[leaf.id] = leaf_tree

            # Export each individual leaf tree to Graphviz format
            leaf_dot_data = leaf_tree.export_tree(
                feature_names=feature_names,
                class_names=class_names,
                class_colors=get_light_colors(len(class_names))
            )
            leaf_graph = graphviz.Source(leaf_dot_data)
            leaf_graph.render(f"dt_with_dts_custom_gini_entropy/leaf_tree_{leaf.id}", format='pdf',
                              cleanup=True)

    # Build combined tree
    combined_tree = build_combined_tree(chaid_tree, leaf_trees)
    combined_tree_dot_data = export_combined_tree(combined_tree.root,
                                                  feature_names_main=feature_names,
                                                  feature_names_subtree=feature_names_flattened,
                                                  class_names=class_names,
                                                  class_colors=class_colors,
                                                  feature_value_names=feature_value_names
                                                  )
    graph = graphviz.Source(combined_tree_dot_data)
    graph.render(filename="combined_tree", format='pdf', cleanup=True)

    for leaf in np.unique(leaf_indices):
        sample_indices = np.where(leaf_indices == leaf)[0]
        decision_paths = chaid_tree_or.decision_path(X[sample_indices])
        export_decision_paths_with_subtrees(
            combined_tree,
            decision_paths,
            feature_names_main=feature_names,
            feature_names_subtree=feature_names_flattened,
            class_colors=class_colors,
            class_names=class_names,
            feature_value_names=feature_value_names,
            output_dir='./decision_paths_chaid_w_subtrees',
            leaf_id=leaf)