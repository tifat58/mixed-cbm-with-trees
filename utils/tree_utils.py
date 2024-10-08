import re
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.tree import DecisionTreeClassifier,  _tree


def extract_features_from_splits(tree):
    feature_indices = set()

    def recurse(node):
        if tree.feature[node] != -2:  # -2 indicates a leaf node
            feature_indices.add(tree.feature[node])
            recurse(tree.children_left[node])
            recurse(tree.children_right[node])

    recurse(0)
    return sorted(list(feature_indices))


def get_light_colors(num_colors):
    # Use a light palette from seaborn
    palette = sns.color_palette("pastel", num_colors)
    # Convert to hex colors
    light_colors = [mcolors.rgb2hex(color) for color in palette]
    return light_colors


def replace_splits(dot_data, old_split="&le; 0.5", new_split="== 0"):
    lines = dot_data.split('\n')
    new_lines = []
    for line in lines:
        new_line = line.replace(old_split, new_split)
        new_lines.append(new_line)
    return '\n'.join(new_lines)


def modify_dot_with_colors(dot_data, color_map, clf, node_color="#DDDDDD"):
    lines = dot_data.split('\n')
    new_lines = []
    for line in lines:
        match = re.match(r'(\d+) \[label=.*\]', line)
        if match:
            node_id = int(match.group(1))
            if (clf.children_left[node_id] != _tree.TREE_LEAF or
                clf.children_right[node_id] != _tree.TREE_LEAF):
                # Node is not a leaf
                color = node_color
            else:
                # Node is a leaf
                node_class = clf.value[node_id].argmax()
                color = color_map[node_class]
            # Add fillcolor and style to the node definition
            line = re.sub(r'(?<=>)\]',
                          f', style="filled,rounded", fillcolor="{color}"]',
                          line)
        new_lines.append(line)
    return '\n'.join(new_lines)

def get_leaf_samples_and_features(tree, X):
    """Group samples by their leaf nodes and extract feature indices used in decision paths."""
    leaf_samples_indices = {}
    leaf_features_per_path = {}
    leaf_indices = tree.apply(X)

    for leaf in np.unique(leaf_indices):
        sample_indices = np.where(leaf_indices == leaf)[0]
        leaf_samples_indices[leaf] = sample_indices
        decision_path = tree.decision_path(X[sample_indices])
        leaf_features_per_path[leaf] = decision_path.feature_indices[decision_path.feature_indptr[0]: decision_path.feature_indptr[1]]

    return leaf_samples_indices, leaf_features_per_path


def get_decision_path_features_and_thresholds(tree, X):
    """
    Returns a simplified dictionary with the features used per decision path and their corresponding thresholds
    for each leaf node. Assigns 0 or 1 based on whether the feature satisfies the inequality for the given input.

    Parameters:
    - tree: An instance of the CustomDecisionTree
    - X: The input data samples

    Returns:
    - decision_path_dict: Simplified dictionary with leaf node indices as keys and dictionaries of features and their
                          corresponding threshold satisfaction (0 or 1) as values.
    """
    decision_path_dict = {}  # Initialize the dictionary to store the paths

    # Get the decision paths using the custom decision_path function
    decision_paths = tree.decision_path(X)

    for i, inputs in enumerate(X):
        current_path_dict = {}
        current_path_nodes = decision_paths.node_indices[
                             decision_paths.node_indptr[i]:
                             decision_paths.node_indptr[i + 1]]
        current_path_features = decision_paths.feature_indices[
                                decision_paths.feature_indptr[i]:
                                decision_paths.feature_indptr[i + 1]]

        node = tree.tree  # Start from the root node
        for j, current_node_id in enumerate(current_path_nodes[:-1]):  # Exclude the leaf node
            feature_index = current_path_features[j]
            threshold = node.threshold  # Get the threshold of the current node
            feature_value = inputs[feature_index]

            # Assign 0 if the feature value satisfies the inequality, 1 otherwise
            current_path_dict[feature_index] = 0 if feature_value < threshold else 1

            # Move to the next node in the path
            if feature_value < threshold:
                node = node.left
            else:
                node = node.right

        # Get the leaf node
        leaf_node = current_path_nodes[-1]  # The last node in the path is the leaf

        # If this leaf has not been added to the dictionary yet, add it
        if leaf_node not in decision_path_dict:
            decision_path_dict[leaf_node] = current_path_dict

    return decision_path_dict

def get_features_used_in_path(tree, X):
    """Get the feature indices used in the decision path to each leaf node."""
    decision_paths = tree.decision_path(X)
    feature_indices = []


    for sample_id in range(X.shape[0]):
        path = decision_paths.indices[
               decision_paths.indptr[sample_id]: decision_paths.indptr[
                   sample_id + 1]
               ]
        features_in_path = np.unique([tree.tree.feature_index if tree.tree.feature_index != -2 else -1 for node_id in path]
        )
        feature_indices.append(features_in_path)

    # Get unique feature indices across all paths for this leaf
    unique_features = np.unique(np.concatenate(feature_indices))

    return unique_features

# def get_features_used_in_path(tree, X):
#     """Get the feature indices used in the decision path to each leaf node."""
#     decision_paths = tree.decision_path(X)
#     feature_indices = []
#
#     for sample_id in range(X.shape[0]):
#         path = decision_paths.indices[
#                decision_paths.indptr[sample_id]: decision_paths.indptr[
#                    sample_id + 1]
#                ]
#         features_in_path = np.unique(
#             tree.tree_.feature[path][
#                 tree.tree_.feature[path] != _tree.TREE_UNDEFINED]
#         )
#         feature_indices.append(features_in_path)
#
#     # Get unique feature indices across all paths for this leaf
#     unique_features = np.unique(np.concatenate(feature_indices))
#
#     return unique_features

def fit_trees_on_leaves(tree, X, y):
    """Fit a new decision tree for the data at each leaf of the original tree."""
    leaf_samples_indices, leaf_features = get_leaf_samples_and_features(tree, X)
    leaf_trees = {}
    for leaf, sample_indices in leaf_samples_indices.items():
        X_leaf, y_leaf = X[sample_indices], y[sample_indices]
        if len(np.unique(
                y_leaf)) > 1:  # Ensure there is more than one class to fit a tree
            new_tree = DecisionTreeClassifier(min_samples_leaf=1,
                                              random_state=0)
            new_tree.fit(X_leaf, y_leaf)
            leaf_trees[leaf] = new_tree
    return leaf_trees, leaf_features

def calculate_accuracy_per_path(path_indices, y, path_classifications):
    accuracies = []
    for indices, classification in zip(path_indices, path_classifications):
        if len(indices) > 0:
            accuracy = np.mean(np.array(y[indices] == classification))
            accuracies.append(accuracy)
        else:
            accuracies.append(0.0)
    return accuracies

def prune_tree(decision_tree):
    """
    Prunes the decision tree by removing branches where the parent node
    and both child nodes have the same classification.

    Parameters:
    decision_tree (sklearn.tree.DecisionTreeClassifier or sklearn.tree.DecisionTreeRegressor): The decision tree to be pruned.
    """

    def is_leaf(node_id):
        return (decision_tree.children_left[node_id] == _tree.TREE_LEAF and
                decision_tree.children_right[node_id] == _tree.TREE_LEAF)

    def prune_node(node_id):
        left_child = decision_tree.children_left[node_id]
        right_child = decision_tree.children_right[node_id]

        # Check if both children are leaves
        if is_leaf(left_child) and is_leaf(right_child):
            # Get the classes of the parent and both children
            parent_class = decision_tree.value[node_id].argmax()
            left_class = decision_tree.value[left_child].argmax()
            right_class = decision_tree.value[right_child].argmax()

            # Prune if both children and parent have the same class
            if parent_class == left_class == right_class:
                decision_tree.children_left[node_id] = _tree.TREE_LEAF
                decision_tree.children_right[node_id] = _tree.TREE_LEAF

    def prune_recursively(node_id):
        # If not a leaf, first prune children
        if not is_leaf(node_id):
            left_child = decision_tree.children_left[node_id]
            right_child = decision_tree.children_right[node_id]
            prune_recursively(left_child)
            prune_recursively(right_child)
            prune_node(node_id)

    # Start pruning from the root
    prune_recursively(0)


def prune_tree_chaid(tree):
    """
    Recursively prunes the decision tree by removing branches where the parent node
    and all child nodes have the same classification.

    Parameters:
    - tree: The entire decision tree object.
    """

    def prune_node(node):
        """
        Recursively prunes child nodes of the given node.

        Parameters:
        - node: The current node in the tree.
        """
        if node.is_leaf:
            return

        # Recursively prune child nodes first
        for category, child in list(
                node.children.items()):  # Use list() to avoid runtime changes
            prune_node(child)

        # Check if all children are leaves and have the same class as the parent
        if all(child.is_leaf for child in node.children.values()):
            child_classes = {child.predicted_class for child in
                             node.children.values()}

            # If all children have the same predicted class and match the parent class, prune the children
            if len(child_classes) == 1:
                # Remove all child nodes and make the current node a leaf
                node.children.clear()
                node.is_leaf = True
                node.samples = sum(child.samples for child in
                                   node.children.values())  # Update sample count

    # Start pruning from the root of the tree
    prune_node(tree.root)

    # Update other properties of the tree as needed
    tree.node_count = sum(1 for _ in traverse_nodes_chaid(tree.root))
    return tree

def traverse_nodes_chaid(node):
    """
    Generator to traverse nodes in a tree and yield each node.
    """
    yield node
    for child in node.children.values():
        yield from traverse_nodes_chaid(child)

def weighted_node_count(tree, X_train):
    """Weighted node count by example"""
    leaf_indices = tree.apply(X_train)
    leaf_counts = np.bincount(leaf_indices)
    leaf_i = np.arange(tree.tree_.node_count)
    node_count = np.dot(leaf_i, leaf_counts) / float(X_train.shape[0])
    return node_count