import os

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import graphviz
from networks.custom_dt_gini_with_entropy_metrics import CustomDecisionTree
from utils import create_group_indices


class DecisionPath:
    def __init__(self, node_indices, feature_indices, categories, node_indptr, feature_indptr):
        self.node_indices = node_indices
        self.feature_indices = feature_indices
        self.categories = categories
        self.node_indptr = node_indptr
        self.feature_indptr = feature_indptr


class CHAIDNode:
    def __init__(self, feature=None, categories=None, is_leaf=False, predicted_class=None, depth=0, id=None, num_samples=0, num_samples_per_class=None):
        self.feature = feature
        self.categories = categories
        self.is_leaf = is_leaf
        self.predicted_class = predicted_class
        self.children = {}
        self.depth = depth
        self.id = id
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class if num_samples_per_class is not None else []

class CHAIDTree:
    def __init__(self, n_classes, concept_names, max_depth=1000, min_child_size=10, alpha=0.05):
        self.max_depth = max_depth
        self.min_child_size = min_child_size
        self.alpha = alpha
        self.root = None
        self.node_count = 0
        self.n_classes = n_classes
        self.concept_names = concept_names
        self.concept_idx_map = create_group_indices(self.concept_names, squeeze_double=False)

    def fit(self, X, y):
        self.node_count = 0
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        id = self.node_count
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        num_samples = len(y)
        self.node_count += 1

        # If all samples belong to the same class, max depth is reached, or not enough samples, create a leaf node
        if len(np.unique(y)) == 1 or depth >= self.max_depth or num_samples < self.min_child_size:
            return CHAIDNode(is_leaf=True, predicted_class=predicted_class,
                             depth=depth, id=id, num_samples=num_samples,
                             num_samples_per_class=num_samples_per_class)

        # Find the best feature and categories to split
        best_feature, best_categories = self._find_best_split(X, y)

        # If no valid split is found, create a leaf node
        if best_feature is None:
            return CHAIDNode(is_leaf=True, predicted_class=predicted_class,
                             depth=depth, id=id, num_samples=num_samples,
                             num_samples_per_class=num_samples_per_class)

        # Create an internal node
        node = CHAIDNode(feature=best_feature, predicted_class=predicted_class,
                         categories=best_categories, depth=depth, id=id,
                         num_samples=num_samples,
                         num_samples_per_class=num_samples_per_class)

        # Split the data and grow child nodes recursively
        for category in best_categories:
            mask = (X[:, best_feature] == category)
            child_X = X[mask]
            child_y = y[mask]

            # If the child node does not satisfy the min_child_size constraint, stop splitting and make this node a leaf
            if len(child_y) < self.min_child_size:
                return CHAIDNode(is_leaf=True, predicted_class=predicted_class,
                                 depth=depth, id=id, num_samples=num_samples,
                                 num_samples_per_class=num_samples_per_class)

            # Create child node if it satisfies the constraint
            child_node = self._grow_tree(child_X, child_y, depth + 1)
            node.children[category] = child_node

        return node

    def _find_best_split(self, X, y):
        best_feature = None
        best_categories = None
        best_p_value = 1.0

        n_features = X.shape[1]

        for feature in range(n_features):
            categories = np.unique(X[:, feature])
            if len(categories) < 2:
                continue  # Skip features with less than two categories

            # Create a contingency table for the chi-square test
            contingency_table = pd.crosstab(X[:, feature], y)
            chi2, p_value, _, _ = chi2_contingency(contingency_table)

            if p_value < self.alpha and p_value < best_p_value:
                best_feature = feature
                best_categories = categories
                best_p_value = p_value

        return best_feature, best_categories

    def predict(self, X):
        # Predict the class for each sample in X
        return [self._predict_sample(sample, self.root) for sample in X]

    def _predict_sample(self, sample, node):
        # Recursively predict the class for a single sample
        if node.is_leaf:
            return node.predicted_class

        # Traverse the tree according to the feature split
        category = sample[node.feature]
        if category in node.children:
            return self._predict_sample(sample, node.children[category])
        else:
            # Default prediction if category not in children
            return node.predicted_class

    def apply(self, X):
        # Return the terminal node ID for each sample
        return [self._apply_sample(sample, self.root) for sample in X]

    def _apply_sample(self, sample, node):
        # Recursively find the terminal node ID for a single sample
        if node.is_leaf:
            return node.id

        category = sample[node.feature]
        if category in node.children:
            return self._apply_sample(sample, node.children[category])
        else:
            # Convention: It means that an unseen sample ends up in a leaf node that is not created
            return -1

    def decision_path(self, X):
        """
        Return the decision path for each sample in the format similar to scikit-learn.

        Parameters:
        - X: The input data, expected as a numpy array.

        Returns:
        - DecisionPath object containing node indices, feature indices,
          categories, and their respective indptr.
        """
        n_samples = X.shape[0]
        node_indices = []
        node_indptr = [0]
        feature_indices = []
        categories = []  # To store the categories leading to the corresponding child
        feature_indptr = [0]

        for sample in X:
            # Get the path for each sample, along with the feature indices and categories used
            path, features, cats = self._get_decision_path(sample, self.root)
            node_indices.extend(path)
            node_indptr.append(len(node_indices))
            categories.extend(cats)
            flattened_features = [self.concept_idx_map[features[i]][cats[i]] for i in range(len(cats))]
            feature_indices.extend(flattened_features)
            feature_indptr.append(len(feature_indices))

        return DecisionPath(np.array(node_indices, dtype=np.int32),
                            np.array(feature_indices, dtype=np.int32),
                            np.array(categories, dtype=np.int32),
                            np.array(node_indptr, dtype=np.int32),
                            np.array(feature_indptr, dtype=np.int32))

    def _get_decision_path(self, sample, node):
        """
        Recursively get the decision path, feature indices, and categories for a single sample.

        Parameters:
        - sample: A single input sample.
        - node: The current node in the tree.

        Returns:
        - A tuple containing a list of node IDs representing the path,
          a list of feature indices used along the path, and a list of categories.
        """
        path = [node.id]
        features = []
        cats = []  # To store categories leading to the next node

        # Check if node is a leaf
        if isinstance(node, CHAIDNode) and node.is_leaf:
            # For CHAIDNode, use is_leaf attribute
            return path, features, cats
        elif isinstance(node, CustomDecisionTree.Node) and (
                node.left is None and node.right is None):
            # For CustomDecisionTree.Node, check if left and right children are None
            return path, features, cats

        # If not a leaf, continue traversing based on the type of node
        if isinstance(node, CHAIDNode):
            # Traverse for CHAIDNode
            features.append(node.feature)
            category = sample[node.feature]
            if category in node.children:
                cats.append(category)  # Store the category leading to the next node
                child_path, child_features, child_cats = self._get_decision_path(
                    sample, node.children[category])
                path.extend(child_path)
                features.extend(child_features)
                cats.extend(child_cats)
        elif isinstance(node, CustomDecisionTree.Node):
            # Traverse for CustomDecisionTree.Node
            features.append(node.feature_index)
            if sample[node.feature_index] <= node.threshold:
                cats.append(0)  # Use 0 to indicate the left branch
                if node.left is not None:
                    child_path, child_features, child_cats = self._get_decision_path(
                        sample, node.left)
                    path.extend(child_path)
                    features.extend(child_features)
                    cats.extend(child_cats)
            else:
                cats.append(1)  # Use 1 to indicate the right branch
                if node.right is not None:
                    child_path, child_features, child_cats = self._get_decision_path(
                        sample, node.right)
                    path.extend(child_path)
                    features.extend(child_features)
                    cats.extend(child_cats)

        return path, features, cats

    def export_tree(self, feature_names, class_names, class_colors,
                    feature_value_names):
        """
        Exports the decision tree to Graphviz format for visualization.

        Parameters:
        - feature_names: List of feature names.
        - class_names: List of class names.
        - class_colors: List of colors corresponding to each class.
        - feature_value_names: Dictionary mapping feature indices and their values to human-readable names.
                              Example: {0: {1: "Low", 2: "Medium", 3: "High"}, 1: {1: "Yes", 0: "No"}}
        """
        dot_data = ["digraph Tree {"]
        dot_data.append(
            'node [shape=box, style="filled, rounded", fontname="helvetica"] ;')
        dot_data.append('edge [fontname="helvetica"] ;')

        def add_node(node, is_root=False):
            if node.is_leaf:
                # Prepare the class distribution for the leaf node
                value_str = "\\n".join(
                    [f"{class_names[i]}: {node.num_samples_per_class[i]}"
                     for i in range(len(class_names)) if
                     node.num_samples_per_class[i] > 0])
                # Determine the predicted class based on majority voting
                predicted_class_name = class_names[node.predicted_class]
                fillcolor = class_colors[
                    node.predicted_class % len(class_colors)]
                dot_data.append(
                    f'{node.id} [label="class: {predicted_class_name} \\n\\n{value_str}\\nsamples = {node.num_samples}", fillcolor="{fillcolor}"] ;')
            else:
                # Only show the feature name for non-leaf nodes
                label = f"{feature_names[node.feature]}"
                # if is_root:
                #     # Add number of samples to the root node
                #     label += f"\\nsamples = {node.num_samples}"
                dot_data.append(f'{node.id} [label="{label}"] ;')

            # Iterate over children, if any
            for category, child_node in node.children.items():
                # Get the human-readable name for the feature value
                if node.feature in feature_value_names and category in \
                        feature_value_names[node.feature]:
                    category_name = feature_value_names[node.feature][category]
                else:
                    category_name = str(category)

                dot_data.append(
                    f'{node.id} -> {child_node.id} [label="{category_name}"] ;')
                add_node(child_node)

        # Start with the root node and mark it as the root
        add_node(self.root, is_root=True)
        dot_data.append("}")
        return "\n".join(dot_data)

    def export_decision_paths(self, decision_paths, feature_names, class_colors,
                              class_names, feature_value_names,
                              output_dir="decision_paths", leaf_id=None):
        """
        Exports the decision paths for visualization.

        Parameters:
        - decision_paths: DecisionPath object containing the node indices and indptr for paths.
        - feature_names: List of feature names.
        - class_colors: List of colors corresponding to each class.
        - class_names: List of class names.
        - feature_value_names: Dictionary mapping feature indices and their values to human-readable names.
        - output_dir: Directory where the output files will be saved.
        - leaf_id: Identifier for the leaf node file.
        """

        # Define the colors for different conditions
        dark_grey = "#DDDDDD"  # Grey color
        light_grey = "#F7F7F7"  # Light grey color

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        node_path = decision_paths.node_indices[
                    decision_paths.node_indptr[0]: decision_paths.node_indptr[
                        1]]

        dot_data = ["digraph Tree {"]
        dot_data.append(
            'node [shape=box, style="filled, rounded", fontname="helvetica"] ;')
        dot_data.append('edge [fontname="helvetica"] ;')

        for i, id in enumerate(node_path):
            node = self._get_node_by_id(id)

            if node.is_leaf:
                # Leaf node: use the color of the predicted class
                fillcolor = class_colors[node.predicted_class % len(class_colors)]
                value_str = "\\n".join(
                    [f"{class_names[idx]}: {count}" for idx, count in
                     enumerate(node.num_samples_per_class) if count > 0])
                predicted_class_name = class_names[node.predicted_class]
                dot_data.append(
                    f'{id} [label="class: {predicted_class_name} \\n\\n{value_str}\\nsamples = {node.num_samples}", fillcolor="{fillcolor}"] ;')
            else:
                # Internal node: show the feature name and categorize
                label = f"{feature_names[node.feature]}"
                if i < len(node_path) - 1:
                    next_id = node_path[i + 1]
                    for category, child_node in node.children.items():
                        if child_node.id == next_id:
                            break

                    # Get the human-readable name for the feature value
                    if node.feature in feature_value_names and category in \
                            feature_value_names[node.feature]:
                        category_name = feature_value_names[node.feature][category]
                    else:
                        category_name = str(category)

                    fillcolor = dark_grey
                    dot_data.append(f'{id} [label="{label}", fillcolor="{fillcolor}"] ;')
                    dot_data.append(f'{id} -> {next_id} [label="{category_name}"] ;')
                else:
                    dot_data.append(f'{id} [label="{label}"] ;')

        dot_data.append("}")

        file_name = os.path.join(output_dir, f"decision_path_leaf_{leaf_id}.dot")
        with open(file_name, "w") as f:
            f.write("\n".join(dot_data))

        graph = graphviz.Source("\n".join(dot_data))
        graph.render(filename=file_name, format='pdf', cleanup=True)

    def _get_node_by_id(self, id):
        """
        Helper function to get a node by its id in a combined CHAID and Custom Decision Tree.

        Parameters:
        - id: The identifier of the node to find.

        Returns:
        - The node (CHAIDNode or CustomDecisionTree.Node) with the specified id or None if not found.
        """
        # Start with the root of the tree
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.id == id:
                return node

            # Traverse children for CHAIDNode
            if isinstance(node, CHAIDNode):
                for child_node in node.children.values():
                    if child_node is not None:
                        stack.append(child_node)

            # Traverse left and right nodes for CustomDecisionTree.Node
            elif isinstance(node, CustomDecisionTree.Node):
                if node.left is not None:
                    stack.append(node.left)
                if node.right is not None:
                    stack.append(node.right)

        # If no node with the specified id is found, return None
        return None



