import os
import graphviz
from networks.custom_dt_gini_with_entropy_metrics import CustomDecisionTree
from networks.custom_chaid_tree import CHAIDNode

def export_decision_paths_with_subtrees(combined_tree, decision_paths,
                                        feature_names_main, feature_names_subtree,
                                        class_colors, class_names,
                                        feature_value_names,
                                        output_dir="decision_paths",
                                        leaf_id=None):
    """
    Exports the decision paths for visualization for a combined CHAID tree with custom decision tree subtrees.

    Parameters:
    - decision_paths: DecisionPath object containing the node indices and indptr for paths.
    - feature_names: List of feature names.
    - class_colors: List of colors corresponding to each class.
    - class_names: List of class names.
    - feature_value_names: Dictionary mapping feature indices and their values to human-readable names.
    - output_dir: Directory where the output files will be saved.
    - leaf_id: Identifier for the leaf node file.
    """
    from decimal import Decimal, getcontext

    # Define the colors for different conditions
    dark_grey = "#DDDDDD"  # Grey color
    light_grey = "#F7F7F7"  # Light grey color

    def format_values(values, class_names):
        value_str = ""
        for i, count in enumerate(values):
            if count > 0:
                value_str += f"{class_names[i]}: {count}\\n"
        return value_str.strip()

    def _custom_print(number):
        # Set a high precision to handle very small numbers
        getcontext().prec = 50

        # Convert the number to a decimal
        num = Decimal(number.item())

        # Convert to string
        num_str = format(num, 'f')

        # Split the string into the integer and decimal parts
        integer_part, decimal_part = num_str.split('.')

        # Find the first three non-zero consecutive decimal digits
        non_zero_count = 0
        for i, digit in enumerate(decimal_part):
            if digit != '0':
                non_zero_count += 1
                if non_zero_count == 3:
                    break

        # Join the integer part with the truncated decimal part
        formatted_num = integer_part + '.' + decimal_part[:i + 1]
        return formatted_num

    def add_node(node, node_id):
        """
        Helper function to add a node to the Graphviz output.
        """
        if isinstance(node, CustomDecisionTree.Node):
            if node.left or node.right:
                threshold = node.threshold
                if threshold == 0.5:
                    threshold_str = "== 0"
                    fillcolor = dark_grey
                else:
                    threshold_str = f"<= {threshold}"
                    fillcolor = light_grey

                dot_data.append(
                    f'{node_id} [label="{feature_names_subtree[node.feature_index]} {threshold_str}\\n'
                    f'info gain = {node.info_gain:.4f}\\nsamples = {node.num_samples}\\n'
                    f'class = {class_names[node.predicted_class]}", fillcolor="{fillcolor}"] ;')

                left_id = node_id * 2 + 1
                right_id = node_id * 2 + 2
                add_node(node.left, left_id)
                add_node(node.right, right_id)
                dot_data.append(f'{node_id} -> {left_id} [label="True"] ;')
                dot_data.append(f'{node_id} -> {right_id} [label="False"] ;')
            else:
                # Leaf node: use the color of the predicted class
                fillcolor = class_colors[
                    node.predicted_class % len(class_colors)]
                value_str = format_values(node.num_samples_per_class,
                                          class_names)
                dot_data.append(
                    f'{node_id} [label="samples = {node.num_samples}\\n'
                    f'{value_str}\\nclass = {class_names[node.predicted_class]}", fillcolor="{fillcolor}"] ;')
        elif isinstance(node, CHAIDNode):
            # Process CHAID leaf nodes
            if node.is_leaf:
                fillcolor = class_colors[
                    node.predicted_class % len(class_colors)]
                value_str = "\\n".join(
                    [f"{class_names[idx]}: {count}" for idx, count in
                     enumerate(node.num_samples_per_class) if count > 0])
                predicted_class_name = class_names[node.predicted_class]
                dot_data.append(
                    f'{node_id} [label="class: {predicted_class_name} \\n\\n{value_str}\\nsamples = {node.num_samples}", fillcolor="{fillcolor}"] ;')

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract the decision path
    node_path = decision_paths.node_indices[
                decision_paths.node_indptr[0]: decision_paths.node_indptr[
                    1]]

    dot_data = ["digraph Tree {"]
    dot_data.append(
        'node [shape=box, style="filled, rounded", fontname="helvetica"] ;')
    dot_data.append('edge [fontname="helvetica"] ;')

    # Process all nodes except the final one
    for i, node_id in enumerate(node_path[:-1]):
        node = combined_tree._get_node_by_id(node_id)

        if isinstance(node, CHAIDNode):
            # Process CHAIDNode internal nodes
            label = f"{feature_names_main[node.feature]}"
            if i < len(node_path) - 1:
                next_id = node_path[i + 1]
                for category, child_node in node.children.items():
                    if child_node.id == next_id:
                        break

                # Get the human-readable name for the feature value
                if node.feature in feature_value_names and category in \
                        feature_value_names[node.feature]:
                    category_name = feature_value_names[node.feature][
                        category]
                else:
                    category_name = str(category)

                fillcolor = dark_grey
                dot_data.append(
                    f'{node_id} [label="samples = {node.num_samples}\\n{label}", fillcolor="{fillcolor}"] ;')
                dot_data.append(
                    f'{node_id} -> {next_id} [label="{category_name}"] ;')
            else:
                dot_data.append(f'{node_id} [label="{label}"] ;')

        elif isinstance(node, CustomDecisionTree.Node):
            # Process CustomDecisionTree internal nodes
            if node.left or node.right:  # It's an internal node
                threshold = node.threshold
                if threshold == 0.5:
                    threshold_str = "== 0"
                    fillcolor = dark_grey
                else:
                    threshold_str = f"<= {threshold}"
                    fillcolor = light_grey

                dot_data.append(
                    f'{node_id} [label="{feature_names_subtree[node.feature_index]} {threshold_str}\\n'
                    f'gini = {node.gini:.2f}\\nsamples = {node.num_samples}\\n'
                    f'class = {class_names[node.predicted_class]}", fillcolor="{fillcolor}"] ;')

                if i < len(node_path) - 1:
                    next_id = node_path[i + 1]
                    if node.left and node.left.id == next_id:
                        dot_data.append(
                            f'{node_id} -> {next_id} [label="True"] ;')
                    elif node.right and node.right.id == next_id:
                        dot_data.append(
                            f'{node_id} -> {next_id} [label="False"] ;')

    # Process the final node in the decision path
    final_node = combined_tree._get_node_by_id(node_path[-1])
    add_node(final_node, final_node.id)

    dot_data.append("}")

    file_name = os.path.join(output_dir,
                             f"decision_path_leaf_{leaf_id}.dot")
    with open(file_name, "w") as f:
        f.write("\n".join(dot_data))

    graph = graphviz.Source("\n".join(dot_data))
    graph.render(filename=file_name, format='pdf', cleanup=True)