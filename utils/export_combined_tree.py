from networks.custom_chaid_tree import CHAIDNode
from networks.custom_dt_gini_with_entropy_metrics import CustomDecisionTree


def export_combined_tree(root, feature_names_main, feature_names_subtree, class_colors, class_names,
                         feature_value_names):
    """
    Exports the combined tree (CHAID tree with custom decision tree subtrees) to Graphviz format for visualization.

    Parameters:
    - feature_names: List of feature names.
    - class_names: List of class names.
    - class_colors: List of colors corresponding to each class.
    - feature_value_names: Dictionary mapping feature indices and their values to human-readable names.
                          Example: {0: {1: "Low", 2: "Medium", 3: "High"}, 1: {1: "Yes", 0: "No"}}
    """
    from decimal import Decimal, getcontext

    # Define the colors for different conditions
    light_grey = "#DDDDDD"  # Grey color
    light_yellow = "#F7F7F7"  # Light grey color

    dot_data = ["digraph Tree {"]
    dot_data.append(
        'node [shape=box, style="filled, rounded", fontname="helvetica"] ;')
    dot_data.append('edge [fontname="helvetica"] ;')

    def format_values(values, class_names):
        value_str = ""
        for i, count in enumerate(values):
            if count > 0:
                value_str += f"{class_names[i]}: {count}\\n"
        return value_str.strip()

    def _custom_print(number):
        # Set a high precision to handle very small numbers
        getcontext().prec = 6

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

    def add_node(node, is_root=False):
        if isinstance(node, CHAIDNode):
            # For CHAID tree node
            label = f"{feature_names_main[node.feature]}"
            dot_data.append(f'{node.id} [label="{label}"] ;')

            # Iterate over children of CHAIDNode
            for category, child_node in node.children.items():
                # Get the human-readable name for the feature value
                if node.feature in feature_value_names and category in \
                        feature_value_names[node.feature]:
                    category_name = feature_value_names[node.feature][
                        category]
                else:
                    category_name = str(category)

                dot_data.append(
                    f'{node.id} -> {child_node.id} [label="{category_name}"] ;')
                add_node(child_node)

        elif isinstance(node, CustomDecisionTree.Node):
            # For Custom Decision Tree node
            if node.left or node.right:  # It's an internal node
                threshold = node.threshold
                # Handle the special case for fixed splits
                if threshold == 0.5:
                    threshold_str = "<= 0.5"
                    fillcolor = light_grey
                else:
                    threshold_str = f"<= {threshold}"
                    # threshold_str = f"<= {_custom_print(threshold)}"
                    fillcolor = light_yellow

                dot_data.append(
                    f'{node.id} [label="{feature_names_subtree[node.feature_index]} {threshold_str}\\n'
                    f'gini = {node.gini:.2f}\\nentropy = {node.entropy:.4f}\\nsamples = {node.num_samples}\\n'
                    f'class = {class_names[node.predicted_class]}\\n'
                    f'info gain = {node.info_gain:.4f}\\ngain ratio = {node.gain_ratio:.4f}", fillcolor="{fillcolor}"] ;')

                add_node(node.left)
                add_node(node.right)
                dot_data.append(f'{node.id} -> {node.left.id} ;')
                dot_data.append(f'{node.id} -> {node.right.id} ;')
            else:
                # Leaf node: use the color of the predicted class
                fillcolor = class_colors[
                    node.predicted_class % len(class_colors)]
                value_str = format_values(node.num_samples_per_class,
                                          class_names)
                dot_data.append(
                    f'{node.id} [label="samples = {node.num_samples}\\n'
                    f'{value_str}\\nclass = {class_names[node.predicted_class]}", fillcolor="{fillcolor}"] ;')

    # Start with the root node and mark it as the root
    add_node(root, is_root=True)
    dot_data.append("}")
    return "\n".join(dot_data)