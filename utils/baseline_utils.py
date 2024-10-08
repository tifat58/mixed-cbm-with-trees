from typing import List, Tuple
import torch
from sklearn.metrics import f1_score, accuracy_score
from sympy import to_dnf, lambdify

def test_explanation(formula: str, x: torch.Tensor, y: torch.Tensor, target_class: int,
                     mask: torch.Tensor = None, threshold: float = 0.5,
                     material: bool = False) -> Tuple[float, torch.Tensor]:
    """
    Tests a logic formula.

    :param formula: logic formula
    :param x: input data
    :param y: input labels (MUST be one-hot encoded)
    :param target_class: target class
    :param mask: sample mask
    :param threshold: threshold to get concept truth values
    :return: Accuracy of the explanation and predictions
    """
    if formula in ['True', 'False', ''] or formula is None:
        predictions = torch.zeros(y.shape[0])
        #y2 = torch.ones(y.shape[0])
        y2 = y[:, target_class]
        return 0.0, predictions, y2

    else:
        assert len(y.shape) == 2
        y2 = y[:, target_class]
        concept_list = [f"feature{i:010}" for i in range(x.shape[1])]
        # get predictions using sympy
        explanation = to_dnf(formula)
        fun = lambdify(concept_list, explanation, 'numpy')
        x = x.cpu().detach().numpy()
        predictions = fun(*[x[:, i] > threshold for i in range(x.shape[1])])
        if material:
            # material implication: (p=>q) <=> (not p or q)
            accuracy = torch.sum(torch.logical_or(torch.logical_not(predictions[mask]), y[mask])) / len(y[mask])
            accuracy = accuracy.item()
        else:
            # material biconditional: (p<=>q) <=> (p and q) or (not p and not q)
            accuracy = accuracy_score(predictions[mask], y2[mask])

        return accuracy, predictions, y2