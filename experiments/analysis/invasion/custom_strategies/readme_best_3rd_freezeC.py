"""THIRD-ORDER ABLATION of readme_best: freeze the partner's action to COOPERATE.

Same as ``readme_best_3rd.py`` but instead of dropping the ``B_action`` terms,
it assumes every partner cooperated. This is the ``B_action := "cooperate"``
freeze requested as the "optimistic" ablation. Effects on readme_best:

    cooperate branch:  B_action == "defect" is never true  -> no -0.3 penalty
    defect   branch:   B_action == "cooperate" always true -> -0.4 always

``decide`` is unchanged, so any difference from readme_best isolates the value
of observing the partner's action versus assuming cooperation.
"""


def observe(
    A_rep: float,
    A_action: str,
    B_rep: float,
    B_action: str,
    my_reputation: float,
) -> float:
    B_action = "cooperate"  # freeze: assume the partner cooperated
    if A_action == "cooperate":
        weight = 0.4 if B_rep >= 0 else 0.2
        A_new = A_rep + weight * (1 - abs(A_rep))
        if B_action == "defect":
            A_new -= 0.3
    else:
        A_new = (
            A_rep - 0.5 if B_rep >= 0 else A_rep + 0.3 * (1 - abs(A_rep))
        )
        if B_action == "cooperate":
            A_new -= 0.4
    return max(-1.0, min(1.0, A_new))


def decide(my_reputation: float, opponent_reputation: float) -> bool:
    if opponent_reputation >= 0.3 and my_reputation > -0.35:
        return True
    if opponent_reputation < -0.2:
        return False
    return opponent_reputation > -0.1
