"""THIRD-ORDER ABLATION of readme_best: remove the partner-action channel.

readme_best is a fourth-order rule in this repo's PD framing: its ``observe``
uses donor action, donor standing, recipient reputation AND recipient action
(``B_action``). The ``B_action`` terms are:

    cooperate branch:  if B_action == "defect":    A_new -= 0.3
    defect   branch:  if B_action == "cooperate": A_new -= 0.4

This file DROPS both terms, leaving a pure third-order rule whose assessment
depends only on (A_rep, A_action, B_rep) -- exactly the set of inputs used by
the canonical Leading Eight assessment tables. This is the principled
reduction: the dependence on the removed variable is removed, not imputed.

Everything else (including ``decide``) is byte-identical to readme_best, so a
performance difference is attributable to the fourth-order channel alone.

Sibling variants for the "freeze" form of the ablation:
    readme_best_3rd_freezeC.py  (assume partner cooperates)
    readme_best_3rd_freezeD.py  (assume partner defects)
"""


def observe(
    A_rep: float,
    A_action: str,
    B_rep: float,
    B_action: str,
    my_reputation: float,
) -> float:
    # Third-order: the partner's action (B_action) is not consulted.
    if A_action == "cooperate":
        weight = 0.4 if B_rep >= 0 else 0.2
        A_new = A_rep + weight * (1 - abs(A_rep))
    else:
        A_new = (
            A_rep - 0.5 if B_rep >= 0 else A_rep + 0.3 * (1 - abs(A_rep))
        )
    return max(-1.0, min(1.0, A_new))


def decide(my_reputation: float, opponent_reputation: float) -> bool:
    if opponent_reputation >= 0.3 and my_reputation > -0.35:
        return True
    if opponent_reputation < -0.2:
        return False
    return opponent_reputation > -0.1
