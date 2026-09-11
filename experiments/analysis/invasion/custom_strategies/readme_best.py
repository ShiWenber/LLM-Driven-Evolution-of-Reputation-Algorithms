"""README-recorded best strategy, adapted to the current one-directional observe API.

ORIGIN
------
This is the "best code" recorded in the project README from an earlier
interface generation. The legacy ``observe`` was BIDIRECTIONAL: it returned a
tuple ``(donor_new, recipient_new)`` computed in a single call.

ADAPTATION (2026-09-09)
-----------------------
The current agent-type1 interface is ONE-DIRECTIONAL. ``observe`` judges a
single target player ``A`` and returns only ``A``'s new reputation; the
framework calls it twice per joint action with the roles swapped (once for
each player) and clamps the result.

The legacy donor/recipient updates are perfectly symmetric and mutually
independent: each branch reads only the PRE-interaction reputations of both
players, and neither side's update depends on the other side's computed new
value. Consequently the one-directional adaptation keeps only the
"judge A (donor)" branch and returns ``A_new``. When the framework calls
``observe`` again with A/B swapped it reproduces the legacy ``recipient_new``
exactly. Numerical equivalence is checked in
``tmp/_verify_readme_best_equiv.py``.

``decide`` is carried over unchanged. ``my_reputation`` is accepted for
interface compatibility but unused (the legacy observe did not use it either).
"""


def observe(
    A_rep: float,
    A_action: str,
    B_rep: float,
    B_action: str,
    my_reputation: float,
) -> float:
    # Judge target player A (the actor/"donor") from both players' actions.
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
