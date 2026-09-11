"""Canonical leading-eight truth tables with quantitative reputation updates.

Tables: Hilbe et al. (2018), PNAS Table 1, doi:10.1073/pnas.1810565115,
following Ohtsuki & Iwasa (2006). Quantitative assessment: Schmid et al.
(2023), doi:10.1038/s41467-023-37817-x. Scores normalise [-R,R] to [-1,1];
positive/negative assessments add/subtract 1/R; score >= 0 is GOOD.

Assessment uses the observer's ratings of the actor and recipient, not
the observer's self-rating. The engine applies this donor rule to both
simultaneous actions using pre-interaction scores. This game adaptation
is not an exact replication of the papers' sequential donation games.
IS and SH are additional comparators, not members of the leading eight.
Legacy plus labels have no validated canonical mapping and are rejected.
"""

BASELINE_VERSION = "leading-eight-hilbe2018-quantitative-v1"
LEADING_EIGHT = tuple(f"L{i}" for i in range(1, 9))
# Rows: GCG, GCB, BCG, BCB, GDG, GDB, BDG, BDB (actor/action/recipient).
ASSESSMENT_TABLES = {
    "L1": "GGGGBGBB", "L2": "GBGGBGBB",
    "L3": "GGGGBGBG", "L4": "GGGBBGBG",
    "L5": "GBGGBGBG", "L6": "GBGBBGBG",
    "L7": "GGGBBGBB", "L8": "GBGBBGBB",
}
# Rows: GG, GB, BG, BB (self/recipient).
ACTION_TABLES = {name: "CDCC" if name in ("L1", "L2") else "CDCD"
                 for name in LEADING_EIGHT}
STEP = 1.0 / 3.0


def make_leading_eight(name: str, radius: int = 3) -> str:
    """Return standalone strategy source; R controls assessment resolution."""
    if name not in ASSESSMENT_TABLES:
        raise KeyError(f"Unknown leading-eight norm: {name}")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
        raise ValueError("radius must be a positive integer")
    return f'''
# {BASELINE_VERSION}; {name}; R={radius}; threshold S=0.
def _good(score):
    # Numerical tolerance only: repeated thirds must recover neutral GOOD.
    return float(score) >= -1e-12

def observe(A_rep, A_action, B_rep, B_action, my_reputation):
    if A_action not in ('cooperate', 'defect'):
        raise ValueError('action must be cooperate or defect')
    row = (0 if A_action == 'cooperate' else 4) + (0 if _good(A_rep) else 2) + (0 if _good(B_rep) else 1)
    delta = 1 if {ASSESSMENT_TABLES[name]!r}[row] == 'G' else -1
    score = max(-1.0, min(1.0, A_rep + delta / {radius}))
    return 0.0 if abs(score) < 1e-12 else score

def decide(my_reputation, opponent_reputation):
    row = (0 if _good(my_reputation) else 2) + (0 if _good(opponent_reputation) else 1)
    return {ACTION_TABLES[name]!r}[row] == 'C'
'''


IS = '''
def observe(A_rep, A_action, B_rep, B_action, my_reputation):
    score = max(-1.0, min(1.0, A_rep + (1 if A_action == 'cooperate' else -1) / 3))
    return 0.0 if abs(score) < 1e-12 else score
def decide(my_reputation, opponent_reputation):
    return opponent_reputation >= -1e-12
'''
SH = '''
def observe(A_rep, A_action, B_rep, B_action, my_reputation):
    good = A_action == 'cooperate' and B_rep >= -1e-12
    score = max(-1.0, min(1.0, A_rep + (1 if good else -1) / 3))
    return 0.0 if abs(score) < 1e-12 else score
def decide(my_reputation, opponent_reputation):
    return opponent_reputation >= -1e-12
'''
ALLC = '''
def observe(A_rep, A_action, B_rep, B_action, my_reputation):
    return 1.0
def decide(my_reputation, opponent_reputation):
    return True
'''
ALLD = '''
def observe(A_rep, A_action, B_rep, B_action, my_reputation):
    return -1.0
def decide(my_reputation, opponent_reputation):
    return False
'''
BASELINES = {name: make_leading_eight(name) for name in LEADING_EIGHT}
SS = BASELINES["L3"]
SJ = BASELINES["L6"]
SC = IS  # Scoring synonym, never counted as a separate leading-eight norm.
BASELINES.update(ALLC=ALLC, ALLD=ALLD, IS=IS, SS=SS, SJ=SJ, SC=SC, SH=SH)


def get_baseline(name: str) -> str:
    if name in ("IS+", "SS+", "SJ+"):
        raise ValueError(f"{name} was a mislabelled legacy rule; choose an explicit L1--L8 norm")
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline: {name}. Available: {list(BASELINES)}")
    return BASELINES[name]
