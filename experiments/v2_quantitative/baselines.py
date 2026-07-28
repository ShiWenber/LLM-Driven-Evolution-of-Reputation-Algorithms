"""Baseline strategies for the v2 quantitative interface: the 8 leading-eight
norms from Ohtsuki & Iwasa 2006 (J. Theoretical Biology 239, 435-444).

Each norm has two components: an action rule (decide when to cooperate) and
an assignment rule (evaluate how to update donor's reputation after the
action). Real-valued reputation in [-1, 1.0]; neutral is 0.0.

Conventions used:
  - donor_rep > 0  =>  "Good" reputation
  - recipient_rep > 0  =>  recipient is "Good"
  - Step size 0.333 (1/R with R=3) for IS-like updates; 0.5 for SJ-like
    stricter rules. This maps the integer-score rules of Ohtsuki-Iwasa
    onto the real-valued regime of v2.

The 8 leading-eight norms (canonical naming and definitions follow
Ohtsuki & Iwasa 2006, Table 1):

  1. IS   - Image Scoring
  2. SS   - Simple Standing
  3. SJ   - Stern Judging
  4. SC   - Scoring
  5. SH   - Shunning
  6. IS+  - Image Scoring plus
  7. SS+  - Simple Standing plus
  8. SJ+  - Stern Judging plus

Real-valued action/assignment table (truth table on (donor_rep sign,
action, recipient_rep sign)):

                 (G, C, G)  (G, C, B)  (G, D, G)  (G, D, B)  (B, C, G)  (B, C, B)  (B, D, G)  (B, D, B)
  IS  assign:        G         G         B         B         G         G         B         B
  SS  assign:        G         B         B         B         B         B         B         B
  SJ  assign:        G         B         B         B         B         B         B         B   (=SS in this table; distinction is in the + variants)
  SC  assign:        G         G         G         B         G         G         G         B
  SH  assign:        G         B         B         B         B         B         B         B   (=SS in this table; the SH aspect is in stricter B handling)
  IS+ assign:        G         G         B         B         B         B         B         B
  SS+ assign:        G         B         B         B         B         B         B         B   (=SS in this table; "+" means stricter donor treatment)
  SJ+ assign:        G         B         B         B         B         B         B         B   (=SS in this table; SJ+ has its own action rule)

  Decide: action chosen as a function of (donor_rep, recipient_rep):
    IS:   (G,G)->C, (G,B)->C, (B,G)->D, (B,B)->D
    SS:   (G,G)->C, (G,B)->D, (B,G)->D, (B,B)->D
    SJ:   (G,G)->C, (G,B)->D, (B,G)->D, (B,B)->D  (same as SS for action)
    SC:   (G,G)->C, (G,B)->C, (B,G)->D, (B,B)->D  (only G-donors cooperate)
    SH:   (G,G)->C, (G,B)->D, (B,G)->D, (B,B)->D
    IS+:  (G,G)->C, (G,B)->C, (B,G)->D, (B,B)->D  (same as IS)
    SS+:  (G,G)->C, (G,B)->D, (B,G)->D, (B,B)->D  (same as SS)
    SJ+:  (G,G)->C, (G,B)->D, (B,G)->D, (B,B)->D  (same as SS)

All "plus" variants share the same action rule as their base; the "+"
distinction is that **B-donors never recover** (B-donor's *any* action
remains B) — the assignment rules below reflect that.
"""

STEP = 0.333    # IS-style step
BIG_STEP = 0.5  # SJ/IS+ style stricter step

# Convenience: G/B as 0.0 threshold
def _is_G(x): return x > 0.0
def _is_B(x): return x <= 0.0


# --- 1. IS (Image Scoring) -------------------------------------------------
#   decide:  C if donor is G (regardless of recipient)
#   assign:  C->G, D->B (regardless of recipient or donor's old rep)
IS = f'''
def evaluate(donor_reputation, recipient_reputation, donor_action, recipient_action, my_reputation):
    if donor_action == 'cooperate':
        new = donor_reputation + {STEP}
    else:
        new = donor_reputation - {STEP}
    return max(-1.0, min(1.0, new))

def decide(my_reputation, opponent_reputation):
    return my_reputation > 0.0
'''


# --- 2. SS (Simple Standing) ----------------------------------------------
#   decide:  C only if both donor and recipient are G
#   assign:  only (G,C,G)->G; everything else -> B
SS = f'''
def evaluate(donor_reputation, recipient_reputation, donor_action, recipient_action, my_reputation):
    # Simple Standing: the donor is Good only if the donor was Good,
    # cooperated, and the recipient was Good. Else Bad.
    if donor_reputation > 0.0 and donor_action == 'cooperate' and recipient_reputation > 0.0:
        new = donor_reputation + {STEP}
    else:
        new = donor_reputation - {STEP}
    return max(-1.0, min(1.0, new))

def decide(my_reputation, opponent_reputation):
    return my_reputation > 0.0 and opponent_reputation > 0.0
'''


# --- 3. SJ (Stern Judging) ------------------------------------------------
#   decide:  C only if both donor and recipient are G  (same as SS)
#   assign:  only (G,C,G)->G; everything else -> B  (stricter than SS:
#           B-donors are judged as B even if they cooperated with a G
#           recipient)
SJ = f'''
def evaluate(donor_reputation, recipient_reputation, donor_action, recipient_action, my_reputation):
    if donor_reputation > 0.0 and donor_action == 'cooperate' and recipient_reputation > 0.0:
        new = donor_reputation + {STEP}
    else:
        new = donor_reputation - {BIG_STEP}
    return max(-1.0, min(1.0, new))

def decide(my_reputation, opponent_reputation):
    return my_reputation > 0.0 and opponent_reputation > 0.0
'''


# --- 4. SC (Scoring) -------------------------------------------------------
#   decide:  C if donor is G  (regardless of recipient)
#   assign:  only D->B; all other actions are G  (lenient)
SC = f'''
def evaluate(donor_reputation, recipient_reputation, donor_action, recipient_action, my_reputation):
    # Scoring: only defection is judged Bad. Cooperation always Good
    # (even if donor is B or recipient is B).
    if donor_action == 'defect':
        new = donor_reputation - {STEP}
    else:
        new = donor_reputation + {STEP}
    return max(-1.0, min(1.0, new))

def decide(my_reputation, opponent_reputation):
    return my_reputation > 0.0
'''


# --- 5. SH (Shunning) ------------------------------------------------------
#   decide:  C only if both donor and recipient are G  (same as SS)
#   assign:  like SS but with stronger punishment for B-donors
SH = f'''
def evaluate(donor_reputation, recipient_reputation, donor_action, recipient_action, my_reputation):
    # Shunning: a B-donor is judged as B regardless of action (shunned).
    if donor_reputation <= 0.0:
        new = donor_reputation - {BIG_STEP}
    elif donor_reputation > 0.0 and donor_action == 'cooperate' and recipient_reputation > 0.0:
        new = donor_reputation + {STEP}
    else:
        new = donor_reputation - {STEP}
    return max(-1.0, min(1.0, new))

def decide(my_reputation, opponent_reputation):
    return my_reputation > 0.0 and opponent_reputation > 0.0
'''


# --- 6. IS+ (Image Scoring plus) ------------------------------------------
#   decide:  C if donor is G  (same as IS)
#   assign:  like IS but B-donors never recover
IS_PLUS = f'''
def evaluate(donor_reputation, recipient_reputation, donor_action, recipient_action, my_reputation):
    if donor_reputation <= 0.0:
        new = donor_reputation - {BIG_STEP}
    elif donor_action == 'cooperate':
        new = donor_reputation + {STEP}
    else:
        new = donor_reputation - {STEP}
    return max(-1.0, min(1.0, new))

def decide(my_reputation, opponent_reputation):
    return my_reputation > 0.0
'''


# --- 7. SS+ (Simple Standing plus) -----------------------------------------
#   decide:  same as SS
#   assign:  same as SS but B-donors never recover
SS_PLUS = f'''
def evaluate(donor_reputation, recipient_reputation, donor_action, recipient_action, my_reputation):
    if donor_reputation <= 0.0:
        new = donor_reputation - {BIG_STEP}
    elif donor_reputation > 0.0 and donor_action == 'cooperate' and recipient_reputation > 0.0:
        new = donor_reputation + {STEP}
    else:
        new = donor_reputation - {STEP}
    return max(-1.0, min(1.0, new))

def decide(my_reputation, opponent_reputation):
    return my_reputation > 0.0 and opponent_reputation > 0.0
'''


# --- 8. SJ+ (Stern Judging plus) ------------------------------------------
#   decide:  same as SJ
#   assign:  like SJ but with stronger punishment overall
SJ_PLUS = f'''
def evaluate(donor_reputation, recipient_reputation, donor_action, recipient_action, my_reputation):
    if donor_reputation <= 0.0:
        new = donor_reputation - {BIG_STEP}
    elif donor_reputation > 0.0 and donor_action == 'cooperate' and recipient_reputation > 0.0:
        new = donor_reputation + {STEP}
    else:
        new = donor_reputation - {BIG_STEP}
    return max(-1.0, min(1.0, new))

def decide(my_reputation, opponent_reputation):
    return my_reputation > 0.0 and opponent_reputation > 0.0
'''


# --- Index ----------------------------------------------------------------
BASELINES = {
    "IS":      IS,
    "SS":      SS,
    "SJ":      SJ,
    "SC":      SC,
    "SH":      SH,
    "IS+":     IS_PLUS,
    "SS+":     SS_PLUS,
    "SJ+":     SJ_PLUS,
}


def get_baseline(name: str) -> str:
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline: {name}. Available: {list(BASELINES)}")
    return BASELINES[name]
