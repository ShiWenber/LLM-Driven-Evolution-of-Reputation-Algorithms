"""Baseline strategies for the v2 quantitative interface (PD version):
the 8 leading-eight norms from Ohtsuki & Iwasa 2006, rewritten for the
2-player simultaneous Prisoner's Dilemma.

Each norm has two components:
    - an assignment rule (observe how to update the OBSERVER's view of
        both players after a joint action)
  - an action rule (decide when to cooperate)

The 5-arg observe signature is:
    def observe(donor_reputation, donor_action,
                            recipient_reputation, recipient_action,
                            my_reputation) -> tuple[float, float]:
            # returns (new_donor_reputation, new_recipient_reputation)

The framework calls observe() once per observed joint action. In PD,
both players are active so BOTH are judged by each observer.

Conventions:
  - target_rep > 0  =>  "Good" reputation
  - target_rep <= 0  =>  "Bad" reputation
  - Step size 0.333 (1/R with R=3) for IS-style updates
  - Step size 0.5 for SJ/IS+ stricter updates
  - Return value clamped to [-1, 1]

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

In PD, each leading-eight's assignment rule is applied to BOTH the
donor and the recipient (independently, using each player's own action).
"""

STEP = 0.333    # IS-style step
BIG_STEP = 0.5  # SJ/IS+ style stricter step

# Convenience: G/B as 0.0 threshold
def _is_G(x): return x > 0.0
def _is_B(x): return x <= 0.0


# --- 1. IS (Image Scoring) -------------------------------------------------
# In PD: reward cooperation, punish defection, regardless of recipient.
IS = f'''
def observe(donor_reputation, donor_action, recipient_reputation, recipient_action, my_reputation):
    def _upd(target_reputation, target_action):
        if target_action == 'cooperate':
            new = target_reputation + {STEP}
        else:
            new = target_reputation - {STEP}
        return max(-1.0, min(1.0, new))
    return _upd(donor_reputation, donor_action), _upd(recipient_reputation, recipient_action)

def decide(my_reputation, opponent_reputation):
    return my_reputation > 0.0
'''


# --- 2. SS (Simple Standing) ----------------------------------------------
# In PD: only reward if BOTH the observer (self) and the target were Good
# AND the target cooperated. Otherwise punish. (No-op self-judgment case:
# if my_rep is B, can't recover; symmetric for target.)
SS = f'''
def observe(donor_reputation, donor_action, recipient_reputation, recipient_action, my_reputation):
    def _upd(target_reputation, target_action):
        if my_reputation > 0.0 and target_reputation > 0.0 and target_action == 'cooperate':
            new = target_reputation + {STEP}
        else:
            new = target_reputation - {STEP}
        return max(-1.0, min(1.0, new))
    return _upd(donor_reputation, donor_action), _upd(recipient_reputation, recipient_action)

def decide(my_reputation, opponent_reputation):
    return my_reputation > 0.0 and opponent_reputation > 0.0
'''


# --- 3. SJ (Stern Judging) ------------------------------------------------
# Like SS, but stronger punishment when the condition fails.
SJ = f'''
def observe(donor_reputation, donor_action, recipient_reputation, recipient_action, my_reputation):
    def _upd(target_reputation, target_action):
        if my_reputation > 0.0 and target_reputation > 0.0 and target_action == 'cooperate':
            new = target_reputation + {STEP}
        else:
            new = target_reputation - {BIG_STEP}
        return max(-1.0, min(1.0, new))
    return _upd(donor_reputation, donor_action), _upd(recipient_reputation, recipient_action)

def decide(my_reputation, opponent_reputation):
    return my_reputation > 0.0 and opponent_reputation > 0.0
'''


# --- 4. SC (Scoring) -------------------------------------------------------
# Lenient: only defection is judged Bad. Cooperation always Good (even
# if observer or target is B). In PD this means target is always
# rewarded for cooperation regardless of context.
SC = f'''
def observe(donor_reputation, donor_action, recipient_reputation, recipient_action, my_reputation):
    def _upd(target_reputation, target_action):
        if target_action == 'cooperate':
            new = target_reputation + {STEP}
        else:
            new = target_reputation - {STEP}
        return max(-1.0, min(1.0, new))
    return _upd(donor_reputation, donor_action), _upd(recipient_reputation, recipient_action)

def decide(my_reputation, opponent_reputation):
    return my_reputation > 0.0
'''


# --- 5. SH (Shunning) ------------------------------------------------------
# A B-target is shamed (always judged as B) regardless of action. A
# G-target that cooperates is rewarded. Otherwise punish.
SH = f'''
def observe(donor_reputation, donor_action, recipient_reputation, recipient_action, my_reputation):
    def _upd(target_reputation, target_action):
        if target_reputation <= 0.0:
            new = target_reputation - {BIG_STEP}
        elif target_action == 'cooperate':
            new = target_reputation + {STEP}
        else:
            new = target_reputation - {STEP}
        return max(-1.0, min(1.0, new))
    return _upd(donor_reputation, donor_action), _upd(recipient_reputation, recipient_action)

def decide(my_reputation, opponent_reputation):
    return my_reputation > 0.0 and opponent_reputation > 0.0
'''


# --- 6. IS+ (Image Scoring plus) ------------------------------------------
# Like IS but B-targets are punished even harder and never recover.
IS_PLUS = f'''
def observe(donor_reputation, donor_action, recipient_reputation, recipient_action, my_reputation):
    def _upd(target_reputation, target_action):
        if target_reputation <= 0.0:
            new = target_reputation - {BIG_STEP}
        elif target_action == 'cooperate':
            new = target_reputation + {STEP}
        else:
            new = target_reputation - {STEP}
        return max(-1.0, min(1.0, new))
    return _upd(donor_reputation, donor_action), _upd(recipient_reputation, recipient_action)

def decide(my_reputation, opponent_reputation):
    return my_reputation > 0.0
'''


# --- 7. SS+ (Simple Standing plus) -----------------------------------------
# Like SS but B-targets never recover (BIG_STEP punishment).
SS_PLUS = f'''
def observe(donor_reputation, donor_action, recipient_reputation, recipient_action, my_reputation):
    def _upd(target_reputation, target_action):
        if target_reputation <= 0.0:
            new = target_reputation - {BIG_STEP}
        elif my_reputation > 0.0 and target_action == 'cooperate':
            new = target_reputation + {STEP}
        else:
            new = target_reputation - {STEP}
        return max(-1.0, min(1.0, new))
    return _upd(donor_reputation, donor_action), _upd(recipient_reputation, recipient_action)

def decide(my_reputation, opponent_reputation):
    return my_reputation > 0.0 and opponent_reputation > 0.0
'''


# --- 8. SJ+ (Stern Judging plus) ------------------------------------------
# Like SJ but EVERY failure case (not just the G-cooperate condition)
# is punished with BIG_STEP. B-targets are shamed.
SJ_PLUS = f'''
def observe(donor_reputation, donor_action, recipient_reputation, recipient_action, my_reputation):
    def _upd(target_reputation, target_action):
        if target_reputation <= 0.0:
            new = target_reputation - {BIG_STEP}
        elif my_reputation > 0.0 and target_reputation > 0.0 and target_action == 'cooperate':
            new = target_reputation + {STEP}
        else:
            new = target_reputation - {BIG_STEP}
        return max(-1.0, min(1.0, new))
    return _upd(donor_reputation, donor_action), _upd(recipient_reputation, recipient_action)

def decide(my_reputation, opponent_reputation):
    return my_reputation > 0.0 and opponent_reputation > 0.0
'''


# --- 0. ALLC / ALLD (sanity baselines, not in the 8 leading-eight) ----------
ALLC = f'''
def observe(donor_reputation, donor_action, recipient_reputation, recipient_action, my_reputation):
    def _upd(target_reputation, target_action):
        if target_action == 'cooperate':
            new = target_reputation + {STEP}
        else:
            new = target_reputation - {STEP}
        return max(-1.0, min(1.0, new))
    return _upd(donor_reputation, donor_action), _upd(recipient_reputation, recipient_action)

def decide(my_reputation, opponent_reputation):
    return True
'''

ALLD = f'''
def observe(donor_reputation, donor_action, recipient_reputation, recipient_action, my_reputation):
    def _upd(target_reputation, target_action):
        if target_action == 'cooperate':
            new = target_reputation + {STEP}
        else:
            new = target_reputation - {STEP}
        return max(-1.0, min(1.0, new))
    return _upd(donor_reputation, donor_action), _upd(recipient_reputation, recipient_action)

def decide(my_reputation, opponent_reputation):
    return False
'''


# --- Index ----------------------------------------------------------------
BASELINES = {
    "ALLC":    ALLC,
    "ALLD":    ALLD,
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
