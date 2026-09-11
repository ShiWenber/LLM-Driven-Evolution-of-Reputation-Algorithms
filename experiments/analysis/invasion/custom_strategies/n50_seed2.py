def observe(
    A_rep: float,
    A_action: str,
    B_rep: float,
    B_action: str,
    my_reputation: float
) -> float:
    # Reputation update focused on building and maintaining trust,
    # with stronger responses to clear exploitation.
    base = 0.3
    adaptive = base * (1.0 + abs(A_rep) * 0.5)
    
    if A_action == 'cooperate':
        if B_action == 'cooperate':
            # Mutual cooperation: solid reward, especially in cold start.
            cold_bonus = 0.5 if A_rep == 0.0 and B_rep == 0.0 else 0.0
            reward = adaptive * (1.5 + cold_bonus)
            # Steady growth without runaway.
            new_rep = A_rep + min(reward, 0.4 + abs(A_rep) * 0.3)
        else:
            # Cooperated but opponent defected.
            if B_rep < -0.1:
                # Opponent known bad – we tried to build, but they betrayed.
                # Slight reward for attempting cooperation with defector?
                # Better: small penalty for misjudgment.
                penalty = adaptive * 0.3
                new_rep = A_rep - min(penalty, 0.2)
            else:
                # Opponent seemed neutral/good but defected.
                # Strong penalty for being exploited.
                penalty = adaptive * (1.5 + max(0.0, B_rep) * 1.0)
                # If own rep high, even more costly.
                if A_rep > 0.2:
                    penalty *= 1.3
                new_rep = A_rep - min(penalty, 0.6)
    else:  # A defected
        if B_action == 'cooperate':
            # Defected against a cooperator.
            # Severe penalty if opponent good, less if opponent bad.
            if B_rep < -0.2:
                # Punishing a known defector – justified, slight reward.
                reward = adaptive * 0.2
                new_rep = A_rep + min(reward, 0.1)
            else:
                # Unjustified defection.
                penalty = adaptive * (3.0 + max(0.0, B_rep) * 2.0)
                # If own rep high, betrayal of trust is worse.
                if A_rep > 0.3:
                    penalty *= 1.5
                new_rep = A_rep - min(penalty, 0.8)
        else:
            # Mutual defection.
            if B_rep < -0.3:
                # Clearly punishing a bad agent – little or no penalty.
                # But if own rep high, might be seen as aggressive.
                if A_rep > 0.2:
                    penalty = adaptive * 0.2
                    new_rep = A_rep - min(penalty, 0.15)
                else:
                    # Justified response – small reward.
                    reward = adaptive * 0.15
                    new_rep = A_rep + min(reward, 0.1)
            elif B_rep < -0.1:
                # Somewhat bad opponent – mild penalty.
                penalty = adaptive * (0.4 - (-B_rep - 0.1) * 0.5)
                penalty = max(0.0, penalty)
                new_rep = A_rep - min(penalty, 0.3)
            else:
                # Neutral or good opponent mutual defection – penalty.
                penalty = adaptive * (1.0 if B_rep > 0 else 0.7)
                new_rep = A_rep - min(penalty, 0.4)
    
    return max(-1.0, min(1.0, new_rep))


def decide(
    my_reputation: float,
    opponent_reputation: float
) -> bool:
    # Strategy: cooperate unless there's strong evidence opponent is defective.
    # Key improvement: more forgiving towards slightly negative reputations,
    # but never cooperate with very bad opponents.
    
    # Cold start: cooperate to build mutual trust.
    if my_reputation == 0.0 and opponent_reputation == 0.0:
        return True
    
    # If opponent is clearly bad, never cooperate.
    if opponent_reputation <= -0.2:
        return False
    
    # If opponent is very good, always cooperate.
    if opponent_reputation >= 0.5:
        return True
    
    # Base threshold: slightly negative to encourage cooperation.
    baseline = -0.1 + 0.1 * my_reputation  # high own rep lowers barrier
    baseline = max(-0.2, min(0.2, baseline))
    
    # Adjust based on opponent rep.
    if opponent_reputation < 0:
        # Negative rep: require higher threshold.
        threshold = baseline + 0.2 * (-opponent_reputation)
    else:
        # Positive rep: lower threshold.
        threshold = baseline - 0.15 * opponent_reputation
    
    # Extra caution if own rep is low – need to rebuild trust.
    if my_reputation < -0.2:
        threshold += 0.1
    # If own rep very high, can be more forgiving.
    elif my_reputation > 0.4:
        threshold -= 0.1
    
    # Cooperate if opponent meets threshold.
    return opponent_reputation >= threshold