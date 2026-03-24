import numpy as np

class PhysicsViolation(Exception):
    """Raised when an experiment result violates a fundamental physics invariant."""
    pass

def validate_minority_game_invariants(result, N_agents: int):
    """
    Validates that the minority game result adheres to physical and logical bounds.
    """
    # 1. Volatility Bound
    if result.volatility < 0:
        raise PhysicsViolation(f"Negative volatility detected: {result.volatility}. Volatility must be non-negative.")

    # 2. Attendance Conservation
    if not np.all((result.attendance >= 0) & (result.attendance <= N_agents)):
        raise PhysicsViolation("Attendance is out of bounds (0 to N_agents).")

    # 3. Binary Winners
    unique_winners = set(np.unique(result.winners))
    if not unique_winners.issubset({0, 1}):
        raise PhysicsViolation(f"Winners contain non-binary values: {unique_winners}")

    # Note: We do not validate the random baseline (M=1, beta=0) here because this validator
    # is run on *all* results, not just the random baseline. The random baseline is checked
    # in the null_test.py.
    
    return True
