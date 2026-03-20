"""Game-theoretic models for quantum cognition."""

from qcccm.games.minority import (
    ClassicalAgent as ClassicalAgent,
    GameHistory as GameHistory,
    MinorityGameParams as MinorityGameParams,
    QuantumAgent as QuantumAgent,
    run_minority_game as run_minority_game,
    volatility_sweep as volatility_sweep,
)
from qcccm.games.agreement import (
    AgreementParams as AgreementParams,
    AgreementResult as AgreementResult,
    IsingParams as IsingParams,
    agreement_scaling as agreement_scaling,
    ising_energy as ising_energy,
    ising_magnetisation as ising_magnetisation,
    phase_diagram as phase_diagram,
    run_agreement_simulation as run_agreement_simulation,
    run_ising as run_ising,
)
