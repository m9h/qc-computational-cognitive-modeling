from pydantic import BaseModel, Field
from typing import List

class MinorityGameSweep(BaseModel):
    memory_values: List[int] = Field(..., description="List of memory sizes (M) to test.")
    quantumness_values: List[float] = Field(..., description="List of quantumness values (q) to test.")
    beta: float = Field(..., description="Softmax inverse temperature for classical/quantum choice.")
    seeds: List[int] = Field(..., description="Random seeds for statistical averaging.")

class MinorityGameHypothesis(BaseModel):
    name: str = Field(..., description="Name of the experiment.")
    rationale: str = Field(..., description="Hypothesis or rationale behind the experiment.")
    N_agents: int = Field(..., description="Number of agents in the minority game. Must be odd.")
    n_rounds: int = Field(..., description="Number of rounds to play per game.")
    sweep: MinorityGameSweep

    def model_post_init(self, __context) -> None:
        if self.N_agents % 2 == 0:
            raise ValueError("N_agents must be an odd number to avoid ties.")
