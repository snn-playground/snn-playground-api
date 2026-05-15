from pydantic import BaseModel, Field
from typing import List, Dict
from enum import Enum

class SamplingMethod(str, Enum):
    BERNOULLI     = "bernoulli"
    UNIFORM       = "uniform"
    DETERMINISTIC = "deterministic"
    DETERMINISTIC2 = "deterministic_2"

class ConnectionType(str, Enum):
    DENSE      = "dense"
    ONE_TO_ONE = "one-to-one"

class BaseSimParams(BaseModel):
    T:            int   = Field(100,  ge=1)
    B:            int   = Field(10,   ge=1)
    N:            int   = Field(100,  ge=1)
    P:            float = Field(0.5,  ge=0.0, le=1.0)
    sampling:     SamplingMethod  = SamplingMethod.BERNOULLI
    H:            int   = Field(3,    ge=1)
    beta:         float = Field(0.95, ge=0.0, le=1.0)
    threshold:    float = Field(1.0)
    weights_mean: float = 0.0
    weights_std:  float = Field(0.1)
    bias_std:     float = Field(0.1)
    use_reset:    bool  = True
    conn_type:    ConnectionType = ConnectionType.DENSE
    seed: int | None = Field(None, description="Random seed for reproducibility")

class SimulationRequest(BaseSimParams):
    pass

class SimulationResponse(BaseModel):
    status: str
    shape:  Dict[str, int]
    input_spikes:        List[List[int]]         # [[t, neuron], ...]
    layer_spikes:        List[List[List[int]]]   # [layer][[t, neuron], ...]
    neuron_firing_rates: List[float]             # per-neuron rate, output layer
    layer_firing_rates:  List[float]

# ── Heatmap: sweep weights_std (X) × bias_std (Y) ───────────────────────────
class HeatmapRequest(BaseModel):
    T:            int   = Field(100,  ge=1)
    B:            int   = Field(5,    ge=1)
    N:            int   = Field(100,  ge=1)
    P:            float = Field(0.5,  ge=0.0, le=1.0)
    sampling:     SamplingMethod  = SamplingMethod.BERNOULLI
    H:            int   = Field(3,    ge=1)
    beta:         float = Field(0.95, ge=0.0, le=1.0)
    threshold:    float = Field(1.0)
    weights_mean: float = 0.0
    use_reset:    bool  = True
    conn_type:    ConnectionType = ConnectionType.DENSE
    weights_std_values: List[float]   # X axis (configurable, fixed at request time)
    bias_std_values:    List[float]   # Y axis
    seed: int | None = Field(None)

class HeatmapResponse(BaseModel):
    status:             str
    weights_std_values: List[float]
    bias_std_values:    List[float]
    probabilities:      List[List[float]]

class ThresholdSweepRequest(BaseModel):
    T:            int   = Field(100,  ge=1)
    B:            int   = Field(5,    ge=1)
    N:            int   = Field(100,  ge=1)
    P:            float = Field(0.5,  ge=0.0, le=1.0)
    sampling:     SamplingMethod  = SamplingMethod.BERNOULLI
    H:            int   = Field(3,    ge=1)
    beta:         float = Field(0.95, ge=0.0, le=1.0)
    weights_mean: float = 0.0
    weights_std:  float = Field(0.1)
    bias_std:     float = Field(0.1)
    use_reset:    bool  = True
    conn_type:    ConnectionType = ConnectionType.DENSE
    threshold_values: List[float]
    seed: int | None = Field(None)

class ThresholdSweepResponse(BaseModel):
    status:        str
    thresholds:    List[float]
    probabilities: List[float]