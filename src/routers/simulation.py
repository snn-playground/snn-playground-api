import torch, random
import numpy as np
from fastapi import APIRouter
from src.utils.model import SNN
from src.utils.dataset import generate_spikes_dataset
from src.schemas.simulation import (
    SimulationRequest, SimulationResponse,
    HeatmapRequest, HeatmapResponse,
    ThresholdSweepRequest, ThresholdSweepResponse,
)

def _apply_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

router = APIRouter(prefix="/simulation", tags=["Simulation"])

def _spikes_to_coords(tensor_2d: np.ndarray) -> list[list[int]]:
    """(T, N) binary array → [[t, neuron], ...]"""
    return np.argwhere(tensor_2d == 1).tolist()


def _output_spike_prob(model: SNN, dataset: torch.Tensor, B: int) -> float:
    """Average spike probability across all neurons/timesteps in the output layer."""
    outputs = []
    with torch.no_grad():
        for i in range(B):
            x = dataset[i : i + 1]
            y = model(x) 
            outputs.append(y)
    outputs = torch.cat(outputs, dim=0)
    return float(outputs.mean().item())

@router.post("", response_model=SimulationResponse)
async def run_simulation(payload: SimulationRequest):
    _apply_seed(payload.seed)
    dataset = generate_spikes_dataset(
        sampling=payload.sampling, n_samples=payload.B,
        t_steps=payload.T, n_neurons=payload.N, p_spike=payload.P,
    )
    model = SNN(
        n_neurons=payload.N, n_layers=payload.H,
        beta=payload.beta, threshold=payload.threshold,
        weights_mean=payload.weights_mean, weights_std=payload.weights_std,
        bias_std=payload.bias_std, use_reset=payload.use_reset,
        conn_type=payload.conn_type,
    )

    all_layer_outputs = [torch.zeros(payload.B, payload.T, payload.N) for _ in range(payload.H)]
    with torch.no_grad():
        for i in range(payload.B):
            x = dataset[i : i + 1]
            _, layers = model(x, return_all_layers=True)
            for l_idx, l_out in enumerate(layers):
                all_layer_outputs[l_idx][i] = l_out[0]

    layer_spikes = []
    for l_out in all_layer_outputs:
        sample = l_out[0].numpy()   # [T, N]
        layer_spikes.append(_spikes_to_coords(sample))

    output_sample = all_layer_outputs[-1][0].numpy()   # [T, N]
    firing_rates  = output_sample.mean(axis=0).tolist() 

    input_sample  = dataset[0].numpy()   # [T, N]
    input_spikes  = _spikes_to_coords(input_sample)

    layer_firing_rates = [
        float(l_out[0].mean().item())   # mean over T and N for sample 0
        for l_out in all_layer_outputs
    ]

    return SimulationResponse(
        status="success",
        shape={"samples": payload.B, "timesteps": payload.T, "neurons": payload.N},
        input_spikes=input_spikes,
        layer_spikes=layer_spikes,
        neuron_firing_rates=firing_rates,
        layer_firing_rates=layer_firing_rates,
    )


@router.post("/heatmap", response_model=HeatmapResponse)
async def run_heatmap(payload: HeatmapRequest):
    _apply_seed(payload.seed)
    dataset = generate_spikes_dataset(
        sampling=payload.sampling, n_samples=payload.B,
        t_steps=payload.T, n_neurons=payload.N, p_spike=payload.P,
    )
    probs: list[list[float]] = []

    for b_std in payload.bias_std_values:         # Y axis
        row: list[float] = []
        for w_std in payload.weights_std_values:  # X axis
            model = SNN(
                n_neurons=payload.N, n_layers=payload.H,
                beta=payload.beta, threshold=payload.threshold,
                weights_mean=payload.weights_mean,
                weights_std=w_std, bias_std=b_std,
                use_reset=payload.use_reset, conn_type=payload.conn_type,
            )
            row.append(_output_spike_prob(model, dataset, payload.B))
        probs.append(row)

    return HeatmapResponse(
        status="success",
        weights_std_values=payload.weights_std_values,
        bias_std_values=payload.bias_std_values,
        probabilities=probs,
    )


@router.post("/threshold-sweep", response_model=ThresholdSweepResponse)
async def run_threshold_sweep(payload: ThresholdSweepRequest):
    _apply_seed(payload.seed)
    dataset = generate_spikes_dataset(
        sampling=payload.sampling, n_samples=payload.B,
        t_steps=payload.T, n_neurons=payload.N, p_spike=payload.P,
    )
    probs: list[float] = []

    for thresh in payload.threshold_values:
        model = SNN(
            n_neurons=payload.N, n_layers=payload.H,
            beta=payload.beta, threshold=thresh,
            weights_mean=payload.weights_mean,
            weights_std=payload.weights_std, bias_std=payload.bias_std,
            use_reset=payload.use_reset, conn_type=payload.conn_type,
        )
        probs.append(_output_spike_prob(model, dataset, payload.B))

    return ThresholdSweepResponse(
        status="success",
        thresholds=payload.threshold_values,
        probabilities=probs,
    )