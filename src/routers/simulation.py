import random
import torch
import numpy as np
from fastapi import APIRouter

from src.utils.model import SNN
from src.utils.dataset import generate_spikes_dataset
from src.schemas.simulation import (
    SimulationRequest, SimulationResponse,
    HeatmapRequest,   HeatmapResponse,
    ThresholdSweepRequest, ThresholdSweepResponse,
)

router = APIRouter(prefix="/simulation", tags=["Simulation"])


def _apply_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _spikes_to_coords(arr: np.ndarray) -> list[list[int]]:
    """(T, N) float32 array → [[t, neuron_idx], ...]"""
    return np.argwhere(arr == 1).tolist()


def _spike_prob(model: SNN, dataset: torch.Tensor) -> float:
    """Mean spike probability across all neurons/timesteps in the output layer."""
    with torch.no_grad():
        y = model(dataset)          # reset_net is called inside forward; [B, T, N]
    return float(y.mean().item())


# ── Main simulation ──────────────────────────────────────────────────────────

@router.post("", response_model=SimulationResponse)
async def run_simulation(payload: SimulationRequest):
    _apply_seed(payload.seed)

    dataset = generate_spikes_dataset(
        sampling=payload.sampling, n_samples=payload.B,
        t_steps=payload.T, n_neurons=payload.N, p_spike=payload.P,
    )
    model = SNN(
        n_neurons=payload.N,     n_layers=payload.H,
        beta=payload.beta,       threshold=payload.threshold,
        weights_mean=payload.weights_mean,
        weights_std=payload.weights_std,
        bias_std=payload.bias_std,
        use_reset=payload.use_reset,
        conn_type=payload.conn_type,
    )

    with torch.no_grad():
        _, all_layer_outputs = model(dataset, return_all_layers=True)

    max_n = min(payload.max_neurons_display, payload.N)

    input_spikes = _spikes_to_coords(dataset[0, :, :max_n].numpy())
    layer_spikes = [
        _spikes_to_coords(l_out[0, :, :max_n].numpy())
        for l_out in all_layer_outputs
    ]

    output_all         = all_layer_outputs[-1][0].numpy()          # [T, N]
    neuron_firing_rates = output_all.mean(axis=0).tolist()          # per neuron
    layer_firing_rates  = [float(l[0].mean().item()) for l in all_layer_outputs]
    
    return SimulationResponse(
        status="success",
        shape={"samples": payload.B, "timesteps": payload.T, "neurons": payload.N},
        input_spikes=input_spikes,
        layer_spikes=layer_spikes,
        neuron_firing_rates=neuron_firing_rates,
        layer_firing_rates=layer_firing_rates,
    )

# ── Heatmap ──────────────────────────────────────────────────────────────────

@router.post("/heatmap", response_model=HeatmapResponse)
async def run_heatmap(payload: HeatmapRequest):
    _apply_seed(payload.seed)

    dataset = generate_spikes_dataset(
        sampling=payload.sampling, n_samples=payload.B,
        t_steps=payload.T, n_neurons=payload.N, p_spike=payload.P,
    )

    probs: list[list[float]] = [
        [
            _spike_prob(
                SNN(
                    n_neurons=payload.N,     n_layers=payload.H,
                    beta=payload.beta,       threshold=payload.threshold,
                    weights_mean=payload.weights_mean,
                    weights_std=w_std,       bias_std=b_std,
                    use_reset=payload.use_reset,
                    conn_type=payload.conn_type,
                ),
                dataset,
            )
            for w_std in payload.weights_std_values
        ]
        for b_std in payload.bias_std_values
    ]

    return HeatmapResponse(
        status="success",
        weights_std_values=payload.weights_std_values,
        bias_std_values=payload.bias_std_values,
        probabilities=probs,
    )


# ── Threshold sweep ──────────────────────────────────────────────────────────

@router.post("/threshold-sweep", response_model=ThresholdSweepResponse)
async def run_threshold_sweep(payload: ThresholdSweepRequest):
    _apply_seed(payload.seed)

    dataset = generate_spikes_dataset(
        sampling=payload.sampling, n_samples=payload.B,
        t_steps=payload.T, n_neurons=payload.N, p_spike=payload.P,
    )

    probs = [
        _spike_prob(
            SNN(
                n_neurons=payload.N,   n_layers=payload.H,
                beta=payload.beta,     threshold=thresh,
                weights_mean=payload.weights_mean,
                weights_std=payload.weights_std,
                bias_std=payload.bias_std,
                use_reset=payload.use_reset,
                conn_type=payload.conn_type,
            ),
            dataset,
        )
        for thresh in payload.threshold_values
    ]

    return ThresholdSweepResponse(
        status="success",
        thresholds=payload.threshold_values,
        probabilities=probs,
    )