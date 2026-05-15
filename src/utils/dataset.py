import numpy as np
import torch
from src.schemas.simulation import SamplingMethod

def generate_spikes_dataset(
    sampling: SamplingMethod,
    n_samples: int,
    t_steps: int,
    n_neurons: int,
    p_spike: float
) -> torch.Tensor:
    """
    Generate synthetic spike train datasets using different sampling strategies.

    The output is a 3D tensor representing binary spike activity over time.

    Args:
        sampling (SamplingMethod):
            Strategy used to generate spike trains.

        n_samples (int):
            Number of independent samples in the dataset.

        t_steps (int):
            Number of timesteps per sample.

        n_neurons (int):
            Number of neurons per sample.

        p_spike (float):
            Controls spike density:
                - Bernoulli: probability of spike per timestep
                - Deterministic: fraction of timesteps with spikes

    Returns:
        torch.Tensor:
            Binary spike dataset of shape:
                [n_samples, t_steps, n_neurons]
            Values are in {0, 1}.

    Notes:
        - Output is float tensor (0.0 / 1.0), not integer.
        - Each neuron is treated independently.
        - Temporal correlations depend entirely on sampling method.
    """
    if sampling == SamplingMethod.BERNOULLI:
        dataset = (np.random.rand(n_samples, t_steps, n_neurons) < p_spike).astype(float)

    elif sampling == SamplingMethod.UNIFORM:
        dataset = np.random.choice([0, 1], size=(n_samples, t_steps, n_neurons)).astype(float)

    elif sampling == SamplingMethod.DETERMINISTIC:
        dataset = np.zeros((n_samples, t_steps, n_neurons))
        n_spikes = int(p_spike * t_steps)

        for s in range(n_samples):
            for n in range(n_neurons):
                spike_times = np.linspace(0, t_steps-1, n_spikes, dtype=int)
                dataset[s, spike_times, n] = 1

    elif sampling == SamplingMethod.DETERMINISTIC2:
        dataset = np.zeros((n_samples, t_steps, n_neurons))
        n_spikes = int(p_spike * t_steps)

        for s in range(n_samples):
            for n in range(n_neurons):
                spike_times = np.random.choice(t_steps, size=n_spikes, replace=False)
                dataset[s, spike_times, n] = 1

    dataset = torch.from_numpy(dataset).float()
    return dataset