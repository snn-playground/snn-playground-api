import numpy as np
import torch
from src.schemas.simulation import SamplingMethod


def generate_spikes_dataset(
    sampling:  SamplingMethod,
    n_samples: int,
    t_steps:   int,
    n_neurons: int,
    p_spike:   float,
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
        data = (np.random.rand(n_samples, t_steps, n_neurons) < p_spike)

    elif sampling == SamplingMethod.UNIFORM:
        # p_spike is ignored — each cell is 0 or 1 with equal probability
        data = np.random.randint(0, 2, size=(n_samples, t_steps, n_neurons))

    elif sampling == SamplingMethod.DETERMINISTIC:
        # Identical evenly-spaced pattern across all samples and neurons
        n_spikes    = int(p_spike * t_steps)
        spike_times = np.round(np.linspace(0, t_steps - 1, n_spikes)).astype(int)
        data        = np.zeros((n_samples, t_steps, n_neurons), dtype=bool)
        data[:, spike_times, :] = True     # broadcast: no loops

    elif sampling == SamplingMethod.DETERMINISTIC2:
        n_spikes = int(p_spike * t_steps)
        # Random permutation trick: argsort(uniform noise) → random ordering
        perm = np.argsort(np.random.rand(n_samples, n_neurons, t_steps), axis=2)
        # First n_spikes positions in the permutation mark spike times
        spike_mask = perm < n_spikes                          # [n_samples, n_neurons, t_steps]
        data       = spike_mask.transpose(0, 2, 1)           # [n_samples, t_steps, n_neurons]
    return torch.from_numpy(data.astype(np.float32))