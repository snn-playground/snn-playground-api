import torch
import numpy as np
import matplotlib.pyplot as plt

def raster(input: torch.Tensor, output: torch.Tensor):
    """
    Args:
        input: torch.tensor([D,T,N])
        output: torch.tensor([D,T,N])
        
    return: Plot spikes across neurons and time steps for a chosen sample in the dataset.
    """
    n_neurons, _ = output.shape

    fig_raster, axes = plt.subplots(1, 2, figsize=(12, 5))

    for neuron in range(n_neurons):
        spike_times = np.where(input[neuron] == 1)[0]
        axes[0].scatter(
            spike_times,
            np.ones_like(spike_times) * neuron,
            s=2,
            color="grey"
        )

    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Neuron index")
    axes[0].set_title(f"Input raster plot (1 sample, {n_neurons} neurons)")

    for neuron in range(n_neurons):
        spike_times = np.where(output[neuron] == 1)[0]
        axes[1].scatter(
            spike_times,
            np.ones_like(spike_times) * neuron,
            s=2,
            color="grey"
        )

    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Neuron index")
    axes[1].set_title(f"Output raster plot (1 sample, {n_neurons} neurons)")
    plt.tight_layout()
    return fig_raster
