import torch
import torch.nn as nn

class LIF(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron model (discrete-time).

    This module simulates neuron membrane dynamics and generates binary spikes.

    Dynamics:

        V(t) = β * V(t-1) + I(t) - R(t)

        S(t) = H(V(t) - θ)

    where:
        - V(t): membrane potential at time t
        - β: decay factor (leak)
        - I(t): input current at time t
        - θ: firing threshold
        - S(t): spike output (0 or 1)
        - H: Heaviside step function

    Reset mechanism (optional):
        R(t) = θ * S(t-1)   if use_reset=True
        R(t) = 0            otherwise

    Parameters:
        beta : float
            Membrane decay factor (0 ≤ beta ≤ 1).

        threshold : float
            Firing threshold θ.

        use_reset : bool
            Whether to apply soft reset based on previous spike.

    Notes:
        - If beta = 0, the neuron behaves like a memoryless threshold unit.
        - Internal state must be reset manually before processing a new sequence.
    """
    def __init__(self, beta=0.95, threshold=1.0, use_reset = False):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.v = None
        self.s_prev = None
        self.use_reset = use_reset

    def _reset(self, batch_size, n_neurons,  device='cpu'):
        self.v = torch.zeros(batch_size, n_neurons, device=device)
        self.s_prev = torch.zeros(batch_size, n_neurons, device=device)

    def forward(self, input_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LIF neuron for a single timestep.

        Args:
            input_t (torch.Tensor):
                Input current at time t.
                Shape: [batch_size, n_neurons]

        Returns:
            torch.Tensor:
                Binary spike output at time t.
                Shape: [batch_size, n_neurons]
                Values are in {0, 1}.
        """
        if self.use_reset:
            re = self.threshold * self.s_prev
        else:
            re = 0
        if self.beta > 0:
            self.v = self.beta * self.v + input_t  - re
        else:
            self.v = input_t  - re
        spike = (self.v >= self.threshold).float()
        self.s_prev = spike
        return spike