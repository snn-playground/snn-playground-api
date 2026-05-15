import torch
from torch import nn
from .neuron import LIF
from src.schemas.simulation import ConnectionType

class SNN(nn.Module):
    """
    Feedforward Spiking Neural Network (SNN) composed of Linear layers followed by
    Leaky Integrate-and-Fire (LIF) neurons.

    This model processes temporal sequences and simulates spiking dynamics over time.

    Parameters:
        n_neurons : int
            Number of neurons per layer (input and hidden sizes).

        n_layers : int
            Number of hidden LIF layers

        beta : float
            Membrane decay factor (0 ≤ beta ≤ 1).
            - beta = 0   → no temporal memory
            - beta ≈ 1   → long memory

        threshold : float
            Firing threshold of LIF neurons.

        weights_mean : float
            Mean for weight initialization.

        weights_std : float
            Standard deviation for weight initialization.

        bias_std : float
            Standard deviation for bias initialization.

        use_reset : bool
            If True, applies soft reset after spike:
                V ← V - threshold * spike_prev
            Otherwise, no reset is applied.

    Notes:
        - Input layer is determined by the dataset
        - Internal LIF states (membrane potentials) are reset at the beginning of each forward pass.
        - Computation is sequential over time (explicit loop over T).
    """
    def __init__(
        self, 
        n_neurons: int = 100, 
        n_layers: int = 3, 
        beta: float = 0.95, 
        threshold: float = 1.0,
        weights_mean: float = 0.0,
        weights_std: float = 0.1,
        bias_std: float = 0.1,
        use_reset: bool = False,
        conn_type: ConnectionType = ConnectionType.DENSE):

        super().__init__()
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.beta = beta
        self.threshold = threshold
        self.weights_mean = weights_mean
        self.weights_std = weights_std
        self.bias_std = bias_std
        self.use_reset = use_reset
        self.conn_type = conn_type

        self.fcs = nn.ModuleList()
        self.lifs = nn.ModuleList()

        for _ in range(n_layers):
            self.fcs.append(self.make_layer(n_neurons, n_neurons))
            self.lifs.append(LIF(beta=beta, threshold=threshold, use_reset=use_reset))

    def make_layer(self, n_in, n_out):
        if self.conn_type == ConnectionType.DENSE:
            layer = nn.Linear(n_in, n_out)
            nn.init.normal_(layer.weight, mean=self.weights_mean, std=self.weights_std)
        elif self.conn_type == ConnectionType.ONE_TO_ONE:
            layer = nn.Linear(n_in, n_out, bias=True)
            with torch.no_grad():
                layer.weight.zero_()
                layer.weight.fill_diagonal_(1.0)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=self.weights_mean, std=self.bias_std)
        return layer
        
    def forward(self, x: torch.Tensor, return_all_layers: bool = False):
        """
        Forward pass of the spiking neural network over time.

        Args:
            x (torch.Tensor):
                Input sequence of shape [batch_size, T, n_neurons]

        Returns:
            torch.Tensor:
                Spike outputs over time with shape [batch_size, T, n_neurons]
                (binary values in {0, 1})
        """
        batch_size, T, n = x.shape
        device = x.device
        for lif in self.lifs:
            lif._reset(batch_size, n, device)

        layer_records = [[] for _ in range(self.n_layers)] if return_all_layers else None

        for t in range(T):
            cur = self.fcs[0](x[:, t, :])
            spk = self.lifs[0](cur)
            if return_all_layers:
                layer_records[0].append(spk)
            for i in range(1, len(self.fcs)):
                cur = self.fcs[i](spk)
                spk = self.lifs[i](cur)
                if return_all_layers:
                    layer_records[i].append(spk)

        if return_all_layers:
            # [T, batch, n] → [batch, T, n] for each layer
            all_layers = [torch.stack(recs, dim=0).permute(1, 0, 2) for recs in layer_records]
            return all_layers[-1], all_layers  # (final, all)

        # Rebuild final output only
        for lif in self.lifs:
            lif._reset(batch_size, n, device)
        spk_rec = []
        for t in range(T):
            cur = self.fcs[0](x[:, t, :])
            spk = self.lifs[0](cur)
            for i in range(1, len(self.fcs)):
                cur = self.fcs[i](spk)
                spk = self.lifs[i](cur)
            spk_rec.append(spk)
        return torch.stack(spk_rec, dim=1)
