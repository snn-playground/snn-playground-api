import torch
from torch import nn
from spikingjelly.activation_based import neuron as sj_neuron, functional, surrogate
from src.schemas.simulation import ConnectionType


class SNN(nn.Module):
    """
    SNN using SpikingJelly's LIFNode with step_mode='m'.

    Dynamics mapping from original:
        tau        = 1 / (1 - beta)         [clamped to avoid div-by-zero]
        use_reset  = True  → soft reset  (v -= threshold after spike)
        use_reset  = False → hard reset  (v  = 0       after spike)

    Note: SpikingJelly applies the reset at the current timestep;
    the original subtracted threshold * S[t-1] at the next timestep.
    Behaviour is qualitatively identical for visualisation purposes.
    """
    def __init__(
        self,
        n_neurons:    int   = 100,
        n_layers:     int   = 3,
        beta:         float = 0.95,
        threshold:    float = 1.0,
        weights_mean: float = 0.0,
        weights_std:  float = 0.1,
        bias_std:     float = 0.1,
        use_reset:    bool  = False,
        conn_type:    ConnectionType = ConnectionType.DENSE,
    ):
        super().__init__()
        self.n_neurons    = n_neurons
        self.n_layers     = n_layers
        self.beta         = beta
        self.threshold    = threshold
        self.weights_mean = weights_mean
        self.weights_std  = weights_std
        self.bias_std     = bias_std
        self.use_reset    = use_reset
        self.conn_type    = conn_type

        tau     = 1.0 / max(1.0 - beta, 1e-6) 
        v_reset = None if use_reset else 0.0     

        self.fcs  = nn.ModuleList()
        self.lifs = nn.ModuleList()

        for _ in range(n_layers):
            self.fcs.append(self._make_layer(n_neurons, n_neurons))
            self.lifs.append(
                sj_neuron.LIFNode(
                    tau=tau,
                    v_threshold=threshold,
                    v_reset=v_reset,
                    surrogate_function=surrogate.ATan(),
                    decay_input=False,
                    step_mode='m',
                )
            )

    def _make_layer(self, n_in: int, n_out: int) -> nn.Linear:
        if self.conn_type == ConnectionType.DENSE:
            layer = nn.Linear(n_in, n_out)
            nn.init.normal_(layer.weight, mean=self.weights_mean, std=self.weights_std)

        elif self.conn_type == ConnectionType.ONE_TO_ONE:
            layer = nn.Linear(n_in, n_out, bias=True)
            with torch.no_grad():
                layer.weight.zero_()
                layer.weight.fill_diagonal_(1.0)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.0, std=self.bias_std)

        return layer

    def forward(self, x: torch.Tensor, return_all_layers: bool = False):
        """
        Args:
            x: [B, T, N]
        Returns:
            [B, T, N]  or  ([B,T,N], list[[B,T,N]])  when return_all_layers=True
        """
        functional.reset_net(self) 

        h = x.permute(1, 0, 2)
        layer_outputs: list[torch.Tensor] = []

        for fc, lif in zip(self.fcs, self.lifs):
            cur = fc(h)                     
            spk = lif(cur)                  
            if return_all_layers:
                layer_outputs.append(spk.permute(1, 0, 2))   
            h = spk

        output = h.permute(1, 0, 2)        

        if return_all_layers:
            return output, layer_outputs
        return output