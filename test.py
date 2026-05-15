from src.utils.model import SNN
from src.utils.dataset import generate_spikes_dataset
from tqdm import tqdm
from src.schemas.simulation import SamplingMethod, ConnectionType
import torch
import numpy as np

T = 10
B = 10
N = 100
P = 0.5
H = 3
WB = 0.1
BB = 0.1
TH = 1.0
BETA = 0.95
WM = 0.0
RESET = True
CT = ConnectionType.DENSE

dataset = generate_spikes_dataset(
    sampling=SamplingMethod.BERNOULLI,
    n_samples=B,
    t_steps=T,
    n_neurons=N,
    p_spike=P
)

model = SNN(
    n_neurons=N,
    n_layers=H,
    beta=BETA,
    threshold=TH,
    weights_mean=WM,
    weights_std=WB,
    bias_std=BB,
    use_reset=RESET,
    conn_type=CT
)

outputs = []

with torch.no_grad():
    with tqdm(total=B, desc="Processing samples") as pbar:
        for i in range(B):
            x = dataset[i:i+1]
            y = model(x)
            outputs.append(y)
            pbar.update(1)

outputs = torch.cat(outputs, dim=0)

test_sample = dataset[0].numpy().T
test_output = outputs[0].numpy().T 

input_spikes = np.argwhere(test_sample == 1)
output_spikes = np.argwhere(test_output == 1)

# fig_raster = raster(test_sample, test_output)
# fig_raster.savefig(os.path.join("./", f"rasterplot.png"), dpi=300)
