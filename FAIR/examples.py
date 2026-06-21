from data.model import *
from methods.tools import pooled_least_squares
from methods.fair_algo import *
from utils import *
import numpy as np
import os
import argparse
import time
import torch

# Enable cuDNN autotune for potential speedups
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

# common setup
parser.add_argument("--seed", help="random seed", type=int, default=9)
parser.add_argument("--n", help="number of samples", type=int, default=2000)
parser.add_argument("--setup", help="setup: (linear) or (nonlinear)", type=str, default='linear')


args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)


# Set data generating process and model





import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    "../../../data2/MT002_ENSAYO_3_NEW_FEATURES.csv",
    sep=";"
)

df = df.dropna()

timestamp_col = "timestamp"
target_col = "likelihood"

y = df[target_col].to_numpy(dtype=np.float32)

X = df.drop(
    columns=[timestamp_col, target_col]
).to_numpy(dtype=np.float32)

X = StandardScaler().fit_transform(X).astype(np.float32)

y = StandardScaler().fit_transform(
    y.reshape(-1, 1)
).ravel().astype(np.float32)

dim_x = X.shape[1]

# ---------------------------------------------------
# temporal split
# ---------------------------------------------------

n_total = len(X)

train_end = int(95000)
valid_end = int(100000)

X_train = X[:train_end]
y_train = y[:train_end]

X_valid = X[train_end:valid_end]
y_valid = y[train_end:valid_end]

X_test = X[valid_end:]
y_test = y[valid_end:]

# ---------------------------------------------------
# build train environments
# ---------------------------------------------------

env_size = 10000

xs = []
ys = []

for start in range(0, len(X_train), env_size):

    end = min(start + env_size, len(X_train))

    xs.append(X_train[start:end])

    ys.append(
        y_train[start:end].reshape(-1, 1)
    )

num_envs = len(xs)

# ---------------------------------------------------
# validation/test
# ---------------------------------------------------

xvs = [X_valid]
yvs = [y_valid.reshape(-1, 1)]

xts = [X_test]
yts = [y_test.reshape(-1, 1)]

valid = (xvs, yvs)
test = (xts, yts)

print(f"Training environments: {num_envs}")

for i in range(num_envs):
    print(f"Train env {i}: {xs[i].shape}")

print(f"Validation: {X_valid.shape}")
print(f"Test: {X_test.shape}")

print(f"Number of environments: {num_envs}")

for i in range(num_envs):
    print(f"Env {i}: {xs[i].shape}")




if args.setup == 'linear':
	
	# linear model with 50k iterations
	niters = 50000
	fair_model = FairMLP(num_envs, dim_x, 0, 0, 0, 0)


if args.setup == 'nonlinear':

	# nonlinear model with 70k iterations
	niters = 70000
	fair_model = FairMLP(num_envs, dim_x, 1, 128, 2, 196)

# Set hyper-parameters

from copy import deepcopy
hyper_params = deepcopy(aos_default_hyper_params)
hyper_params['niters'] = niters
hyper_params['batch_size'] = 256

# Set losses
def np_mse(x, y):
	return np.mean(np.square(x - y))

def torch_mse(y_hat, y):
	return 0.5 * torch.mean((y_hat - y) ** 2)


algo = FairGumbelAlgo(
    num_envs,
    dim_x,
    fair_model,
    5,
    torch_mse,
    hyper_params
)
feature_names = list(
    df.drop(columns=[timestamp_col, target_col]).columns
)


packs = algo.run_gumbel((xs, ys), eval_metric=np_mse, me_valid_data=valid, me_test_data=test,
                       eval_iter=niters//10, log=True, device='gpu', diagnostics=False)
print(packs.keys())

print_gate_during_training_features(
    feature_names,
    packs["gate_rec"],
    f"saved_results/Aingura-{args.setup}-logits.pdf"
)

print_train_valid_losses(
    packs["loss_rec"],
    num_train_envs=num_envs,
    num_valid_envs=1,
    tofile=f"saved_results/Aingura-{args.setup}-losses.pdf"
)


if args.setup == 'linear':
	beta = np.reshape(fair_model.g.relu_stack.weight.detach().cpu().numpy(), (-1)) * packs['gate_rec'][-1, :]
	print(f'Estimated beta = {beta}')
else:
	print(packs['loss_rec'])


feature_names = df.drop(columns=[timestamp_col, target_col]).columns




if args.setup == 'linear':

    for name, coef in zip(feature_names, beta):
        print(f"{name}: {coef:.6f}")

else:

    final_gate = packs['gate_rec'][-1]

    for name, gate in zip(feature_names, final_gate):
        print(f"{name}: {gate:.6f}")