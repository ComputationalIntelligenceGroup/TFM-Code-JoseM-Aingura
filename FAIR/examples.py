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
parser.add_argument("--setup", help="setup: (linear) or (nonlinear)", type=str, default="linear")

parser.add_argument("--target_col", help="target column name", type=str, default="likelihood")
parser.add_argument(
    "--remove_cols",
    help="extra columns to remove from X, separated by commas",
    type=str,
    default=""
)

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    "../../../data2/MT002_ENSAYO_3_NEW_FEATURES.csv",
    sep=";"
)

df = df.dropna()

timestamp_col = "timestamp"
target_col = args.target_col

# Convertir string "a,b,c" en lista ["a", "b", "c"]
extra_remove_cols = [
    col.strip()
    for col in args.remove_cols.split(",")
    if col.strip() != ""
]

# Comprobar que target existe
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataframe.")

# Columnas a eliminar de X
cols_to_drop = [timestamp_col, target_col] + extra_remove_cols

# Comprobar columnas inexistentes
missing_cols = [col for col in cols_to_drop if col not in df.columns]

if missing_cols:
    raise ValueError(f"Columns not found in dataframe: {missing_cols}")

y = df[target_col].to_numpy(dtype=np.float32)

X = df.drop(
    columns=cols_to_drop
).to_numpy(dtype=np.float32)

print("Target column:", target_col)
print("Columns removed from X:", cols_to_drop)
print("X shape:", X.shape)
print("y shape:", y.shape)

X = StandardScaler().fit_transform(X).astype(np.float32)

y = StandardScaler().fit_transform(
    y.reshape(-1, 1)
).ravel().astype(np.float32)

dim_x = X.shape[1]

# ---------------------------------------------------
# changepoint-based split
# ---------------------------------------------------

changepoints = [
    2460, 21325, 27365, 33015, 41890,
    54895, 58030, 68965, 89525, 98150
]

n_total = len(X)

# Mantener solo changepoints internos válidos
changepoints = [cp for cp in changepoints if 0 < cp < n_total]

# Este es el último changepoint interno: 98150
last_internal_cp = changepoints[-1]

# Este es el final real del dataset, no aparece en changepoints
final_endpoint = n_total

# ---------------------------------------------------
# train environments before the last environment
# ---------------------------------------------------

# ---------------------------------------------------
# build environments using changepoints
# ---------------------------------------------------

changepoints = [
    2460, 21325, 27365, 33015, 41890,
    54895, 58030, 68965, 89525, 98150
]

n_total = len(X)

# Añadimos inicio y final real del dataset
breaks = [0] + changepoints + [n_total]

# Mantener solo cortes válidos
breaks = sorted(set(b for b in breaks if 0 <= b <= n_total))

xs = []
ys = []

for start, end in zip(breaks[:-1], breaks[1:]):
    if end > start:
        xs.append(X[start:end])
        ys.append(y[start:end].reshape(-1, 1))

num_envs = len(xs)

print(f"Total samples: {n_total}")
print(f"Number of environments: {num_envs}")

for i, (x_env, y_env) in enumerate(zip(xs, ys)):
    print(f"Env {i}: X={x_env.shape}, y={y_env.shape}")

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


packs = algo.run_gumbel((xs, ys), eval_metric=np_mse, me_valid_data=None, me_test_data=None,
                       eval_iter=niters//10, log=True, device='gpu', diagnostics=True)
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