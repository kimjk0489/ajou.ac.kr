import streamlit as st
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf

# Title
st.title("\U0001F52C Slurry 조성 추천 (Bayesian Optimization 기반)")

# Load data
csv_file = "slurry_data.csv"
df = pd.read_csv(csv_file)

# Features and target
i_cols = ["graphite_g", "carbon_black_g", "CMC_g", "solvent_g"]
o_cols = ["yield_stress"]  # 목표는 yield_stress만 높이는 것

X_raw = df[i_cols].values
Y_raw = df[o_cols].values

# Normalize
x_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X_raw)

train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw, dtype=torch.double)

# Ensure train_y is 2D
if train_y.ndim == 1:
    train_y = train_y.unsqueeze(1)

# GP Model
model = SingleTaskGP(train_x, train_y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# Acquisition Function
best_f = train_y.max()
ei = ExpectedImprovement(model=model, best_f=best_f, maximize=True)

# Bounds (normalized: 0~1)
bounds = torch.tensor([
    [0.0] * train_x.shape[1],
    [1.0] * train_x.shape[1]
], dtype=torch.double)

# Optimize acquisition function
candidate, _ = optimize_acqf(
    acq_function=ei,
    bounds=bounds,
    q=1,
    num_restarts=5,
    raw_samples=20,
)

# Rescale result
recommended_scaled = candidate.detach().numpy()
recommended_original = x_scaler.inverse_transform(recommended_scaled)

# Output
st.subheader(":bulb: 최적 Slurry 조성 추천")
for i, name in enumerate(i_cols):
    st.write(f"**{name}**: {recommended_original[0][i]:.4f} g")