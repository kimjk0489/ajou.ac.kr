# app.py
import streamlit as st
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf

# 제목
st.title("Slurry 조성 추천 (Bayesian Optimization 기반)")

# 데이터 업로드
df = pd.read_csv("slurry_data.csv")

x_cols = ["graphite_g", "carbon_black_g", "CMC_g", "solvent_g"]
y_cols = ["yield_stress"]

X_raw = df[x_cols].values
Y_raw = df[y_cols].values

# 정규화
x_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X_raw)

train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw, dtype=torch.double)

# GPR 모델
model = SingleTaskGP(train_x, train_y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# EI 계산
best_y = train_y.max()
acq_fn = LogExpectedImprovement(model=model, best_f=best_y, maximize=True)

bounds = torch.stack([
    torch.zeros(train_x.shape[1], dtype=torch.double),
    torch.ones(train_x.shape[1], dtype=torch.double)
])

candidate_scaled, _ = optimize_acqf(
    acq_function=acq_fn,
    bounds=bounds,
    q=1,
    num_restarts=5,
    raw_samples=20,
)

candidate_np = candidate_scaled.detach().numpy()
candidate_original = x_scaler.inverse_transform(candidate_np)

# 결과 출력
st.subheader("📌 추천된 조성:")
for i, name in enumerate(x_cols):
    st.write(f"{name}: {candidate_original[0][i]:.4f}")
