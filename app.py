import streamlit as st
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf

# 1. 제목
st.title("🔬 Slurry 조성 추천 (Bayesian Optimization 기반)")

# 2. 데이터 불러오기
try:
    df = pd.read_csv("slurry_data.csv")
except FileNotFoundError:
    st.error("❌ slurry_data.csv 파일이 프로젝트 루트에 없습니다.")
    st.stop()

# 3. 입력(X), 출력(Y) 설정
x_cols = ["graphite_g", "carbon_black_g", "CMC_g", "solvent_g"]
y_col = "yield_stress"

X = df[x_cols].values
Y = df[[y_col]].values  # shape: (n, 1)

# 4. MinMax 정규화
x_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X)

y_tensor = torch.tensor(Y, dtype=torch.double)
x_tensor = torch.tensor(X_scaled, dtype=torch.double)

# 5. 모델 학습
model = SingleTaskGP(x_tensor, y_tensor)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# 6. 획득 함수 정의 (EI 사용)
best_y = y_tensor.max()
acq_fn = ExpectedImprovement(model=model, best_f=best_y, maximize=True)

# 7. 후보 조성 최적화 (정규화 범위 [0, 1])
bounds = torch.stack([
    torch.zeros(x_tensor.shape[1], dtype=torch.double),
    torch.ones(x_tensor.shape[1], dtype=torch.double)
])

candidate_scaled, _ = optimize_acqf(
    acq_function=acq_fn,
    bounds=bounds,
    q=1,
    num_restarts=5,
    raw_samples=20,
)

# 8. 역정규화 후 결과 출력
candidate_np = candidate_scaled.detach().numpy()
recommended = x_scaler.inverse_transform(candidate_np)

st.subheader("📌 추천된 조성 (원래 단위)")
for i, name in enumerate(x_cols):
    st.write(f"- {name}: {recommended[0][i]:.4f} g")
