import streamlit as st
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf

st.title("🔬 Slurry 조성 추천 (Bayesian Optimization 기반)")

# CSV 데이터 로드
df = pd.read_csv("slurry_data.csv")

# 입력(X), 출력(Y) 정의
x_cols = ["graphite_g", "carbon_black_g", "CMC_g", "solvent_g"]
y_cols = ["yield_stress"]

X_raw = df[x_cols].values
Y_raw = df[y_cols].values

# 정규화
x_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X_raw)

# 텐서 변환 + 차원 조정
train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw, dtype=torch.double).view(-1, 1)  # 🔥 (N, 1) 형태

# 모델 생성 및 학습
model = SingleTaskGP(train_x, train_y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# Expected Improvement 계산
best_y = train_y.max()
acq_fn = ExpectedImprovement(model=model, best_f=best_y, maximize=True)

# 정규화된 경계 설정
bounds = torch.stack([
    torch.zeros(train_x.shape[1], dtype=torch.double),
    torch.ones(train_x.shape[1], dtype=torch.double)
])

# 최적화 (EI 최대화)
candidate_scaled, _ = optimize_acqf(
    acq_function=acq_fn,
    bounds=bounds,
    q=1,
    num_restarts=5,
    raw_samples=20,
)

# 역정규화
candidate_np = candidate_scaled.detach().numpy()
candidate_original = x_scaler.inverse_transform(candidate_np)

# 결과 출력
st.subheader("📌 추천된 조성 (단위: g)")
for i, name in enumerate(x_cols):
    st.write(f"**{name}**: {candidate_original[0][i]:.4f} g")
