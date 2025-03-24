import streamlit as st
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

st.title("🔬 Slurry 조성 추천 (Bayesian Optimization 기반)")

# 1. 데이터 로딩
df = pd.read_csv("slurry_data.csv")

# 2. 입력(X)과 출력(y) 설정
x_cols = ["graphite_g", "carbon_black_g", "CMC_g", "solvent_g"]
y_cols = ["yield_stress"]  # 예: yield_stress만 최적화 대상

X_raw = df[x_cols].values
Y_raw = df[y_cols].values

# 3. 정규화
x_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X_raw)

# 4. Tensor로 변환 (주의: train_y는 .unsqueeze(-1) + .detach() 필요)
train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw, dtype=torch.double).unsqueeze(-1).detach()

# 5. Gaussian Process 모델 생성 및 학습
model = SingleTaskGP(train_x, train_y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# 6. Expected Improvement 계산
best_y = train_y.max()
acq_fn = ExpectedImprovement(model=model, best_f=best_y.item(), maximize=True)

# 7. 최적의 조성 탐색
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

# 8. 추천된 조성 복원
candidate_np = candidate_scaled.detach().numpy()
candidate_original = x_scaler.inverse_transform(candidate_np)

# 9. 결과 출력
st.subheader("✅ 추천된 조성 (원래 단위 기준)")
for i, name in enumerate(x_cols):
    st.write(f"**{name}**: {candidate_original[0][i]:.4f} g")
