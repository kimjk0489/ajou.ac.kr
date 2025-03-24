import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf

# 1. 데이터 불러오기
df = pd.read_csv("C:/Dev/PythonProject/slurry_data.csv")

# 2. 입력(X), 출력(Y) 분리
x_cols = ["graphite_g", "carbon_black_g", "CMC_g", "solvent_g"]
y_cols = ["yield_stress"]

X_raw = df[x_cols].values
Y_raw = df[y_cols].values

# 3. MinMax 정규화 (0~1 범위)
x_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X_raw)

# 4. Torch 텐서 변환 (double precision)
train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw, dtype=torch.double)

# 5. GP 모델 생성 및 학습
model = SingleTaskGP(train_x, train_y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# 6. Bayesian Optimization with LogExpectedImprovement
best_y = train_y.max()
acq_fn = LogExpectedImprovement(model=model, best_f=best_y, maximize=True)

# bounds는 정규화했기 때문에 항상 [0, 1]
bounds = torch.stack([
    torch.zeros(train_x.shape[1], dtype=torch.double),
    torch.ones(train_x.shape[1], dtype=torch.double)
])

# 7. EI 최대화 지점 탐색
candidate_scaled, _ = optimize_acqf(
    acq_function=acq_fn,
    bounds=bounds,
    q=1,
    num_restarts=5,
    raw_samples=20,
)

# 8. 추천된 조성 역변환 (정규화 → 원래 단위)
candidate_np = candidate_scaled.detach().numpy()
candidate_original = x_scaler.inverse_transform(candidate_np)

# 9. 출력
print("📌 추천된 조성 (원래 스케일):")
for i, name in enumerate(x_cols):
    print(f"{name}: {candidate_original[0][i]:.4f}")
