# app.py

import streamlit as st
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf

# -------------------- 🌟 Streamlit UI --------------------
st.title("🔬 Slurry 조성 추천 (Bayesian Optimization 기반)")
st.write("슬러리 조성에 따른 Yield Stress 값을 최대화하기 위한 추천 조성을 계산합니다.")

# -------------------- 📂 데이터 로딩 --------------------
try:
    df = pd.read_csv("slurry_data.csv")
except FileNotFoundError:
    st.error("❌ 'slurry_data.csv' 파일이 프로젝트 디렉토리에 없습니다.")
    st.stop()

x_cols = ["graphite_g", "carbon_black_g", "CMC_g", "solvent_g"]
y_cols = ["yield_stress"]

# -------------------- ⚙️ 데이터 전처리 --------------------
X_raw = df[x_cols].values
Y_raw = df[y_cols].values

x_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X_raw)

train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw, dtype=torch.double).unsqueeze(-1)  # (N, 1)

# -------------------- 📈 GP 모델 학습 --------------------
if st.button("📌 추천 조성 계산하기"):

    try:
        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # -------------------- 🎯 Acquisition Function --------------------
        best_y = train_y.max()
        acq_fn = ExpectedImprovement(model=model, best_f=best_y, maximize=True)

        bounds = torch.stack([
            torch.zeros(train_x.shape[1], dtype=torch.double),
            torch.ones(train_x.shape[1], dtype=torch.double)
        ])

        # -------------------- 🔍 최적 조성 탐색 --------------------
        candidate_scaled, _ = optimize_acqf(
            acq_function=acq_fn,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )

        candidate_np = candidate_scaled.detach().numpy()
        candidate_original = x_scaler.inverse_transform(candidate_np)

        # -------------------- ✅ 결과 출력 --------------------
        st.success("✅ 추천된 슬러리 조성 (g 기준):")
        for i, name in enumerate(x_cols):
            st.write(f"- **{name}**: {candidate_original[0][i]:.4f} g")

    except Exception as e:
        st.error(f"⚠️ 오류 발생: {e}")
