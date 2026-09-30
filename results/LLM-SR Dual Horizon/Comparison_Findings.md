# Comparison Findings: Standard LLM-SR vs. Dual-Horizon LLM-SR

## Simplified Performance Summary Table

| Benchmark System | Metric | Standard LLM-SR | Dual-Horizon LLM-SR | Winner / Benefit |
| :--- | :--- | :--- | :--- | :--- |
| **Exp. Decay (0% Noise)** | Traj. MSE / $R^2$ | $1.62 \times 10^{-6}$ / $1.0000$ | $1.62 \times 10^{-6}$ / $1.0000$ | **Tied** (Exact recovery) |
| **Exp. Decay (5% Noise)** | Traj. MSE / $R^2$ | $4.06 \times 10^{-3}$ / $0.9994$ | $4.06 \times 10^{-3}$ / $0.9994$ | **Tied** (Exact recovery) |
| **Exp. Decay (10% Noise)** | Traj. MSE<br>Traj. $R^2$<br>OOS $R^2$<br>Solver Stability | $0.4055 \pm 0.5077$<br>$0.9394 \pm 0.0758$<br>$0.6450 \pm 0.1658$<br>**1 Solver Crash** | $\mathbf{0.1337 \pm 0.1440}$<br>$\mathbf{0.9800 \pm 0.0215}$<br>$\mathbf{0.9800 \pm 0.0215}$<br>**100% Stable** | **Dual-Horizon**: Prevents blow-ups, $-67\%$ MSE error, $+51.9\%$ OOS $R^2$. |
| **Log. Growth (0% Noise)** | Traj. MSE / $R^2$ | $0.1184$ / $1.0000$ | $0.1184$ / $1.0000$ | **Tied** (Exact recovery) |
| **Log. Growth (5% Noise)** | Traj. MSE / $R^2$ | $7.4715$ / $0.9997$ | $7.4715$ / $0.9997$ | **Tied** (Exact recovery) |
| **Log. Growth (10% Noise)**| Traj. MSE / $R^2$ | $98.319$ / $0.9961$ | $98.319$ / $0.9961$ | **Tied** (Exact recovery) |
| **Oscillator (0%, Raw)** | Traj. MSE / $R^2$ | $1.16 \times 10^{-4}$ / $0.9999$ | $1.16 \times 10^{-4}$ / $0.9999$ | **Tied** (Exact $\sin\theta$) |
| **Oscillator (0%, Filtered)** | Traj. MSE / OOS $R^2$ | $1.33 \times 10^{-3}$ / **0.9988** | $5.58 \times 10^{-3}$ / 0.0202 | **Standard**: Filter distortion caused Dual-Horizon to prefer a linear fit. |
| **Oscillator (5%, Raw)** | Traj. MSE / $R^2$ | $1.0163$ / $0.1848$ | $\mathbf{0.7655}$ / $\mathbf{0.3860}$ | **Dual-Horizon**: $-24.7\%$ trajectory MSE. |
| **Oscillator (5%, Filtered)**| Traj. MSE / $R^2$ | $1.8734$ / $-0.5027$ | $\mathbf{1.7406}$ / $\mathbf{-0.3962}$ | **Dual-Horizon**: Lower rollout error. |
| **Oscillator (10%, Raw)** | Traj. MSE / $R^2$ | $0.7193$ / $0.4230$ | $1.0647$ / $0.1460$ | **Mixed**: Severe 2nd-derivative noise. |
| **Oscillator (10%, Filtered)**| Traj. MSE / $R^2$ | $0.6385$ / $0.4879$ | $\mathbf{0.2486}$ / $\mathbf{0.8006}$ | **Dual-Horizon**: $-61.1\%$ MSE, $+64.1\%$ $R^2$. |

