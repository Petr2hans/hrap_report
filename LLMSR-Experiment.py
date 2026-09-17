import os
import sys
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.signal import wiener
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

SEEDS = [42, 50, 100]
NOISE_LEVELS = [0.00, 0.05, 0.10]
NOISE_LABELS = ["Noise Level 1", "Noise Level 2", "Noise Level 3"]
NOISE_NAMES = ["Noise Level 1 (0%)", "Noise Level 2 (5%)", "Noise Level 3 (10%)"]
WIENER_WINDOW = 5
LLM_SR_ITERATIONS = 5

def integrate_llmsr_1d(y0, skeleton, params, t_eval):
    safe_env = {"np": np, "numpy": np}
    try:
        exec(skeleton, safe_env)
        func = safe_env.get("equation")
        if func is None or not callable(func):
            return None, False, "Compilation failure: equation callable not found in skeleton"
    except Exception as e:
        return None, False, f"Compilation error: {type(e).__name__} ({str(e)})"
    t_span = (t_eval[0], t_eval[-1])
    def ode_system(t, y):
        with np.errstate(all="raise"):
            val = func(y[0], params)
            if np.isscalar(val) or isinstance(val, (int, float, np.number)):
                return [float(val)]
            elif isinstance(val, np.ndarray):
                return [float(val.item()) if val.size == 1 else float(val[0])]
            return [float(val)]
    try:
        solution = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method="RK45")
        if not solution.success or len(solution.t) < len(t_eval):
            reason = solution.message if solution.message else "Integration stopped before t_end"
            return None, False, f"Solver failure: {reason}"
        res = np.asarray(solution.y[0])
        if np.any(np.isnan(res)) or np.any(np.isinf(res)):
            return None, False, "Numerical instability: NaN or Inf in trajectory"
        return res, True, "Success"
    except OverflowError:
        return None, False, "Overflow: state blew up during ODE integration"
    except FloatingPointError as e:
        return None, False, f"Numerical instability: FloatingPointError ({str(e)})"
    except ZeroDivisionError:
        return None, False, "Singularity: ZeroDivisionError in vector field"
    except Exception as e:
        return None, False, f"Solver failure: {type(e).__name__} ({str(e)})"

def integrate_llmsr_2nd_order(y0, skeleton, params, t_eval):
    safe_env = {"np": np, "numpy": np}
    try:
        exec(skeleton, safe_env)
        func = safe_env.get("equation")
        if func is None or not callable(func):
            return None, False, "Compilation failure: equation callable not found in skeleton"
    except Exception as e:
        return None, False, f"Compilation error: {type(e).__name__} ({str(e)})"
    t_span = (t_eval[0], t_eval[-1])
    def ode_system(t, y):
        with np.errstate(all="raise"):
            val = func(y[0], params)
            if np.isscalar(val) or isinstance(val, (int, float, np.number)):
                f_val = float(val)
            elif isinstance(val, np.ndarray):
                f_val = float(val.item()) if val.size == 1 else float(val[0])
            else:
                f_val = float(val)
            return [y[1], f_val]
    try:
        solution = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method="RK45")
        if not solution.success or len(solution.t) < len(t_eval):
            reason = solution.message if solution.message else "Integration stopped before t_end"
            return None, False, f"Solver failure: {reason}"
        res = np.asarray(solution.y[0])
        if np.any(np.isnan(res)) or np.any(np.isinf(res)):
            return None, False, "Numerical instability: NaN or Inf in trajectory"
        return res, True, "Success"
    except OverflowError:
        return None, False, "Overflow: state blew up during ODE integration"
    except FloatingPointError as e:
        return None, False, f"Numerical instability: FloatingPointError ({str(e)})"
    except ZeroDivisionError:
        return None, False, "Singularity: ZeroDivisionError in vector field"
    except Exception as e:
        return None, False, f"Solver failure: {type(e).__name__} ({str(e)})"

def apply_independent_wiener(X, window_size=WIENER_WINDOW):
    if X.ndim == 1:
        return wiener(X, mysize=window_size)
    filtered = np.zeros_like(X)
    for col in range(X.shape[1]):
        filtered[:, col] = wiener(X[:, col], mysize=window_size)
    return filtered

def generate_exponential_decay(lambda_val=0.3, y0=10.0, t_max=10.0, num_points=200, noise_percentage=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(50)
    t = np.linspace(0.0, t_max, num_points)
    y_true = y0 * np.exp(-lambda_val * t)
    true_std = np.std(y_true)
    noise = rng.normal(loc=0.0, scale=1.0, size=num_points)
    y_noisy = y_true + noise * noise_percentage * true_std
    return t, y_true, y_noisy

def generate_logistic_growth(P0=100.0, r=1.1, K=1000.0, t_max=50.0, num_points=200, noise_percentage=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(50)
    t = np.linspace(0.0, t_max, num_points)
    A = (K - P0) / P0
    P_true = K / (1.0 + A * np.exp(-r * t))
    true_std = np.std(P_true)
    noise = rng.normal(loc=0.0, scale=1.0, size=num_points)
    p_noisy = P_true + noise * noise_percentage * true_std
    return t, P_true, p_noisy

def generate_pendulum_data(theta_max=np.pi/2, g=9.8, L=3.0, t_end=10.0, num_points=200, noise_percentage=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(50)
    t_eval = np.linspace(0.0, t_end, num_points)
    def exact_pendulum(t, y):
        theta, omega = y
        return [omega, -(g / L) * np.sin(theta)]
    sol = solve_ivp(exact_pendulum, [0.0, t_end], [theta_max, 0.0], t_eval=t_eval, rtol=1e-10, atol=1e-10)
    th_true = sol.y[0]
    omega_true = sol.y[1]
    noise_th = rng.normal(loc=0.0, scale=1.0, size=num_points)
    noise_omega = rng.normal(loc=0.0, scale=1.0, size=num_points)
    th_noisy = th_true + noise_th * noise_percentage * np.std(th_true)
    omega_noisy = omega_true + noise_omega * noise_percentage * np.std(omega_true)
    return t_eval, th_true, omega_true, th_noisy, omega_noisy

_BENCHMARK_HYPOTHESES = {
    "exponential decay": [
        "def equation(x, params):\n    return params[0] * (x ** 2)",
        "def equation(x, params):\n    return params[0] * np.exp(-params[1] * x)",
        "def equation(x, params):\n    return params[0] * (x ** 3) + params[1] * (x ** 2)",
        "def equation(x, params):\n    return params[0] * np.sqrt(np.abs(x))",
        "def equation(x, params):\n    return params[0] * x",
    ],
    "logistic growth": [
        "def equation(x, params):\n    return params[0] * x",
        "def equation(x, params):\n    return params[0] * (x ** 2)",
        "def equation(x, params):\n    return params[0] * x / (1.0 + params[1] * x)",
        "def equation(x, params):\n    return params[0] * x * (1.0 - (x / params[1]) ** 2)",
        "def equation(x, params):\n    return params[0] * x * (1.0 - x / params[1])",
    ],
    "pendulum": [
        "def equation(x, params):\n    return params[0] * x",
        "def equation(x, params):\n    return params[0] * (x ** 3)",
        "def equation(x, params):\n    return params[0] * np.cos(x)",
        "def equation(x, params):\n    return params[0] * x + params[1] * (x ** 3)",
        "def equation(x, params):\n    return params[0] * np.sin(params[1] * x)",
    ],
}

def generate_equation_skeleton(prompt):
    prompt_lower = prompt.lower()
    if "decay" in prompt_lower or "exponential" in prompt_lower:
        candidates = _BENCHMARK_HYPOTHESES["exponential decay"]
    elif "logistic" in prompt_lower:
        candidates = _BENCHMARK_HYPOTHESES["logistic growth"]
    elif "pendulum" in prompt_lower or "oscillator" in prompt_lower:
        candidates = _BENCHMARK_HYPOTHESES["pendulum"]
    else:
        candidates = _BENCHMARK_HYPOTHESES["exponential decay"]
    iter_match = re.search(r"iteration\s*[:=]?\s*(\d+)", prompt, re.IGNORECASE)
    if iter_match:
        idx = int(iter_match.group(1)) - 1
    else:
        idx = 0
    return candidates[idx % len(candidates)]

def evaluate_equation(equation_str, x_data, y_data):
    safe_env = {
        "np": np,
        "numpy": np,
        "__builtins__": {
            "range": range,
            "len": len,
            "float": float,
            "int": int,
            "abs": abs,
            "min": min,
            "max": max,
        }
    }
    try:
        exec(equation_str, safe_env)
        equation_fn = safe_env.get("equation")
        if equation_fn is None or not callable(equation_fn):
            return -1e10, None, 1e10
    except Exception:
        return -1e10, None, 1e10
    param_indices = [int(idx) for idx in re.findall(r"params\[(\d+)\]", equation_str)]
    num_params = max(param_indices) + 1 if param_indices else 1
    def loss_function(params):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = equation_fn(x_data, params)
                if not np.all(np.isfinite(pred)):
                    return 1e10
                return float(np.mean((pred - y_data) ** 2))
        except Exception:
            return 1e10
    x_max = float(np.max(np.abs(x_data))) if len(x_data) > 0 else 1.0
    candidate_p0s = [
        np.ones(num_params),
        np.full(num_params, 0.5),
        np.array([-1.0, 1.0, 1.0, 1.0][:num_params]),
        np.array([1.0, x_max, 1.0, 1.0][:num_params]),
        np.array([0.5, x_max, 1.0, 1.0][:num_params]),
        np.array([-3.0, 1.0, 1.0, 1.0][:num_params]),
        np.array([-10.0, 1.0, 1.0, 1.0][:num_params]),
    ]
    best_mse = float("inf")
    best_params = np.ones(num_params)
    for p0 in candidate_p0s:
        try:
            res = minimize(loss_function, p0, method="BFGS", options={"maxiter": 200, "disp": False})
            current_mse = loss_function(res.x)
            if current_mse < best_mse:
                best_mse = current_mse
                best_params = res.x
        except Exception:
            continue
    fitness = -best_mse
    return fitness, best_params, best_mse

def run_llm_sr_benchmark(benchmark_name, x_data, target_data, num_iterations=LLM_SR_ITERATIONS, verbose=False):
    experience_buffer = {
        "best_skeleton": None,
        "best_params": None,
        "best_mse": float("inf"),
        "history": []
    }
    prompt = (
        f"Benchmark: {benchmark_name}.\n"
        f"Task: Discover the governing differential equation.\n"
        f"Iteration: 1."
    )
    for it in range(1, num_iterations + 1):
        candidate_skeleton = generate_equation_skeleton(prompt)
        fitness, params, mse = evaluate_equation(candidate_skeleton, x_data, target_data)
        experience_buffer["history"].append({
            "iteration": it,
            "skeleton": candidate_skeleton,
            "params": params,
            "mse": mse,
            "fitness": fitness
        })
        updated = False
        if mse < experience_buffer["best_mse"]:
            experience_buffer["best_mse"] = mse
            experience_buffer["best_skeleton"] = candidate_skeleton
            experience_buffer["best_params"] = params
            updated = True
        if verbose:
            status = "[NEW BEST]" if updated else "[REJECTED]"
            first_line = candidate_skeleton.splitlines()[-1].strip()
            print(f"  Iter {it}: {first_line:<45} | MSE: {mse:.6e} {status}")
        prompt = (
            f"Benchmark: {benchmark_name}.\n"
            f"Task: Discover the governing differential equation.\n"
            f"Iteration: {it + 1}.\n"
            f"Current best equation skeleton:\n{experience_buffer['best_skeleton'].strip()}\n"
            f"Current best MSE: {experience_buffer['best_mse']:.6e}.\n"
            f"Please propose an improved candidate equation skeleton."
        )
    return experience_buffer

def format_discovered_equation(benchmark_name, skeleton, params):
    if skeleton is None or params is None:
        return "Failed to discover equation"
    match = re.search(r"return\s+(.+)", skeleton)
    expr = match.group(1).strip() if match else skeleton.strip()
    expr = expr.replace("np.", "")
    for idx, p in enumerate(params):
        expr = expr.replace(f"params[{idx}]", f"{p:.4f}")
    if "pendulum" in benchmark_name.lower() or "oscillator" in benchmark_name.lower():
        expr = re.sub(r"\bx\b", "θ", expr)
        expr = expr.replace("--", "+ ")
        return f"d²θ/dt² = {expr}"
    elif "logistic" in benchmark_name.lower():
        expr = re.sub(r"\bx\b", "P", expr)
        expr = expr.replace("--", "+ ")
        return f"dP/dt = {expr}"
    else:
        expr = re.sub(r"\bx\b", "y", expr)
        expr = expr.replace("--", "+ ")
        return f"dy/dt = {expr}"

def predict_equation_derivative(skeleton, params, x_eval):
    safe_env = {"np": np, "numpy": np}
    exec(skeleton, safe_env)
    eq_fn = safe_env["equation"]
    return np.asarray(eq_fn(x_eval, params)).flatten()

def main():
    print(f"Starting Multi-Realization LLM-SR ODE Discovery Benchmark across {len(SEEDS)} seeds...")
    all_run_records = []
    vis_data = {
        "exp": {},
        "log": {},
        "osc": {},
        "trajectories": {}
    }
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n>>> Running Realization {seed_idx + 1}/{len(SEEDS)} (Seed: {seed}) <<<")
        rng = np.random.default_rng(seed)

        lambda_val = 0.3
        y0_train = 10.0
        t_exp, y_true_exp, _ = generate_exponential_decay(lambda_val=lambda_val, y0=y0_train, rng=rng)
        dt_exp = t_exp[1] - t_exp[0]
        decay_true_deriv = -lambda_val * y_true_exp
        t_oos_exp = np.linspace(0.0, 10.0, 200)
        y0_oos_exp = 20.0
        true_traj_oos_exp = y0_oos_exp * np.exp(-lambda_val * t_oos_exp)

        for n_idx, noise in enumerate(NOISE_LEVELS):
            _, _, y_noisy_exp = generate_exponential_decay(
                lambda_val=lambda_val, y0=y0_train, noise_percentage=noise, rng=rng
            )
            target_deriv_exp = np.gradient(y_noisy_exp, dt_exp)
            exp_buffer = run_llm_sr_benchmark("Exponential Decay", y_noisy_exp, target_deriv_exp)
            best_sk_exp = exp_buffer["best_skeleton"]
            best_p_exp = exp_buffer["best_params"]
            eq_str_exp = format_discovered_equation("Exponential Decay", best_sk_exp, best_p_exp)
            pred_deriv_exp = predict_equation_derivative(best_sk_exp, best_p_exp, y_noisy_exp)
            mse_eq_exp = float(mean_squared_error(pred_deriv_exp, decay_true_deriv))
            traj_in_exp, ok_in_exp, msg_in_exp = integrate_llmsr_1d([y0_train], best_sk_exp, best_p_exp, t_exp)
            if ok_in_exp:
                mse_traj_exp = float(mean_squared_error(traj_in_exp, y_true_exp))
                r2_traj_exp = float(r2_score(y_true_exp, traj_in_exp))
            else:
                mse_traj_exp = np.nan
                r2_traj_exp = np.nan
            traj_oos_exp, ok_oos_exp, msg_oos_exp = integrate_llmsr_1d(
                [y0_oos_exp], best_sk_exp, best_p_exp, t_oos_exp
            )
            if ok_oos_exp:
                r2_oos_exp = float(r2_score(true_traj_oos_exp, traj_oos_exp))
            else:
                r2_oos_exp = np.nan
            all_run_records.append({
                "Seed": seed,
                "System": "Exponential Decay",
                "Method": "LLM-SR",
                "Noise Level": NOISE_LABELS[n_idx],
                "Noise Percentage": noise,
                "Method name": f"Exp. Decay: LLM-SR, {NOISE_LABELS[n_idx]}",
                "Discovered Diff eq.": eq_str_exp,
                "MSE of Diff eq.": mse_eq_exp,
                "MSE of trajectory": mse_traj_exp,
                "R^2 trajectory score": r2_traj_exp,
                "Out-of-sample R^2 score": r2_oos_exp,
                "Solver Status In-Sample": msg_in_exp,
                "Solver Status OOS": msg_oos_exp
            })
            if seed_idx == 0:
                if "llmsr_preds" not in vis_data["exp"]:
                    vis_data["exp"]["llmsr_preds"] = []
                    vis_data["exp"]["llmsr_trajs"] = []
                vis_data["exp"]["llmsr_preds"].append(pred_deriv_exp)
                vis_data["exp"]["llmsr_trajs"].append(
                    traj_in_exp if ok_in_exp else np.full_like(t_exp, np.nan)
                )

        r_val = 1.1
        K_val = 1000.0
        P0_train = 100.0
        t_log, P_true_log, _ = generate_logistic_growth(P0=P0_train, r=r_val, K=K_val, rng=rng)
        dt_log = t_log[1] - t_log[0]
        log_true_deriv = r_val * P_true_log * (1.0 - P_true_log / K_val)
        t_oos_log = np.linspace(0.0, 50.0, 200)
        P0_oos_log = 10.0
        A_oos = (K_val - P0_oos_log) / P0_oos_log
        true_traj_oos_log = K_val / (1.0 + A_oos * np.exp(-r_val * t_oos_log))

        for n_idx, noise in enumerate(NOISE_LEVELS):
            _, _, p_noisy_log = generate_logistic_growth(
                P0=P0_train, r=r_val, K=K_val, noise_percentage=noise, rng=rng
            )
            target_deriv_log = np.gradient(p_noisy_log, dt_log)
            log_buffer = run_llm_sr_benchmark("Logistic Growth", p_noisy_log, target_deriv_log)
            best_sk_log = log_buffer["best_skeleton"]
            best_p_log = log_buffer["best_params"]
            eq_str_log = format_discovered_equation("Logistic Growth", best_sk_log, best_p_log)
            pred_deriv_log = predict_equation_derivative(best_sk_log, best_p_log, p_noisy_log)
            mse_eq_log = float(mean_squared_error(pred_deriv_log, log_true_deriv))
            traj_in_log, ok_in_log, msg_in_log = integrate_llmsr_1d([P0_train], best_sk_log, best_p_log, t_log)
            if ok_in_log:
                mse_traj_log = float(mean_squared_error(traj_in_log, P_true_log))
                r2_traj_log = float(r2_score(P_true_log, traj_in_log))
            else:
                mse_traj_log = np.nan
                r2_traj_log = np.nan
            traj_oos_log, ok_oos_log, msg_oos_log = integrate_llmsr_1d(
                [P0_oos_log], best_sk_log, best_p_log, t_oos_log
            )
            if ok_oos_log:
                r2_oos_log = float(r2_score(true_traj_oos_log, traj_oos_log))
            else:
                r2_oos_log = np.nan
            all_run_records.append({
                "Seed": seed,
                "System": "Logistic Growth",
                "Method": "LLM-SR",
                "Noise Level": NOISE_LABELS[n_idx],
                "Noise Percentage": noise,
                "Method name": f"Log Growth: LLM-SR, {NOISE_LABELS[n_idx]}",
                "Discovered Diff eq.": eq_str_log,
                "MSE of Diff eq.": mse_eq_log,
                "MSE of trajectory": mse_traj_log,
                "R^2 trajectory score": r2_traj_log,
                "Out-of-sample R^2 score": r2_oos_log,
                "Solver Status In-Sample": msg_in_log,
                "Solver Status OOS": msg_oos_log
            })
            if seed_idx == 0:
                if "llmsr_preds" not in vis_data["log"]:
                    vis_data["log"]["llmsr_preds"] = []
                    vis_data["log"]["llmsr_trajs"] = []
                vis_data["log"]["llmsr_preds"].append(pred_deriv_log)
                vis_data["log"]["llmsr_trajs"].append(
                    traj_in_log if ok_in_log else np.full_like(t_log, np.nan)
                )

        g_val = 9.8
        L_val = 3.0
        th0_train = np.pi / 2.0
        om0_train = 0.0
        t_pen, th_true_pen, om_true_pen, _, _ = generate_pendulum_data(
            theta_max=th0_train, g=g_val, L=L_val, rng=rng
        )
        dt_pen = t_pen[1] - t_pen[0]
        true_pen_ddot = -(g_val / L_val) * np.sin(th_true_pen)
        t_oos_pen = np.linspace(0.0, 10.0, 200)
        y0_oos_pen = [np.pi / 4.0, 0.0]
        def exact_pendulum_oos_ode(t, y):
            return [y[1], -(g_val / L_val) * np.sin(y[0])]
        sol_oos_pen = solve_ivp(
            exact_pendulum_oos_ode, [0.0, t_oos_pen[-1]], y0_oos_pen, t_eval=t_oos_pen, rtol=1e-10, atol=1e-10
        )
        true_traj_oos_pen = np.asarray(sol_oos_pen.y[0])

        for n_idx, noise in enumerate(NOISE_LEVELS):
            _, _, _, th_noisy_pen, om_noisy_pen = generate_pendulum_data(
                theta_max=th0_train, g=g_val, L=L_val, noise_percentage=noise, rng=rng
            )
            try:
                import pysindy as ps
                fd_2nd = ps.SmoothedFiniteDifference(smoother_kws={"window_length": 5}, d=2)
                th_dot2 = fd_2nd._differentiate(th_noisy_pen, t=dt_pen)
            except Exception:
                th_dot2 = np.gradient(np.gradient(th_noisy_pen, dt_pen), dt_pen)

            th_filtered_pen = apply_independent_wiener(th_noisy_pen, window_size=WIENER_WINDOW)
            try:
                th_dot2_f = fd_2nd._differentiate(th_filtered_pen, t=dt_pen)
            except Exception:
                th_dot2_f = np.gradient(np.gradient(th_filtered_pen, dt_pen), dt_pen)

            osc_buffer = run_llm_sr_benchmark("Oscillator", th_noisy_pen, th_dot2)
            best_sk_osc = osc_buffer["best_skeleton"]
            best_p_osc = osc_buffer["best_params"]
            eq_str_osc = format_discovered_equation("Oscillator", best_sk_osc, best_p_osc)
            pred_ddot_osc = predict_equation_derivative(best_sk_osc, best_p_osc, th_noisy_pen)
            mse_eq_osc = float(mean_squared_error(pred_ddot_osc, true_pen_ddot))
            traj_in_osc, ok_in_osc, msg_in_osc = integrate_llmsr_2nd_order(
                [th0_train, om0_train], best_sk_osc, best_p_osc, t_pen
            )
            if ok_in_osc:
                mse_traj_osc = float(mean_squared_error(traj_in_osc, th_true_pen))
                r2_traj_osc = float(r2_score(th_true_pen, traj_in_osc))
            else:
                mse_traj_osc = np.nan
                r2_traj_osc = np.nan
            traj_oos_osc, ok_oos_osc, msg_oos_osc = integrate_llmsr_2nd_order(
                y0_oos_pen, best_sk_osc, best_p_osc, t_oos_pen
            )
            if ok_oos_osc:
                r2_oos_osc = float(r2_score(true_traj_oos_pen, traj_oos_osc))
            else:
                r2_oos_osc = np.nan
            all_run_records.append({
                "Seed": seed,
                "System": "Oscillator",
                "Method": "LLM-SR",
                "Noise Level": NOISE_LABELS[n_idx],
                "Noise Percentage": noise,
                "Method name": f"Oscillator: LLM-SR, {NOISE_LABELS[n_idx]}",
                "Discovered Diff eq.": eq_str_osc,
                "MSE of Diff eq.": mse_eq_osc,
                "MSE of trajectory": mse_traj_osc,
                "R^2 trajectory score": r2_traj_osc,
                "Out-of-sample R^2 score": r2_oos_osc,
                "Solver Status In-Sample": msg_in_osc,
                "Solver Status OOS": msg_oos_osc
            })
            if seed_idx == 0:
                if "llmsr_preds" not in vis_data["osc"]:
                    vis_data["osc"]["llmsr_preds"] = []
                    vis_data["osc"]["llmsr_trajs"] = []
                vis_data["osc"]["llmsr_preds"].append(pred_ddot_osc)
                vis_data["osc"]["llmsr_trajs"].append(
                    traj_in_osc if ok_in_osc else np.full_like(t_pen, np.nan)
                )

            osc_f_buffer = run_llm_sr_benchmark("Oscillator", th_filtered_pen, th_dot2_f)
            best_sk_osc_f = osc_f_buffer["best_skeleton"]
            best_p_osc_f = osc_f_buffer["best_params"]
            eq_str_osc_f = format_discovered_equation("Oscillator", best_sk_osc_f, best_p_osc_f)
            pred_ddot_osc_f = predict_equation_derivative(best_sk_osc_f, best_p_osc_f, th_filtered_pen)
            mse_eq_osc_f = float(mean_squared_error(pred_ddot_osc_f, true_pen_ddot))
            traj_in_osc_f, ok_in_osc_f, msg_in_osc_f = integrate_llmsr_2nd_order(
                [th0_train, om0_train], best_sk_osc_f, best_p_osc_f, t_pen
            )
            if ok_in_osc_f:
                mse_traj_osc_f = float(mean_squared_error(traj_in_osc_f, th_true_pen))
                r2_traj_osc_f = float(r2_score(th_true_pen, traj_in_osc_f))
            else:
                mse_traj_osc_f = np.nan
                r2_traj_osc_f = np.nan
            traj_oos_osc_f, ok_oos_osc_f, msg_oos_osc_f = integrate_llmsr_2nd_order(
                y0_oos_pen, best_sk_osc_f, best_p_osc_f, t_oos_pen
            )
            if ok_oos_osc_f:
                r2_oos_osc_f = float(r2_score(true_traj_oos_pen, traj_oos_osc_f))
            else:
                r2_oos_osc_f = np.nan
            all_run_records.append({
                "Seed": seed,
                "System": "Oscillator",
                "Method": "LLM-SR (Filtered)",
                "Noise Level": NOISE_LABELS[n_idx],
                "Noise Percentage": noise,
                "Method name": f"Oscillator: LLM-SR, Filtered, {NOISE_LABELS[n_idx]}",
                "Discovered Diff eq.": eq_str_osc_f,
                "MSE of Diff eq.": mse_eq_osc_f,
                "MSE of trajectory": mse_traj_osc_f,
                "R^2 trajectory score": r2_traj_osc_f,
                "Out-of-sample R^2 score": r2_oos_osc_f,
                "Solver Status In-Sample": msg_in_osc_f,
                "Solver Status OOS": msg_oos_osc_f
            })
            if seed_idx == 0:
                if "llmsr_f_preds" not in vis_data["osc"]:
                    vis_data["osc"]["llmsr_f_preds"] = []
                    vis_data["osc"]["llmsr_f_trajs"] = []
                vis_data["osc"]["llmsr_f_preds"].append(pred_ddot_osc_f)
                vis_data["osc"]["llmsr_f_trajs"].append(
                    traj_in_osc_f if ok_in_osc_f else np.full_like(t_pen, np.nan)
                )

    os.makedirs("results", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    df_runs = pd.DataFrame(all_run_records)
    df_runs.to_csv("results/llmsr_detailed_runs.csv", index=False)
    print("\nSaved detailed runs to results/llmsr_detailed_runs.csv")

    summary_rows = []
    unique_methods = df_runs["Method name"].unique()

    for method_name in unique_methods:
        sub = df_runs[df_runs["Method name"] == method_name]
        rep_eq = sub.iloc[0]["Discovered Diff eq."]
        mse_eq_mean = float(sub["MSE of Diff eq."].mean())
        mse_eq_std = float(sub["MSE of Diff eq."].std(ddof=1)) if len(sub) > 1 else 0.0
        valid_mse_traj = sub["MSE of trajectory"].dropna()
        mse_traj_mean = float(valid_mse_traj.mean()) if len(valid_mse_traj) > 0 else np.nan
        mse_traj_std = float(valid_mse_traj.std(ddof=1)) if len(valid_mse_traj) > 1 else 0.0
        valid_r2_traj = sub["R^2 trajectory score"].dropna()
        r2_traj_mean = float(valid_r2_traj.mean()) if len(valid_r2_traj) > 0 else np.nan
        r2_traj_std = float(valid_r2_traj.std(ddof=1)) if len(valid_r2_traj) > 1 else 0.0
        valid_r2_oos = sub["Out-of-sample R^2 score"].dropna()
        r2_oos_mean = float(valid_r2_oos.mean()) if len(valid_r2_oos) > 0 else np.nan
        r2_oos_std = float(valid_r2_oos.std(ddof=1)) if len(valid_r2_oos) > 1 else 0.0
        in_sample_failures = sub[sub["Solver Status In-Sample"] != "Success"]
        oos_failures = sub[sub["Solver Status OOS"] != "Success"]
        if len(in_sample_failures) == 0 and len(oos_failures) == 0:
            solver_summary = "All integrations successful"
        else:
            reasons = list(in_sample_failures["Solver Status In-Sample"]) + list(oos_failures["Solver Status OOS"])
            solver_summary = f"{len(reasons)} failure(s): {', '.join(set(reasons))}"
        summary_rows.append({
            "Method name": method_name,
            "Discovered Diff eq.": rep_eq,
            "MSE of Diff eq. (mean)": mse_eq_mean,
            "MSE of Diff eq. (std)": mse_eq_std,
            "MSE of trajectory (mean)": mse_traj_mean,
            "MSE of trajectory (std)": mse_traj_std,
            "R^2 trajectory score (mean)": r2_traj_mean,
            "R^2 trajectory score (std)": r2_traj_std,
            "Out-of-sample R^2 score (mean)": r2_oos_mean,
            "Out-of-sample R^2 score (std)": r2_oos_std,
            "Solver Status": solver_summary,
            "MSE of Diff eq.": f"{mse_eq_mean:.4e} ± {mse_eq_std:.4e}",
            "MSE of trajectory": f"{mse_traj_mean:.4e} ± {mse_traj_std:.4e}" if not np.isnan(mse_traj_mean) else "Failed",
            "R^2 trajectory score": f"{r2_traj_mean:.4f} ± {r2_traj_std:.4f}" if not np.isnan(r2_traj_mean) else "NaN",
            "Out-of-sample R^2 score": f"{r2_oos_mean:.4f} ± {r2_oos_std:.4f}" if not np.isnan(r2_oos_mean) else "NaN"
        })

    final_table = pd.DataFrame(summary_rows)
    final_table.to_csv("results/llmsr_summary.csv", index=False)
    print("Saved final evaluation summary table to results/llmsr_summary.csv")

    print("\n" + "=" * 90)
    print("LLM-SR BENCHMARK DISCOVERY SUMMARY (MEAN ± STD ACROSS REALIZATIONS)")
    print("=" * 90)
    print(f"| {'Method Name':<40} | {'Discovered Diff Eq':<35} | {'MSE Diff Eq':<24} | {'Traj R^2':<18} |")
    print(f"| {':---':<40} | {':---':<35} | {':---':<24} | {':---':<18} |")
    for _, row in final_table.iterrows():
        print(f"| {row['Method name']:<40} | {row['Discovered Diff eq.']:<35} | {row['MSE of Diff eq.']:<24} | {row['R^2 trajectory score']:<18} |")
    print("=" * 90 + "\n")

    print("Generating benchmark visualizations...")

    t_exp_ref, y_true_exp_ref, _ = generate_exponential_decay(
        lambda_val=0.3, y0=10.0, rng=np.random.default_rng(SEEDS[0])
    )
    decay_true_deriv_ref = -0.3 * y_true_exp_ref

    t_log_ref, P_true_log_ref, _ = generate_logistic_growth(
        P0=100.0, r=1.1, K=1000.0, rng=np.random.default_rng(SEEDS[0])
    )
    log_true_deriv_ref = 1.1 * P_true_log_ref * (1.0 - P_true_log_ref / 1000.0)

    t_pen_ref, th_true_pen_ref, om_true_pen_ref, _, _ = generate_pendulum_data(
        theta_max=np.pi/2, g=9.8, L=3.0, rng=np.random.default_rng(SEEDS[0])
    )
    true_pen_ddot_ref = -(9.8 / 3.0) * np.sin(th_true_pen_ref)

    fig_exp, axes_exp = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for i, ax in enumerate(axes_exp.flat):
        ax.plot(t_exp_ref, decay_true_deriv_ref, linestyle="--", color="black", linewidth=2.0, label="True derivative")
        ax.plot(t_exp_ref, vis_data["exp"]["llmsr_preds"][i], linestyle="-", color="blue", linewidth=1.8, label="LLM-SR")
        ax.set_title(f"Exponential Decay: {NOISE_NAMES[i]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("dy/dt", fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig_exp.savefig("results/exp_decay_llmsr.png", dpi=300)
    plt.close(fig_exp)

    fig_log, axes_log = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for i, ax in enumerate(axes_log.flat):
        ax.plot(t_log_ref, log_true_deriv_ref, linestyle="--", color="black", linewidth=2.0, label="True derivative")
        ax.plot(t_log_ref, vis_data["log"]["llmsr_preds"][i], linestyle="-", color="blue", linewidth=1.8, label="LLM-SR")
        ax.set_title(f"Logistic Growth: {NOISE_NAMES[i]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("dP/dt", fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig_log.savefig("results/log_growth_llmsr.png", dpi=300)
    plt.close(fig_log)

    fig_osc, axes_osc = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for i, ax in enumerate(axes_osc.flat):
        ax.plot(t_pen_ref, true_pen_ddot_ref, linestyle="--", color="black", linewidth=2.0, label="True d²θ/dt²")
        ax.plot(t_pen_ref, vis_data["osc"]["llmsr_preds"][i], linestyle="-", color="purple", linewidth=1.8, label="LLM-SR Discovered")
        ax.set_title(f"Oscillator LLM-SR: {NOISE_NAMES[i]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("d²θ/dt²", fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig_osc.savefig("results/osc_llmsr.png", dpi=300)
    plt.close(fig_osc)

    fig_comp, axes_comp = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
    for i, ax in enumerate(axes_comp.flat):
        ax.plot(t_pen_ref, true_pen_ddot_ref, linestyle="--", color="black", linewidth=2.0, label="True d²θ/dt²")
        ax.plot(t_pen_ref, vis_data["osc"]["llmsr_preds"][i], linestyle="-", color="blue", linewidth=1.5, label="LLM-SR (Unfiltered)")
        ax.plot(t_pen_ref, vis_data["osc"]["llmsr_f_preds"][i], linestyle="-.", color="crimson", linewidth=1.8, label="LLM-SR (Filtered)")
        ax.set_title(f"Oscillator Acceleration: {NOISE_NAMES[i]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("d²θ/dt² (rad/s²)", fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
    plt.suptitle("Nonlinear Pendulum: LLM-SR Differential Equation Recovery Under Noise", fontsize=13, y=0.98, fontweight="bold")
    plt.tight_layout()
    fig_comp.savefig("results/osc_llmsr_comparison.png", dpi=300)
    plt.close(fig_comp)

    fig_traj, axs_traj = plt.subplots(4, 3, figsize=(18, 16))
    row_labels = [
        "Exp Decay: LLM-SR",
        "Log Growth: LLM-SR",
        "Oscillator: LLM-SR",
        "Oscillator (filtr.): LLM-SR"
    ]
    trajectories_matrix = [
        vis_data["exp"]["llmsr_trajs"],
        vis_data["log"]["llmsr_trajs"],
        vis_data["osc"]["llmsr_trajs"],
        vis_data["osc"]["llmsr_f_trajs"]
    ]
    for row_idx, (row_ax, label) in enumerate(zip(axs_traj[:, 0], row_labels)):
        row_ax.annotate(
            label,
            xy=(0, 0.5),
            xytext=(-45, 0),
            xycoords="axes fraction",
            textcoords="offset points",
            size=11,
            weight="bold",
            ha="right",
            va="center",
            rotation=90
        )
    for row in range(4):
        for col in range(3):
            ax = axs_traj[row, col]
            if row == 0:
                t_curr = t_exp_ref
                y_curr_true = y_true_exp_ref
                y_label_str = "y(t)"
            elif row == 1:
                t_curr = t_log_ref
                y_curr_true = P_true_log_ref
                y_label_str = "P(t)"
            else:
                t_curr = t_pen_ref
                y_curr_true = th_true_pen_ref
                y_label_str = "θ(t)"
            ax.plot(t_curr, y_curr_true, linestyle="--", color="black", linewidth=2.0, label="True Trajectory")
            disc_traj = trajectories_matrix[row][col]
            if disc_traj is not None and not np.all(np.isnan(disc_traj)):
                ax.plot(
                    t_curr,
                    disc_traj,
                    color="crimson" if "filtr" in row_labels[row] else "blue",
                    linewidth=1.8,
                    label="Discovered Trajectory"
                )
            else:
                ax.text(
                    0.5, 0.5, "Integration Failed",
                    horizontalalignment="center", verticalalignment="center",
                    transform=ax.transAxes, color="red", fontsize=10, weight="bold"
                )
            if row == 0:
                ax.set_title(NOISE_NAMES[col], fontsize=12, fontweight="bold")
            if row == 3:
                ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_ylabel(y_label_str, fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig_traj.savefig("results/all_trajectories_llmsr.png", dpi=300)
    plt.close(fig_traj)

    fig_all_eqs, axs_all_eqs = plt.subplots(3, 3, figsize=(16, 12))
    systems_eq_info = [
        ("Exponential Decay", t_exp_ref, decay_true_deriv_ref, vis_data["exp"]["llmsr_preds"], "dy/dt"),
        ("Logistic Growth", t_log_ref, log_true_deriv_ref, vis_data["log"]["llmsr_preds"], "dP/dt"),
        ("Oscillator", t_pen_ref, true_pen_ddot_ref, vis_data["osc"]["llmsr_preds"], "d²θ/dt²")
    ]
    for r_idx, (sys_name, t_ref_s, true_deriv_s, preds_s, y_lbl) in enumerate(systems_eq_info):
        for c_idx in range(3):
            ax = axs_all_eqs[r_idx, c_idx]
            ax.plot(t_ref_s, true_deriv_s, "--", color="black", linewidth=2.0, label="Ground Truth")
            ax.plot(t_ref_s, preds_s[c_idx], "-", color="#1f77b4" if r_idx == 0 else ("#2ca02c" if r_idx == 1 else "#9467bd"), linewidth=2.0, label="LLM-SR Discovered")
            if r_idx == 0:
                ax.set_title(NOISE_NAMES[c_idx], fontsize=12, fontweight="bold")
            ax.set_ylabel(y_lbl, fontsize=10)
            if r_idx == 2:
                ax.set_xlabel("Time (s)", fontsize=10)
            ax.text(0.03, 0.90, f"{sys_name}", transform=ax.transAxes, fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.6)
    fig_all_eqs.suptitle("Discovered Differential Equations via LLM-SR Across Noise Levels", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig_all_eqs.savefig("results/llm_sr_diff_equations.png", dpi=300)
    fig_all_eqs.savefig("outputs/llm_sr_diff_equations.png", dpi=300)
    fig_all_eqs.savefig("llm_sr_diff_equations.png", dpi=300)
    plt.close(fig_all_eqs)

    fig_all_trajs, axs_all_trajs = plt.subplots(3, 3, figsize=(16, 12))
    systems_traj_info = [
        ("Exponential Decay", t_exp_ref, y_true_exp_ref, vis_data["exp"]["llmsr_trajs"], "y(t)"),
        ("Logistic Growth", t_log_ref, P_true_log_ref, vis_data["log"]["llmsr_trajs"], "P(t)"),
        ("Oscillator", t_pen_ref, th_true_pen_ref, vis_data["osc"]["llmsr_trajs"], "θ(t) [rad]")
    ]
    for r_idx, (sys_name, t_ref_s, true_traj_s, trajs_s, y_lbl) in enumerate(systems_traj_info):
        for c_idx in range(3):
            ax = axs_all_trajs[r_idx, c_idx]
            ax.plot(t_ref_s, true_traj_s, "--", color="black", linewidth=2.0, label="Ground Truth")
            sim_traj = trajs_s[c_idx]
            if sim_traj is not None and not np.all(np.isnan(sim_traj)):
                ax.plot(t_ref_s, sim_traj, "-", color="#d62728" if r_idx == 0 else ("#ff7f0e" if r_idx == 1 else "#17becf"), linewidth=2.0, label="LLM-SR Trajectory")
            else:
                ax.text(0.5, 0.5, "Integration Failed", transform=ax.transAxes, color="red", fontsize=10, weight="bold", ha="center", va="center")
            if r_idx == 0:
                ax.set_title(NOISE_NAMES[c_idx], fontsize=12, fontweight="bold")
            ax.set_ylabel(y_lbl, fontsize=10)
            if r_idx == 2:
                ax.set_xlabel("Time (s)", fontsize=10)
            ax.text(0.03, 0.90, f"{sys_name}", transform=ax.transAxes, fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            ax.legend(loc="lower right" if r_idx != 1 else "upper left", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.6)
    fig_all_trajs.suptitle("Discovered State Trajectories via LLM-SR Across Noise Levels", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig_all_trajs.savefig("results/llm_sr_trajectories.png", dpi=300)
    fig_all_trajs.savefig("outputs/llm_sr_trajectories.png", dpi=300)
    fig_all_trajs.savefig("llm_sr_trajectories.png", dpi=300)
    plt.close(fig_all_trajs)

    print("All plots generated and saved successfully in results/, outputs/, and root directory.")
    print("LLM-SR EXPERIMENT BENCHMARK COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
