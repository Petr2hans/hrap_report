import warnings
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import wiener
from sklearn.metrics import mean_squared_error, r2_score
import pysindy as ps
from pysr import PySRRegressor

# Filter external warnings
warnings.filterwarnings("ignore")

# BENCHMARK CONFIGURATION
SEEDS = [42, 50, 100]
NOISE_LEVELS = [0.00, 0.05, 0.10]
NOISE_LABELS = ["Noise Level 1", "Noise Level 2", "Noise Level 3"]
NOISE_NAMES = ["Noise Level 1 (0%)", "Noise Level 2 (5%)", "Noise Level 3 (10%)"]
WIENER_WINDOW = 5  # Refined window: ~6% of oscillation period
PYSR_ITERATIONS = 300

def integrate_pysr_1d(y0, sym, eq, t_eval):

    symbol = sp.Symbol(sym)
    try:
        func = sp.lambdify(symbol, eq, "numpy")
    except Exception as e:
        return None, False, f"Lambdify error: {type(e).__name__} ({str(e)})"

    t_span = (t_eval[0], t_eval[-1])

    def ode_system(t, y):
        with np.errstate(all='raise'):
            val = func(y[0])
            if np.isscalar(val) or isinstance(val, (int, float, np.number)):
                return [float(val)]
            elif isinstance(val, np.ndarray):
                return [float(val.item()) if val.size == 1 else float(val[0])]
            return [float(val)]

    try:
        solution = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='RK45')
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


def integrate_pysr_2nd_order(y0, sym, eq, t_eval):

    symbol = sp.Symbol(sym)
    try:
        func = sp.lambdify(symbol, eq, "numpy")
    except Exception as e:
        return None, False, f"Lambdify error: {type(e).__name__} ({str(e)})"

    t_span = (t_eval[0], t_eval[-1])

    def ode_system(t, y):
        with np.errstate(all='raise'):
            val = func(y[0])
            if np.isscalar(val) or isinstance(val, (int, float, np.number)):
                f_val = float(val)
            elif isinstance(val, np.ndarray):
                f_val = float(val.item()) if val.size == 1 else float(val[0])
            else:
                f_val = float(val)
            return [y[1], f_val]

    try:
        solution = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='RK45')
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

def simulate_pysindy_safe(pysindy_model, x0, t_eval):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traj = pysindy_model.simulate(x0, t_eval)
        if traj is None or len(traj) < len(t_eval):
            return None, False, "Solver failure: Simulation stopped before t_end"
        traj_arr = np.asarray(traj)
        if np.any(np.isnan(traj_arr)) or np.any(np.isinf(traj_arr)):
            return None, False, "Numerical instability: NaN or Inf in simulated trajectory"
        return traj_arr, True, "Success"
    except OverflowError:
        return None, False, "Overflow: state blew up during PySINDy simulation"
    except FloatingPointError as e:
        return None, False, f"Numerical instability: FloatingPointError ({str(e)})"
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

    # Inject comparable Gaussian noise scaled by standard deviation of each state
    noise_th = rng.normal(loc=0.0, scale=1.0, size=num_points)
    noise_omega = rng.normal(loc=0.0, scale=1.0, size=num_points)

    th_noisy = th_true + noise_th * noise_percentage * np.std(th_true)
    omega_noisy = omega_true + noise_omega * noise_percentage * np.std(omega_true)

    return t_eval, th_true, omega_true, th_noisy, omega_noisy

print(f"Starting Multi-Realization ODE Discovery Benchmark across {len(SEEDS)} seeds...")

all_run_records = []

vis_data = {
    'exp': {},
    'log': {},
    'osc': {},
    'trajectories': {}
}

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n>>> Running Realization {seed_idx + 1}/{len(SEEDS)} (Seed: {seed}) <<<")
    rng = np.random.default_rng(seed)

    # SYSTEM 1: dy/dt = -0.3 * y
    lambda_val = 0.3
    y0_train = 10.0
    t_exp, y_true_exp, _ = generate_exponential_decay(lambda_val=lambda_val, y0=y0_train, rng=rng)
    dt_exp = t_exp[1] - t_exp[0]
    decay_true_deriv = -lambda_val * y_true_exp

    # Set up OOS grid
    t_oos_exp = np.linspace(0.0, 10.0, 200)
    y0_oos_exp = 20.0
    true_traj_oos_exp = y0_oos_exp * np.exp(-lambda_val * t_oos_exp)

    for n_idx, noise in enumerate(NOISE_LEVELS):
        _, _, y_noisy = generate_exponential_decay(lambda_val=lambda_val, y0=y0_train, noise_percentage=noise, rng=rng)
        target_deriv = np.gradient(y_noisy, dt_exp)
        data_in = y_noisy.reshape(-1, 1)

        # 1A. PySR Exponential Decay
        model_sr = PySRRegressor(
            random_state=seed,
            deterministic=True,
            parallelism='serial',
            niterations=PYSR_ITERATIONS,
            binary_operators=["*", "/", "+", "-"],
            unary_operators=["exp"],
            model_selection="best",
            progress=False,
            verbosity=0
        )
        model_sr.fit(data_in, target_deriv, variable_names=["pt"])
        sr_pred = np.asarray(model_sr.predict(data_in)).flatten()
        mse_eq = mean_squared_error(sr_pred, decay_true_deriv)

        # In-sample trajectory
        traj_sr, ok_sr, msg_sr = integrate_pysr_1d([y0_train], "pt", model_sr.sympy(), t_exp)
        if ok_sr:
            mse_traj = mean_squared_error(traj_sr, y_true_exp)
            r2_traj = r2_score(y_true_exp, traj_sr)
        else:
            mse_traj = np.nan
            r2_traj = np.nan

        # Out-of-sample trajectory
        traj_oos_sr, ok_oos_sr, msg_oos_sr = integrate_pysr_1d([y0_oos_exp], "pt", model_sr.sympy(), t_oos_exp)
        if ok_oos_sr:
            r2_oos = r2_score(true_traj_oos_exp, traj_oos_sr)
        else:
            r2_oos = np.nan

        all_run_records.append({
            'Seed': seed,
            'System': 'Exponential Decay',
            'Method': 'PySR',
            'Noise Level': NOISE_LABELS[n_idx],
            'Noise Percentage': noise,
            'Method name': f"Exp. Decay: PySR, {NOISE_LABELS[n_idx]}",
            'Discovered Diff eq.': str(model_sr.sympy()),
            'MSE of Diff eq.': mse_eq,
            'MSE of trajectory': mse_traj,
            'R^2 trajectory score': r2_traj,
            'Out-of-sample R^2 score': r2_oos,
            'Solver Status In-Sample': msg_sr,
            'Solver Status OOS': msg_oos_sr
        })
        if seed_idx == 0:
            if 'pysr_preds' not in vis_data['exp']:
                vis_data['exp']['pysr_preds'] = []
                vis_data['exp']['pysr_trajs'] = []
            vis_data['exp']['pysr_preds'].append(sr_pred)
            vis_data['exp']['pysr_trajs'].append(traj_sr if ok_sr else np.full_like(t_exp, np.nan))

        #PySINDy Exponential Decay
        opt_si = ps.STLSQ(threshold=0.1)
        model_si = ps.SINDy(differentiation_method=ps.FiniteDifference(), optimizer=opt_si, feature_library=ps.PolynomialLibrary(degree=1))
        model_si.fit(y_noisy, t=dt_exp, feature_names=["nt"])
        si_pred = np.asarray(model_si.predict(y_noisy)).flatten()
        mse_eq_si = mean_squared_error(si_pred, decay_true_deriv)

        # In-sample simulation
        traj_si, ok_si, msg_si = simulate_pysindy_safe(model_si, [y_noisy[0]], t_exp)
        if ok_si:
            traj_si_1d = np.asarray(traj_si).flatten()
            mse_traj_si = mean_squared_error(traj_si_1d, y_true_exp)
            r2_traj_si = r2_score(y_true_exp, traj_si_1d)
        else:
            mse_traj_si = np.nan
            r2_traj_si = np.nan
            traj_si_1d = np.full_like(t_exp, np.nan)

        # Out-of-sample simulation
        traj_oos_si, ok_oos_si, msg_oos_si = simulate_pysindy_safe(model_si, [y0_oos_exp], t_oos_exp)
        if ok_oos_si:
            r2_oos_si = r2_score(true_traj_oos_exp, np.asarray(traj_oos_si).flatten())
        else:
            r2_oos_si = np.nan

        all_run_records.append({
            'Seed': seed,
            'System': 'Exponential Decay',
            'Method': 'PySINDy',
            'Noise Level': NOISE_LABELS[n_idx],
            'Noise Percentage': noise,
            'Method name': f"Exp. Decay: PySINDy, {NOISE_LABELS[n_idx]}",
            'Discovered Diff eq.': str(model_si.equations()),
            'MSE of Diff eq.': mse_eq_si,
            'MSE of trajectory': mse_traj_si,
            'R^2 trajectory score': r2_traj_si,
            'Out-of-sample R^2 score': r2_oos_si,
            'Solver Status In-Sample': msg_si,
            'Solver Status OOS': msg_oos_si
        })
        if seed_idx == 0:
            if 'pysindy_preds' not in vis_data['exp']:
                vis_data['exp']['pysindy_preds'] = []
                vis_data['exp']['pysindy_trajs'] = []
            vis_data['exp']['pysindy_preds'].append(si_pred)
            vis_data['exp']['pysindy_trajs'].append(traj_si_1d)

    # SYSTEM 2: dP/dt = 1.1 * P * (1 - P/1000))
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
        _, _, p_noisy = generate_logistic_growth(P0=P0_train, r=r_val, K=K_val, noise_percentage=noise, rng=rng)
        target_deriv_log = np.gradient(p_noisy, dt_log)
        data_in_log = p_noisy.reshape(-1, 1)
        #PySR Logistic Growth
        model_sr_log = PySRRegressor(
            parallelism='serial',
            deterministic=True,
            random_state=seed,
            niterations=PYSR_ITERATIONS,
            binary_operators=["*", "/", "+", "-"],
            unary_operators=["exp"],
            model_selection="best",
            progress=False,
            verbosity=0
        )
        model_sr_log.fit(data_in_log, target_deriv_log, variable_names=["pt"])
        sr_pred_log = np.asarray(model_sr_log.predict(data_in_log)).flatten()
        mse_eq_sr_log = mean_squared_error(sr_pred_log, log_true_deriv)

        # In-sample trajectory
        traj_sr_log, ok_sr_log, msg_sr_log = integrate_pysr_1d([P0_train], "pt", model_sr_log.sympy(), t_log)
        if ok_sr_log:
            mse_traj_sr_log = mean_squared_error(traj_sr_log, P_true_log)
            r2_traj_sr_log = r2_score(P_true_log, traj_sr_log)
        else:
            mse_traj_sr_log = np.nan
            r2_traj_sr_log = np.nan

        # Out-of-sample trajectory
        traj_oos_sr_log, ok_oos_sr_log, msg_oos_sr_log = integrate_pysr_1d([P0_oos_log], "pt", model_sr_log.sympy(), t_oos_log)
        if ok_oos_sr_log:
            r2_oos_sr_log = r2_score(true_traj_oos_log, traj_oos_sr_log)
        else:
            r2_oos_sr_log = np.nan

        all_run_records.append({
            'Seed': seed,
            'System': 'Logistic Growth',
            'Method': 'PySR',
            'Noise Level': NOISE_LABELS[n_idx],
            'Noise Percentage': noise,
            'Method name': f"Log Growth: PySR, {NOISE_LABELS[n_idx]}",
            'Discovered Diff eq.': str(model_sr_log.sympy()),
            'MSE of Diff eq.': mse_eq_sr_log,
            'MSE of trajectory': mse_traj_sr_log,
            'R^2 trajectory score': r2_traj_sr_log,
            'Out-of-sample R^2 score': r2_oos_sr_log,
            'Solver Status In-Sample': msg_sr_log,
            'Solver Status OOS': msg_oos_sr_log
        })

        if seed_idx == 0:
            if 'pysr_preds' not in vis_data['log']:
                vis_data['log']['pysr_preds'] = []
                vis_data['log']['pysr_trajs'] = []
            vis_data['log']['pysr_preds'].append(sr_pred_log)
            vis_data['log']['pysr_trajs'].append(traj_sr_log if ok_sr_log else np.full_like(t_log, np.nan))

        #PySINDy Logistic Growth
        opt_si_log = ps.STLSQ(threshold=0.0001)
        lib_log = ps.PolynomialLibrary(degree=3)
        model_si_log = ps.SINDy(differentiation_method=ps.FiniteDifference(), optimizer=opt_si_log, feature_library=lib_log)
        model_si_log.fit(p_noisy, t=dt_log, feature_names=["pt"])
        si_pred_log = np.asarray(model_si_log.predict(p_noisy)).flatten()
        mse_eq_si_log = mean_squared_error(si_pred_log, log_true_deriv)

        # In-sample simulation
        traj_si_log, ok_si_log, msg_si_log = simulate_pysindy_safe(model_si_log, [p_noisy[0]], t_log)
        if ok_si_log:
            traj_si_log_1d = np.asarray(traj_si_log).flatten()
            mse_traj_si_log = mean_squared_error(traj_si_log_1d, P_true_log)
            r2_traj_si_log = r2_score(P_true_log, traj_si_log_1d)
        else:
            mse_traj_si_log = np.nan
            r2_traj_si_log = np.nan
            traj_si_log_1d = np.full_like(t_log, np.nan)

        # Out-of-sample simulation
        traj_oos_si_log, ok_oos_si_log, msg_oos_si_log = simulate_pysindy_safe(model_si_log, [P0_oos_log], t_oos_log)
        if ok_oos_si_log:
            r2_oos_si_log = r2_score(true_traj_oos_log, np.asarray(traj_oos_si_log).flatten())
        else:
            r2_oos_si_log = np.nan

        all_run_records.append({
            'Seed': seed,
            'System': 'Logistic Growth',
            'Method': 'PySINDy',
            'Noise Level': NOISE_LABELS[n_idx],
            'Noise Percentage': noise,
            'Method name': f"Log Growth: PySINDy, {NOISE_LABELS[n_idx]}",
            'Discovered Diff eq.': str(model_si_log.equations()),
            'MSE of Diff eq.': mse_eq_si_log,
            'MSE of trajectory': mse_traj_si_log,
            'R^2 trajectory score': r2_traj_si_log,
            'Out-of-sample R^2 score': r2_oos_si_log,
            'Solver Status In-Sample': msg_si_log,
            'Solver Status OOS': msg_oos_si_log
        })

        if seed_idx == 0:
            if 'pysindy_preds' not in vis_data['log']:
                vis_data['log']['pysindy_preds'] = []
                vis_data['log']['pysindy_trajs'] = []
            vis_data['log']['pysindy_preds'].append(si_pred_log)
            vis_data['log']['pysindy_trajs'].append(traj_si_log_1d)

    # SYSTEM 3: d(theta)/dt = omega, d(omega)/dt = -(g/L)*sin(theta)
    g_val = 9.8
    L_val = 3.0
    th0_train = np.pi / 2
    om0_train = 0.0
    t_pen, th_true_pen, om_true_pen, _, _ = generate_pendulum_data(theta_max=th0_train, g=g_val, L=L_val, rng=rng)
    dt_pen = t_pen[1] - t_pen[0]

    # Exact 2-state derivatives
    pen_dtheta_true = om_true_pen
    pen_domega_true = -(g_val / L_val) * np.sin(th_true_pen)

    t_oos_pen = np.linspace(0.0, 10.0, 200)
    y0_oos_pen = [np.pi / 4, 0.0]

    def exact_pendulum_oos_ode(t, y):
        return [y[1], -(g_val / L_val) * np.sin(y[0])]

    sol_oos_pen = solve_ivp(exact_pendulum_oos_ode, [0.0, t_oos_pen[-1]], y0_oos_pen, t_eval=t_oos_pen, rtol=1e-10, atol=1e-10)
    true_traj_oos_pen = np.asarray(sol_oos_pen.y[0])

    # UPDATED PySINDy Library
    pen_library_functions = [lambda x: x, lambda x: np.sin(x)]
    pen_library_function_names = [lambda x: x, lambda x: f"sin({x})"]
    pen_lib = ps.CustomLibrary(library_functions=pen_library_functions, function_names=pen_library_function_names)
    pen_optimizer = ps.STLSQ(threshold=0.2)
    pen_diff_method = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 5})

    for n_idx, noise in enumerate(NOISE_LEVELS):
        _, _, _, th_noisy, om_noisy = generate_pendulum_data(
            theta_max=th0_train, g=g_val, L=L_val, noise_percentage=noise, rng=rng
        )
        X_noisy = np.stack([th_noisy, om_noisy], axis=-1)

        # WIener filter applied independently to each state variable with reduced window width
        X_filtered = apply_independent_wiener(X_noisy, window_size=WIENER_WINDOW)

        #PySINDy (Unfiltered)
        model_si_pen = ps.SINDy(
            optimizer=pen_optimizer,
            feature_library=pen_lib,
            differentiation_method=pen_diff_method
        )
        model_si_pen.fit(X_noisy, t=dt_pen, feature_names=["theta", "omega"])
        si_pen_pred = np.asarray(model_si_pen.predict(X_noisy))

        # Derivatives MSE across both equations
        mse_eq_si_pen = 0.5 * (mean_squared_error(si_pen_pred[:, 0], pen_dtheta_true) +
                               mean_squared_error(si_pen_pred[:, 1], pen_domega_true))

        # In-sample simulation
        x0_noisy = [th_noisy[0], om_noisy[0]]
        traj_si_pen, ok_si_pen, msg_si_pen = simulate_pysindy_safe(model_si_pen, x0_noisy, t_pen)
        if ok_si_pen:
            mse_traj_si_pen = mean_squared_error(traj_si_pen[:, 0], th_true_pen)
            r2_traj_si_pen = r2_score(th_true_pen, traj_si_pen[:, 0])
            pen_traj_vis = traj_si_pen[:, 0]
        else:
            mse_traj_si_pen = np.nan
            r2_traj_si_pen = np.nan
            pen_traj_vis = np.full_like(t_pen, np.nan)

        # Out-of-sample simulation
        traj_oos_si_pen, ok_oos_si_pen, msg_oos_si_pen = simulate_pysindy_safe(model_si_pen, y0_oos_pen, t_oos_pen)
        if ok_oos_si_pen:
            r2_oos_si_pen = r2_score(true_traj_oos_pen, traj_oos_si_pen[:, 0])
        else:
            r2_oos_si_pen = np.nan

        all_run_records.append({
            'Seed': seed,
            'System': 'Oscillator',
            'Method': 'PySINDy',
            'Noise Level': NOISE_LABELS[n_idx],
            'Noise Percentage': noise,
            'Method name': f"Oscillator: PySINDy, {NOISE_LABELS[n_idx]}",
            'Discovered Diff eq.': str(model_si_pen.equations()),
            'MSE of Diff eq.': mse_eq_si_pen,
            'MSE of trajectory': mse_traj_si_pen,
            'R^2 trajectory score': r2_traj_si_pen,
            'Out-of-sample R^2 score': r2_oos_si_pen,
            'Solver Status In-Sample': msg_si_pen,
            'Solver Status OOS': msg_oos_si_pen
        })

        if seed_idx == 0:
            if 'pysindy_preds' not in vis_data['osc']:
                vis_data['osc']['pysindy_preds'] = []
                vis_data['osc']['pysindy_trajs'] = []
            vis_data['osc']['pysindy_preds'].append(si_pen_pred)
            vis_data['osc']['pysindy_trajs'].append(pen_traj_vis)

        #PySINDy (Filtered with Independent Wiener Filter)
        model_si_pen_f = ps.SINDy(
            optimizer=pen_optimizer,
            feature_library=pen_lib,
            differentiation_method=pen_diff_method
        )
        model_si_pen_f.fit(X_filtered, t=dt_pen, feature_names=["theta", "omega"])
        si_pen_f_pred = np.asarray(model_si_pen_f.predict(X_filtered))

        mse_eq_si_pen_f = 0.5 * (mean_squared_error(si_pen_f_pred[:, 0], pen_dtheta_true) +
                                 mean_squared_error(si_pen_f_pred[:, 1], pen_domega_true))

        # In-sample simulation
        traj_si_pen_f, ok_si_pen_f, msg_si_pen_f = simulate_pysindy_safe(model_si_pen_f, x0_noisy, t_pen)
        if ok_si_pen_f:
            mse_traj_si_pen_f = mean_squared_error(traj_si_pen_f[:, 0], th_true_pen)
            r2_traj_si_pen_f = r2_score(th_true_pen, traj_si_pen_f[:, 0])
            pen_f_traj_vis = traj_si_pen_f[:, 0]
        else:
            mse_traj_si_pen_f = np.nan
            r2_traj_si_pen_f = np.nan
            pen_f_traj_vis = np.full_like(t_pen, np.nan)

        # Out-of-sample simulation (on dedicated t_oos_pen grid)
        traj_oos_si_pen_f, ok_oos_si_pen_f, msg_oos_si_pen_f = simulate_pysindy_safe(model_si_pen_f, y0_oos_pen, t_oos_pen)
        if ok_oos_si_pen_f:
            r2_oos_si_pen_f = r2_score(true_traj_oos_pen, traj_oos_si_pen_f[:, 0])
        else:
            r2_oos_si_pen_f = np.nan

        all_run_records.append({
            'Seed': seed,
            'System': 'Oscillator',
            'Method': 'PySINDy (Filtered)',
            'Noise Level': NOISE_LABELS[n_idx],
            'Noise Percentage': noise,
            'Method name': f"Oscillator: PySINDy, Filtered, {NOISE_LABELS[n_idx]}",
            'Discovered Diff eq.': str(model_si_pen_f.equations()),
            'MSE of Diff eq.': mse_eq_si_pen_f,
            'MSE of trajectory': mse_traj_si_pen_f,
            'R^2 trajectory score': r2_traj_si_pen_f,
            'Out-of-sample R^2 score': r2_oos_si_pen_f,
            'Solver Status In-Sample': msg_si_pen_f,
            'Solver Status OOS': msg_oos_si_pen_f
        })

        if seed_idx == 0:
            if 'pysindy_f_preds' not in vis_data['osc']:
                vis_data['osc']['pysindy_f_preds'] = []
                vis_data['osc']['pysindy_f_trajs'] = []
            vis_data['osc']['pysindy_f_preds'].append(si_pen_f_pred)
            vis_data['osc']['pysindy_f_trajs'].append(pen_f_traj_vis)

        #PySR Oscillator (2nd Order single variable formulation d2(theta)/dt2 = f(theta))
        fd_2nd = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 5}, d=2)
        th_dot2 = fd_2nd._differentiate(th_noisy, t=dt_pen)

        model_sr_pen = PySRRegressor(
            niterations=PYSR_ITERATIONS,
            deterministic=True,
            random_state=seed,
            parallelism='serial',
            binary_operators=["-", "+", "*", "/"],
            unary_operators=["sin", "exp", "cos", "sqrt"],
            model_selection="best",
            progress=False,
            verbosity=0
        )
        model_sr_pen.fit(th_noisy.reshape(-1, 1), th_dot2, variable_names=["th_t"])
        sr_pen_pred = np.asarray(model_sr_pen.predict(th_noisy.reshape(-1, 1))).flatten()
        mse_eq_sr_pen = mean_squared_error(sr_pen_pred, pen_domega_true)

        # In-sample trajectory
        traj_sr_pen, ok_sr_pen, msg_sr_pen = integrate_pysr_2nd_order([th0_train, om0_train], "th_t", model_sr_pen.sympy(), t_pen)
        if ok_sr_pen:
            mse_traj_sr_pen = mean_squared_error(traj_sr_pen, th_true_pen)
            r2_traj_sr_pen = r2_score(th_true_pen, traj_sr_pen)
            pen_sr_traj_vis = traj_sr_pen
        else:
            mse_traj_sr_pen = np.nan
            r2_traj_sr_pen = np.nan
            pen_sr_traj_vis = np.full_like(t_pen, np.nan)

        # Out-of-sample trajectory
        traj_oos_sr_pen, ok_oos_sr_pen, msg_oos_sr_pen = integrate_pysr_2nd_order(y0_oos_pen, "th_t", model_sr_pen.sympy(), t_oos_pen)
        if ok_oos_sr_pen:
            r2_oos_sr_pen = r2_score(true_traj_oos_pen, traj_oos_sr_pen)
        else:
            r2_oos_sr_pen = np.nan

        all_run_records.append({
            'Seed': seed,
            'System': 'Oscillator',
            'Method': 'PySR',
            'Noise Level': NOISE_LABELS[n_idx],
            'Noise Percentage': noise,
            'Method name': f"Oscillator: PySR, {NOISE_LABELS[n_idx]}",
            'Discovered Diff eq.': str(model_sr_pen.sympy()),
            'MSE of Diff eq.': mse_eq_sr_pen,
            'MSE of trajectory': mse_traj_sr_pen,
            'R^2 trajectory score': r2_traj_sr_pen,
            'Out-of-sample R^2 score': r2_oos_sr_pen,
            'Solver Status In-Sample': msg_sr_pen,
            'Solver Status OOS': msg_oos_sr_pen
        })

        if seed_idx == 0:
            if 'pysr_preds' not in vis_data['osc']:
                vis_data['osc']['pysr_preds'] = []
                vis_data['osc']['pysr_trajs'] = []
            vis_data['osc']['pysr_preds'].append(sr_pen_pred)
            vis_data['osc']['pysr_trajs'].append(pen_sr_traj_vis)


#Summary
df_runs = pd.DataFrame(all_run_records)
df_runs.to_csv("results/detailed_runs.csv", index=False)
print("\nSaved detailed runs to results/detailed_runs.csv")

# Compute Mean and Standard Deviation across realizations
summary_rows = []
unique_methods = df_runs['Method name'].unique()

for method_name in unique_methods:
    sub = df_runs[df_runs['Method name'] == method_name]
    rep_eq = sub.iloc[0]['Discovered Diff eq.']
    
    # Statistical aggregations
    mse_eq_mean = sub['MSE of Diff eq.'].mean()
    mse_eq_std = sub['MSE of Diff eq.'].std(ddof=1) if len(sub) > 1 else 0.0

    valid_mse_traj = sub['MSE of trajectory'].dropna()
    mse_traj_mean = valid_mse_traj.mean() if len(valid_mse_traj) > 0 else np.nan
    mse_traj_std = valid_mse_traj.std(ddof=1) if len(valid_mse_traj) > 1 else 0.0

    valid_r2_traj = sub['R^2 trajectory score'].dropna()
    r2_traj_mean = valid_r2_traj.mean() if len(valid_r2_traj) > 0 else np.nan
    r2_traj_std = valid_r2_traj.std(ddof=1) if len(valid_r2_traj) > 1 else 0.0

    valid_r2_oos = sub['Out-of-sample R^2 score'].dropna()
    r2_oos_mean = valid_r2_oos.mean() if len(valid_r2_oos) > 0 else np.nan
    r2_oos_std = valid_r2_oos.std(ddof=1) if len(valid_r2_oos) > 1 else 0.0

    # Solvers status summary
    in_sample_failures = sub[sub['Solver Status In-Sample'] != 'Success']
    oos_failures = sub[sub['Solver Status OOS'] != 'Success']

    if len(in_sample_failures) == 0 and len(oos_failures) == 0:
        solver_summary = "All integrations successful"
    else:
        reasons = list(in_sample_failures['Solver Status In-Sample']) + list(oos_failures['Solver Status OOS'])
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
final_table.to_csv("results/summary.csv", index=False)
print("Saved final evaluation summary table to results/summary.csv")

#Plotting
print("\nGenerating updated benchmark visualizations...")

# Reference curves for plotting from Seed 1
t_exp, y_true_exp, _ = generate_exponential_decay(lambda_val=0.3, y0=10.0, rng=np.random.default_rng(SEEDS[0]))
decay_true_deriv = -0.3 * y_true_exp

t_log, P_true_log, _ = generate_logistic_growth(P0=100.0, r=1.1, K=1000.0, rng=np.random.default_rng(SEEDS[0]))
log_true_deriv = 1.1 * P_true_log * (1.0 - P_true_log / 1000.0)

t_pen, th_true_pen, om_true_pen, _, _ = generate_pendulum_data(theta_max=np.pi/2, g=9.8, L=3.0, rng=np.random.default_rng(SEEDS[0]))
true_pen_ddot = -(9.8 / 3.0) * np.sin(th_true_pen)
true_pen_dtheta = om_true_pen

# Plot 1: Exponential Decay (PySR vs PySINDy)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
for i, ax in enumerate(axes.flat):
    ax.plot(t_exp, decay_true_deriv, linestyle="--", color="black", label="True derivative")
    ax.plot(t_exp, vis_data['exp']['pysr_preds'][i], linestyle="dotted", color="blue", label="PySR")
    ax.plot(t_exp, vis_data['exp']['pysindy_preds'][i], linestyle="-", color="green", label="PySINDy")
    ax.set_title(f"Exponential Decay: {NOISE_NAMES[i]}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dy/dt")
    ax.legend()
plt.tight_layout()
fig.savefig("results/exp_decay_pysr_pysindy.png", dpi=300)
plt.close(fig)

# Plot 2: Logistic Growth (PySR vs PySINDy)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
for i, ax in enumerate(axes.flat):
    ax.plot(t_log, log_true_deriv, linestyle="--", color="black", label="True derivative")
    ax.plot(t_log, vis_data['log']['pysr_preds'][i], linestyle="dotted", color="blue", label="PySR")
    ax.plot(t_log, vis_data['log']['pysindy_preds'][i], linestyle="-", color="green", label="PySINDy")
    ax.set_title(f"Logistic Growth: {NOISE_NAMES[i]}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dP/dt")
    ax.legend()
plt.tight_layout()
fig.savefig("results/log_growth_pysr_pysindy.png", dpi=300)
plt.close(fig)

# Plot 3: Oscillator PySR (2nd Order)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
for i, ax in enumerate(axes.flat):
    ax.plot(t_pen, true_pen_ddot, linestyle="--", color="black", label="True d2(theta)/dt2")
    ax.plot(t_pen, vis_data['osc']['pysr_preds'][i], linestyle="-", color="purple", label="PySR Discovered")
    ax.set_title(f"Oscillator PySR: {NOISE_NAMES[i]}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("d2(theta)/dt2")
    ax.legend()
plt.tight_layout()
fig.savefig("results/osc_pysr.png", dpi=300)
plt.close(fig)

# Plot 4:  ExpandedOscillator PySINDy (2x3 Grid Displaying BOTH Equations of the Two-State System)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 9))

# Row 1: d(theta)/dt = omega
for i in range(3):
    ax = axes[0, i]
    ax.plot(t_pen, true_pen_dtheta, linestyle="--", color="black", linewidth=2, label="True dθ/dt (ω)")
    ax.plot(t_pen, vis_data['osc']['pysindy_preds'][i][:, 0], linestyle="-", color="blue", label="PySINDy (Unfiltered)")
    ax.plot(t_pen, vis_data['osc']['pysindy_f_preds'][i][:, 0], linestyle="-.", color="crimson", label="PySINDy (Filtered)")
    ax.set_title(f"State 1: dθ/dt | {NOISE_NAMES[i]}", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dθ/dt (rad/s)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Row 2: d(omega)/dt = -(g/L)*sin(theta)
for i in range(3):
    ax = axes[1, i]
    ax.plot(t_pen, true_pen_ddot, linestyle="--", color="black", linewidth=2, label="True dω/dt (-g/L sinθ)")
    ax.plot(t_pen, vis_data['osc']['pysindy_preds'][i][:, 1], linestyle="-", color="blue", label="PySINDy (Unfiltered)")
    ax.plot(t_pen, vis_data['osc']['pysindy_f_preds'][i][:, 1], linestyle="-.", color="crimson", label="PySINDy (Filtered)")
    ax.set_title(f"State 2: dω/dt | {NOISE_NAMES[i]}", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dω/dt (rad/s²)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle("Nonlinear Pendulum: 2-State PySINDy Equation Recovery (Both dθ/dt and dω/dt)", fontsize=14, y=0.98)
plt.tight_layout()
fig.savefig("results/osc_pysindy_pysindyfiltered.png", dpi=300)
plt.close(fig)

# Plot 5: All Trajectories (7 rows x 3 columns)
fig, axs = plt.subplots(7, 3, figsize=(18, 22))
row_labels = [
    'Exp Decay: PySR',
    'Exp Decay: PySINDy',
    'Log Growth: PySR',
    'Log Growth: PySINDy',
    'Oscillator: PySINDy',
    'Oscillator (filtr.): PySINDy',
    'Oscillator: PySR'
]

# Assemble trajectories list for primary seed
trajectories_matrix = [
    vis_data['exp']['pysr_trajs'],
    vis_data['exp']['pysindy_trajs'],
    vis_data['log']['pysr_trajs'],
    vis_data['log']['pysindy_trajs'],
    vis_data['osc']['pysindy_trajs'],
    vis_data['osc']['pysindy_f_trajs'],
    vis_data['osc']['pysr_trajs']
]

for row_idx, (row_ax, label) in enumerate(zip(axs[:, 0], row_labels)):
    row_ax.annotate(
        label,
        xy=(0, 0.5),
        xytext=(-40, 0),
        xycoords='axes fraction',
        textcoords='offset points',
        size=11,
        weight='bold',
        ha='right',
        va='center',
        rotation=90
    )

for row in range(7):
    for col in range(3):
        ax = axs[row, col]
        if row in [0, 1]:  # Exponential Decay
            t_curr = t_exp
            y_curr_true = y_true_exp
            y_label_str = "y(t)"
        elif row in [2, 3]:  # Logistic Growth
            t_curr = t_log
            y_curr_true = P_true_log
            y_label_str = "P(t)"
        else:  # Oscillator
            t_curr = t_pen
            y_curr_true = th_true_pen
            y_label_str = "θ(t)"

        ax.plot(t_curr, y_curr_true, linestyle="--", color="black", label="True Trajectory")
        disc_traj = trajectories_matrix[row][col]
        if disc_traj is not None and not np.all(np.isnan(disc_traj)):
            ax.plot(t_curr, disc_traj, color="crimson" if "filtr" in row_labels[row] else "blue", label="Discovered Trajectory")
        else:
            ax.text(0.5, 0.5, "Integration Failed", horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='red', fontsize=10, weight='bold')

        if row == 0:
            ax.set_title(NOISE_NAMES[col], fontsize=12)
        if row == 6:
            ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel(y_label_str, fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.25)

plt.tight_layout()
fig.savefig("results/all_trajectories.png", dpi=300)
plt.close(fig)

print("All plots generated and saved successfully in results/ directory.")
print("BENCHMARK COMPLETED SUCCESSFULLY.")
