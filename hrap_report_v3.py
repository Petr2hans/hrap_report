from pysr import *
import pysindy as ps
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
import sympy as sp
from sklearn.metrics import mean_squared_error, r2_score
from scipy.integrate import solve_ivp
from scipy.signal import wiener
warnings.filterwarnings("ignore") #Ignoring PySINDy warnings

#Setting random seeds for NumPy and PySR (Review 2)
np_rand_seed = 50
rng=np.random.default_rng(seed = np_rand_seed)
pysr_rand_seed = 50

def integrate_pysr_eq(y0, sym, eq, t_eval):
    symbol = sp.Symbol(sym)
    func = sp.lambdify(symbol, eq, "numpy")
    t_span = (t_eval[0], t_eval[-1])
    def ode_system(t, y):
        return func(y[0])
    solution = solve_ivp(ode_system, t_span, y0, t_eval=t_eval)
    return solution.y[0]


def integrate_pysr_eq_2nd_order(y0, sym, eq, t_eval):
    # y0 must contain [initial_position, initial_velocity]
    symbol = sp.Symbol(sym)
    func = sp.lambdify(symbol, eq, "numpy")
    t_span = (t_eval[0], t_eval[-1])

    def ode_system(t, y):
        return [y[1], func(y[0])]

    solution = solve_ivp(ode_system, t_span, y0, t_eval=t_eval)
    return solution.y[0]

final_dict = {
    "Method name" : [],
    "Discovered Diff eq." : [],
    "MSE of Diff eq." : [],
    "MSE of trajectory" : [],
    "R^2 trajectory score": []
}
final_table = pd.DataFrame(final_dict)
trajectories_list = []

def generate_exponential_decay(lambda_val, y0=10.0, t_max=10.0, num_points=200, noise_percentage=0.0):
    t = np.linspace(0, t_max, num_points)
    y_true = y0 * np.exp(-lambda_val * t)
    true_std = np.std(y_true)
    noise = rng.normal(loc=0.0, scale=1.0, size= num_points)
    y_noisy = y_true + noise*noise_percentage*true_std
    df = pd.DataFrame({
        'time': t,
        'y_true': y_true,
        'y_noisy': y_noisy
    })
    return df
df_low = generate_exponential_decay(lambda_val = 0.3, noise_percentage=0.01)
df_moderate = generate_exponential_decay(lambda_val = 0.3, noise_percentage=0.05)
df_noisy = generate_exponential_decay(lambda_val = 0.3, noise_percentage=0.1)

t=df_low["time"].to_numpy()
y_l=df_low["y_noisy"].to_numpy()
y_m = df_moderate["y_noisy"].to_numpy()
y_n = df_noisy["y_noisy"].to_numpy()
data = [y_l.reshape(-1,1), y_m.reshape(-1,1), y_n.reshape(-1,1)]
dt = t[1] - t[0] #calculating time increments
decay_true = np.gradient(df_low["y_true"].to_numpy(), dt)

target = [np.gradient(y_l, dt), np.gradient(y_m, dt), np.gradient(y_n, dt)] #Review 2: fixed
results = []
model = PySRRegressor(
    random_state = pysr_rand_seed, #initial random seed
    deterministic= True,
    parallelism = 'serial', #required to avoid randomness in parallel evolutions
    niterations =1000,
    binary_operators=["*", "/", "^"],
    unary_operators=["exp"],
    model_selection="best"
)
for i in range (len(target)):
    model.fit(data[i], target[i], variable_names=["pt"])
    results.append(model.predict(data[i]))
    loss = mean_squared_error(results[i], decay_true)
    trajectory = integrate_pysr_eq([10], "pt", model.sympy(), t)
    trajectories_list.append(trajectory)
    trajectory_loss = mean_squared_error(trajectory, df_low["y_true"].to_numpy())
    r2 = r2_score(df_low["y_true"].to_numpy(), trajectory)
    new_row = pd.DataFrame([{
        "Method name": f"Exp. Decay: PySR, Noise Level {i+1}",
        "Discovered Diff eq.": model.sympy(),
        "MSE of Diff eq.": loss,
        "MSE of trajectory": trajectory_loss,
        "R^2 trajectory score": r2
    }])
    final_table = pd.concat([final_table, new_row], ignore_index=True)

optimizer = ps.STLSQ(threshold= 0.1)
pysindy_model = ps.SINDy(differentiation_method=ps.FiniteDifference(), optimizer=optimizer, feature_library=ps.PolynomialLibrary(degree=1))
data = [y_l, y_m, y_n]
si_results = []
j = 0
for i in data:
    pysindy_model.fit(i, t=dt, feature_names=["nt"])
    trajectory = pysindy_model.simulate(i[0].flatten(), t)
    trajectory_loss = mean_squared_error(trajectory, df_low["y_true"].to_numpy())
    r2 = r2_score(df_low["y_true"].to_numpy(), trajectory)
    trajectories_list.append(trajectory)
    si_results.append(pysindy_model.predict(i))
    loss = mean_squared_error(si_results[j], decay_true)
    new_row = pd.DataFrame([{
        "Method name": f"Exp. Decay: PySINDy, Noise Level {j + 1}",
        "Discovered Diff eq.": pysindy_model.equations(),
        "MSE of Diff eq.": loss,
        "MSE of trajectory": trajectory_loss,
        "R^2 trajectory score": r2
    }])
    final_table = pd.concat([final_table, new_row], ignore_index=True)
    j += 1

def gen_log_growth_data(P0 = 100, r = 1.1, K = 1000, t = 50.0, num_points = 200, noise_percentage=0.0):
    t = np.linspace(0, t, num_points)
    A = (K-P0)/P0
    P = K/(1+A*np.exp(-r*t))
    true_std = np.std(P)
    noise = rng.normal(loc=0.0, scale=1, size=num_points)
    p_noisy = P + noise * noise_percentage*true_std
    df = pd.DataFrame({
        'time': t,
        'pop_true': P,
        'pop_noisy': p_noisy
    })
    return df
df_log_l = gen_log_growth_data(noise_percentage=0.01)
df_log_m = gen_log_growth_data(noise_percentage = 0.05)
df_log_n = gen_log_growth_data(noise_percentage=0.1)

t = df_log_l["time"].to_numpy()
dt = t[1] - t[0]
p_l = df_log_l['pop_noisy'].to_numpy()
p_m = df_log_m['pop_noisy'].to_numpy()
p_n = df_log_n['pop_noisy'].to_numpy()
log_true = np.gradient(df_log_l['pop_true'].to_numpy(), dt)

data = [p_l.reshape(-1,1), p_m.reshape(-1,1), p_n.reshape(-1,1)]
target = [np.gradient(p_l, dt), np.gradient(p_m, dt), np.gradient(p_n, dt)]
sr_log_results = []
model = PySRRegressor(
    parallelism = 'serial',
    deterministic = True,
    random_state = pysr_rand_seed,
    niterations =1000,
    binary_operators=["*", "/", "+", "-"],
    unary_operators=["exp"],
    model_selection="best"
)
for i in range (len(target)):
    model.fit(data[i], target[i], variable_names=["pt"])
    sr_log_results.append(model.predict(data[i]))
    trajectory = integrate_pysr_eq([100],"pt", model.sympy(), t)
    trajectory_loss = mean_squared_error(trajectory, df_log_l["pop_true"].to_numpy())
    r2 = r2_score(df_log_l["pop_true"].to_numpy(), trajectory)
    trajectories_list.append(trajectory)
    loss = mean_squared_error(sr_log_results[i], log_true)
    new_row = pd.DataFrame([{
        "Method name": f"Log Growth: PySR, Noise Level {i + 1}",
        "Discovered Diff eq.": model.sympy(),
        "MSE of Diff eq.": loss,
        "MSE of trajectory": trajectory_loss,
        "R^2 trajectory score": r2
    }])
    final_table = pd.concat([final_table, new_row], ignore_index=True)

si_log_results = []
data = [p_l, p_m, p_n]
optimizer = ps.STLSQ(threshold= 0.0001)
lib = ps.PolynomialLibrary(degree=3)
pysindy_model = ps.SINDy(differentiation_method=ps.FiniteDifference(), optimizer=optimizer, feature_library=lib)
j = 0
for i in data:
    pysindy_model.fit(i, t=dt, feature_names = ["pt"])
    si_log_results.append(pysindy_model.predict(i))
    trajectory = pysindy_model.simulate(i[0].flatten(), t)
    trajectory_loss = mean_squared_error(trajectory, df_log_l["pop_true"].to_numpy())
    r2 = r2_score(df_log_l["pop_true"].to_numpy(), trajectory)
    trajectories_list.append(trajectory)
    loss = mean_squared_error(si_log_results[j], log_true)
    new_row = pd.DataFrame([{
        "Method name": f"Log Growth: PySINDy, Noise Level {j + 1}",
        "Discovered Diff eq.": pysindy_model.equations(),
        "MSE of Diff eq.": loss,
        "MSE of trajectory": trajectory_loss,
        "R^2 trajectory score": r2
    }])
    final_table = pd.concat([final_table, new_row], ignore_index=True)
    j += 1

def gen_pend_data(theta_max = np.pi/2, g = 9.8, L = 3, t_end = 10.0, num_points = 200, noise_percentage=0.0):
    t_eval = np.linspace(0, t_end, num_points)

    # Define the exact non-linear ODE system
    def exact_pendulum(t, y):
        theta, omega = y
        return [omega, -(g/L) * np.sin(theta)]

    # Solve the ODE (Initial conditions: start at theta_max, zero initial velocity)
    sol = solve_ivp(exact_pendulum, [0, t_end], [theta_max, 0], t_eval=t_eval)

    th = sol.y[0]
    omega = sol.y[1] #Review 4: extracting angular velocity
    true_std = np.std(th)
    noise = rng.normal(loc=0.0, scale= 1, size=num_points)
    th_noisy = th + noise * noise_percentage * true_std

    df = pd.DataFrame({'time': sol.t, 'th_true': th, 'th_noisy': th_noisy})

    return df, omega

df_pen_l, omega_l = gen_pend_data(noise_percentage=0.01)
df_pen_m, omega_m = gen_pend_data(noise_percentage=0.05)
df_pen_n, omega_n = gen_pend_data(noise_percentage=0.1)

X_l = np.stack((df_pen_l["th_noisy"], omega_l), axis = -1)
X_m = np.stack((df_pen_m["th_noisy"], omega_m), axis = -1)
X_n = np.stack((df_pen_n["th_noisy"], omega_n), axis = -1)

t = df_pen_l["time"].to_numpy()
dt = t[1] - t[0]
data = [X_l, X_m, X_n]
thetas = [df_pen_l["th_noisy"], df_pen_m["th_noisy"], df_pen_n["th_noisy"]]
omegas = [omega_l, omega_m, omega_n]
fd = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 5}, d=2)
pen_true = df_pen_l['th_true'].to_numpy()
pen_true = fd._differentiate(pen_true, t=dt)

si_pen_results = []
# If lambda x:x (linear term) is added to the library below, it will inevitably appear in discovered equations (as stated in the introduction), thus exaggerating sin(x) term.
library_functions = [lambda x: np.sin(x)]
library_function_names = [lambda x: f"sin({x})"]
lib = ps.CustomLibrary(library_functions=library_functions, function_names=library_function_names)

optimizer = ps.STLSQ(threshold=0.1)
pysindy_model = ps.SINDy(optimizer=optimizer, feature_library=lib)
for i in range (len(data)):
    pysindy_model.fit(data[i], t=dt,  feature_names=["theta", "omega"])
    x0=[thetas[i][0], omegas[i][0]]
    X_simulated = pysindy_model.simulate(x0, t)
    trajectory = X_simulated[:, 0]
    trajectory_loss = mean_squared_error(trajectory, df_pen_l['th_true'].to_numpy())
    r2 = r2_score(df_pen_l['th_true'].to_numpy(), trajectory)
    trajectories_list.append(trajectory)
    si_pen_results.append(pysindy_model.predict(data[i]))
    loss = mean_squared_error(si_pen_results[i][:, 1], pen_true)
    new_row = pd.DataFrame([{
        "Method name": f"Oscillator: PySINDy, Noise Level {j + 1}",
        "Discovered Diff eq.": pysindy_model.equations(),
        "MSE of Diff eq.": loss,
        "MSE of trajectory": trajectory_loss,
        "R^2 of trajectory": r2
    }])
    final_table = pd.concat([final_table, new_row], ignore_index=True)

si_pen_filtered_results = []
filtered_data= [wiener(X_l, mysize=21), wiener(X_m, mysize=21), wiener(X_n, mysize=21)]
x_dot_filtered = [fd._differentiate(filtered_data[0], dt), fd._differentiate(filtered_data[1], dt), fd._differentiate(filtered_data[2], dt)]
j = 0
for i in range(len(filtered_data)):
    pysindy_model.fit(filtered_data[i], t=dt, feature_names=["theta", "omega"])
    x0 = [thetas[i][0], omegas[i][0]]
    X_simulated = pysindy_model.simulate(x0, t)
    trajectory = X_simulated[:, 0]
    trajectory_loss = mean_squared_error(trajectory, df_pen_l['th_true'].to_numpy())
    r2 = r2_score(df_pen_l['th_true'].to_numpy(), trajectory)
    trajectories_list.append(trajectory)
    si_pen_filtered_results.append(pysindy_model.predict(filtered_data[i]))
    loss = mean_squared_error(si_pen_filtered_results[i][:, 1], pen_true)
    new_row = pd.DataFrame([{
        "Method name": f"Oscillator: PySINDy, Filtered, Noise Level {j + 1}",
        "Discovered Diff eq.": pysindy_model.equations(),
        "MSE of Diff eq.": loss,
        "MSE of trajectory": trajectory_loss,
        "R^2 of trajectory": r2
    }])
    final_table = pd.concat([final_table, new_row], ignore_index=True)

pen_l = df_pen_l['th_noisy'].to_numpy()
pen_m = df_pen_m['th_noisy'].to_numpy()
pen_n = df_pen_n['th_noisy'].to_numpy()

data = [pen_l.reshape(-1,1), pen_m.reshape(-1,1),  pen_n.reshape(-1,1)]
sr_pen_results = []
model = PySRRegressor(
    niterations = 1000,
    deterministic= True,
    random_state= pysr_rand_seed,
    parallelism= 'serial',
    binary_operators= ["-", "+", "*", "/"],
    unary_operators= ["sin", "exp", "cos", "sqrt"]
)
j = 0
for i in data: #Review 2: fixed
    x_dot = fd._differentiate(i, t = dt)
    model.fit(i, x_dot, variable_names = ["th_t"])
    sr_pen_results.append(model.predict(i))
    loss = mean_squared_error(sr_pen_results[j], pen_true)
    trajectory = integrate_pysr_eq_2nd_order([np.pi / 2, 0], "th_t", model.sympy(), t)
    # Catch unstable equations that failed to integrate over the full time span
    if len(trajectory) == len(t):
        trajectory_loss = mean_squared_error(trajectory, df_pen_l['th_true'].to_numpy())
        r2 = r2_score(df_pen_l['th_true'].to_numpy(), trajectory)
    else:
        # Penalize the unstable model heavily
        trajectory_loss = np.inf
        r2 = np.inf

        # Pad the incomplete trajectory with NaNs
        trajectory = np.pad(trajectory, (0, len(t) - len(trajectory)), constant_values=np.nan)
    trajectories_list.append(trajectory)
    new_row = pd.DataFrame([{
        "Method name": f"Oscillator: PySR, Noise Level {j + 1}",
        "Discovered Diff eq.": model.sympy(),
        "MSE of Diff eq.": loss,
        "MSE of trajectory": trajectory_loss,
        "R^2 of trajectory": r2
    }])
    final_table = pd.concat([final_table, new_row], ignore_index=True)
    j += 1
t = df_low["time"].to_numpy()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
for i, ax in enumerate(axes.flat):
    ax.plot(t, decay_true, linestyle="--", label = "True eq.")
    ax.plot(t, results[i], linestyle="dotted", label = "Discovered eq. PySR")
    ax.plot(t, si_results[i], label = "Discovered eq. PySINDy")
    ax.legend()
plt.tight_layout()
fig.savefig("results/exp_decay_pysr_pysindy.png")
plt.show() #dotted line – true equation

t = df_log_l["time"].to_numpy()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
for i, ax in enumerate(axes.flat):
    ax.plot(t, sr_log_results[i], linestyle="dotted", label = "Discovered eq. PySR")
    ax.plot(t, si_log_results[i], label = "Discovered eq. PyINDy")
    ax.plot(t, log_true, linestyle="--", label = "True eq.")
    ax.legend()
plt.tight_layout()
fig.savefig("results/log_growth_pysr_pysindy.png")
plt.show() #dotted line – true equation

true = -9.8/3*np.sin(df_pen_l["th_true"])
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, ax in enumerate(axes.flat):
    ax.plot(t, sr_pen_results[i], label = "Discovered eq.")
    ax.plot(t, true, linestyle="--", label = "True eq.")
    ax.legend()
plt.tight_layout()
fig.savefig("results/osc_pysr.png")
plt.show() # dotted line – true equation
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, ax in enumerate(axes.flat):
    ax.plot(t, si_pen_results[i][:, 1], label = "Discovered eq.")
    ax.plot(t, si_pen_filtered_results[i][:, 1], color="red", label = "Eq. from filtered data")
    ax.plot(t, true, linestyle="--", label = "True eq.")
    ax.legend()
plt.tight_layout()
fig.savefig("results/osc_pysindy_pysindyfiltered.png")
plt.show()


#Plotting trajectories
fig, axs = plt.subplots(7, 3, figsize=(60, 20))
row_labels = ['Exp Decay: SR', 'Exp Decay: SINDy', 'Log Growth: SR', 'Log Growth: SINDy', 'Oscillator: SINDy', 'Oscillator (filtr.): SINDy', "Oscillator: SR"]

for ax, label in zip(axs[:, 0], row_labels):
    ax.annotate(label,
                xy=(0, 0.5),
                xytext=(-40, 0),
                xycoords='axes fraction',
                textcoords='offset points',
                size='large',
                ha='right',
                va='center',
                rotation=90)
axs = axs.flatten()
for i in range(len(trajectories_list)):
    axs[i].plot(t, trajectories_list[i], label="Discovered Trajectory")
    if i <= 5:
        axs[i].plot(t, df_low["y_true"].to_numpy(), linestyle="--", label="True Trajectory")
    elif i > 5 and i <= 11:
        axs[i].plot(t, df_log_l["pop_true"].to_numpy(), linestyle="--", label="True Trajectory")
    elif i > 11:
        axs[i].plot(t, df_pen_l["th_true"].to_numpy(), linestyle="--", label="True Trajectory")
    else:
        raise ValueError
    axs[i].legend()
plt.tight_layout()
fig.savefig("results/all_trajectories.png")
plt.show()

final_table.to_csv("results/summary.csv")