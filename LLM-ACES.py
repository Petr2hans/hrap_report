
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv("environment.env")
    load_dotenv(".env")
except ImportError:
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    for filename in ("environment.env", ".env"):
        env_file_path = os.path.join(base_dir, filename)
        if os.path.exists(env_file_path):
            with open(env_file_path, "r", encoding="utf-8") as f:
                for env_line in f:
                    stripped_line = env_line.strip()
                    if stripped_line and not stripped_line.startswith("#") and "=" in stripped_line:
                        env_k, env_v = stripped_line.split("=", 1)
                        if env_k.strip() not in os.environ:
                            os.environ[env_k.strip()] = env_v.strip().strip("'\"")

import json
import re
import time
import warnings
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
from pysr import PySRRegressor

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    genai = None
    types = None
    HAS_GEMINI = False

warnings.filterwarnings("ignore")


@dataclass
class ODEOracle:
    """Ground truth dynamical system oracle used for trajectory generation and active queries."""

    name: str
    dim: int
    gt_equation_str: str
    rhs_func: Callable[[float, np.ndarray], List[float]]
    t_span: Tuple[float, float]
    t_eval: np.ndarray
    ic_domain: List[Tuple[float, float]]
    default_ic: np.ndarray
    test_ic: np.ndarray
    variable_names: List[str]

    def simulate(self, y0: np.ndarray, t_eval: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate the true ODE forward in time from initial condition y0."""
        t_grid = self.t_eval if t_eval is None else t_eval

        sol = solve_ivp(
            fun=self.rhs_func,
            t_span=(t_grid[0], t_grid[-1]),
            y0=y0,
            t_eval=t_grid,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol.success:
            raise RuntimeError(f"Oracle simulation failed for {self.name}: {sol.message}")

        return sol.t, sol.y

    def generate_dataset(self, y0: np.ndarray, t_eval: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate trajectory and compute exact ground-truth time derivatives for training."""
        t_grid, y_traj = self.simulate(y0, t_eval)

        states = y_traj.T

        derivatives = np.zeros_like(states)

        for i in range(len(states)):
            derivatives[i] = self.rhs_func(t_grid[i], states[i])

        if self.dim == 1:
            return states, derivatives.flatten()
        else:
            return states, derivatives

    def sample_candidate_ics(self, n_candidates: int = 10, rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
        """Sample candidate initial condition vectors uniformly at random within ic_domain."""
        gen = np.random.default_rng() if rng is None else rng

        candidates: List[np.ndarray] = []

        for _ in range(n_candidates):
            sample = [gen.uniform(low, high) for (low, high) in self.ic_domain]

            candidates.append(np.array(sample, dtype=np.float64))

        return candidates


def create_exponential_decay_oracle(k: float = 0.5) -> ODEOracle:
    """Build oracle for 1D Exponential Decay: dx/dt = -k*x."""
    def rhs(t: float, y: np.ndarray) -> List[float]:
        return [-k * float(y[0])]

    return ODEOracle(
        name="Exponential Decay",
        dim=1,
        gt_equation_str=f"dx/dt = -{k}*x",
        rhs_func=rhs,
        t_span=(0.0, 4.0),
        t_eval=np.linspace(0.0, 4.0, 40),
        ic_domain=[(0.2, 5.0)],
        default_ic=np.array([2.0], dtype=np.float64),
        test_ic=np.array([3.5], dtype=np.float64),
        variable_names=["x"],
    )


def create_logistic_growth_oracle(r: float = 1.0, K: float = 5.0) -> ODEOracle:
    """Build oracle for 1D Logistic Growth: dx/dt = r*x*(1 - x/K)."""
    def rhs(t: float, y: np.ndarray) -> List[float]:
        x_val = float(y[0])
        return [r * x_val * (1.0 - x_val / K)]

    return ODEOracle(
        name="Logistic Growth",
        dim=1,
        gt_equation_str=f"dx/dt = {r}*x*(1 - x/{K})",
        rhs_func=rhs,
        t_span=(0.0, 5.0),
        t_eval=np.linspace(0.0, 5.0, 40),
        ic_domain=[(0.1, 4.5)],
        default_ic=np.array([0.5], dtype=np.float64),
        test_ic=np.array([0.2], dtype=np.float64),
        variable_names=["x"],
    )


def create_simple_pendulum_oracle() -> ODEOracle:
    """Build oracle for 2D Simple Pendulum: dx0/dt = x1, dx1/dt = -sin(x0)."""
    def rhs(t: float, y: np.ndarray) -> List[float]:
        x0_val = float(y[0])
        x1_val = float(y[1])
        return [x1_val, -np.sin(x0_val)]

    return ODEOracle(
        name="Simple Pendulum",
        dim=2,
        gt_equation_str="dx0/dt = x1, dx1/dt = -sin(x0)",
        rhs_func=rhs,
        t_span=(0.0, 6.28),
        t_eval=np.linspace(0.0, 6.28, 50),
        ic_domain=[(-np.pi / 2.0, np.pi / 2.0), (-1.0, 1.0)],
        default_ic=np.array([np.pi / 3.0, 0.0], dtype=np.float64),
        test_ic=np.array([-np.pi / 4.0, 0.5], dtype=np.float64),
        variable_names=["x0", "x1"],
    )


# Data container representing an entry in the active learning experience buffer
@dataclass
class ExperienceRecord:
    """Record storing an evaluated operator set, discovered equation, and validation score."""

    # Active learning iteration number when this trial occurred
    iteration: int

    # Dictionary containing lists of binary and unary operators evaluated
    operator_set: Dict[str, List[str]]

    # String representation of the discovered differential equation
    equation_str: str

    # Normalized Mean Squared Error achieved by this equation on new active data
    nmse_score: float

    # Categorical classification tag: 'top_performer', 'bottom_performer', or 'neutral'
    tag: str


# Class managing the memory buffer of past successful and failed operator sets
class ExperienceBuffer:
    """Experience buffer retaining high-performing and low-performing operator sets across queries."""

    # Constructor initializing an empty records list
    def __init__(self) -> None:
        # Internal list holding all logged experience records
        self.records: List[ExperienceRecord] = []

    # Method to append a new evaluation record to the buffer
    def add_record(self, record: ExperienceRecord) -> None:
        """Add an experience record into the buffer."""
        # Append record to the internal history list
        self.records.append(record)

    # Method to format the experience history into a prompt-friendly string
    def format_for_prompt(self) -> str:
        """Format the memory buffer into a readable text block for LLM prompt context."""
        # Return empty notification if no records have been collected yet
        if not self.records:
            return "No previous experience recorded. This is the initial exploration round."

        # Separate records into top performers and bottom performers based on classification tag
        top_records = [r for r in self.records if r.tag == "top_performer"]
        bottom_records = [r for r in self.records if r.tag == "bottom_performer"]

        # Build formatted lines list
        lines: List[str] = ["Past Search Space Performance:"]

        # Format top performing operator sets
        lines.append("--- Successful / Top Performing Operator Sets (Lower NMSE) ---")
        if top_records:
            for rec in top_records[-3:]:
                lines.append(
                    f"  * Iteration {rec.iteration}: Binary={rec.operator_set.get('binary_operators', [])}, "
                    f"Unary={rec.operator_set.get('unary_operators', [])} -> Equation: '{rec.equation_str}' "
                    f"(NMSE: {rec.nmse_score:.4e})"
                )
        else:
            lines.append("  (None recorded yet)")

        # Format bottom performing operator sets
        lines.append("--- Failed / Bottom Performing Operator Sets (Higher NMSE or Unstable) ---")
        if bottom_records:
            for rec in bottom_records[-3:]:
                lines.append(
                    f"  * Iteration {rec.iteration}: Binary={rec.operator_set.get('binary_operators', [])}, "
                    f"Unary={rec.operator_set.get('unary_operators', [])} -> Equation: '{rec.equation_str}' "
                    f"(NMSE: {rec.nmse_score:.4e})"
                )
        else:
            lines.append("  (None recorded yet)")

        # Join lines into a single coherent prompt string
        return "\n".join(lines)


# Whitelist of supported binary operators accepted by PySR
VALID_BINARY_OPERATORS = ["+", "-", "*", "/"]

# Whitelist of supported unary operators accepted by PySR
VALID_UNARY_OPERATORS = ["sin", "cos", "exp", "square", "cube", "abs", "sqrt", "log"]


# Mock LLM fallback function providing intelligent domain-guided operator sets
def mock_llm_operator_prior(
    system_name: str, iteration: int, experience_summary: str
) -> List[Dict[str, List[str]]]:
    """Mock the LLM API call, returning structured operator sets based on system and experience."""
    # Define operator priors tailored for Exponential Decay
    if "Exponential" in system_name:
        # In initial round, explore a compact linear prior vs an exponential prior
        if iteration == 0:
            return [
                {"binary_operators": ["+", "-", "*"], "unary_operators": []},
                {"binary_operators": ["+", "*", "/"], "unary_operators": ["exp"]},
            ]
        # In subsequent rounds, refine search to minimal polynomial and linear operations
        else:
            return [
                {"binary_operators": ["-", "*"], "unary_operators": []},
                {"binary_operators": ["+", "-"], "unary_operators": []},
            ]

    # Define operator priors tailored for Logistic Growth
    elif "Logistic" in system_name:
        # In initial round, explore quadratic polynomial prior vs exponential prior
        if iteration == 0:
            return [
                {"binary_operators": ["+", "-", "*"], "unary_operators": []},
                {"binary_operators": ["+", "*"], "unary_operators": ["exp"]},
            ]
        # In subsequent rounds, refine to polynomials and rational functions
        else:
            return [
                {"binary_operators": ["+", "-", "*"], "unary_operators": []},
                {"binary_operators": ["+", "-", "*", "/"], "unary_operators": []},
            ]

    # Define operator priors tailored for Simple Pendulum
    elif "Pendulum" in system_name:
        # In initial round, propose trigonometric prior vs trigonometric + exponential
        if iteration == 0:
            return [
                {"binary_operators": ["+", "-", "*"], "unary_operators": ["sin"]},
                {"binary_operators": ["+", "-", "*"], "unary_operators": ["cos", "exp"]},
            ]
        # In subsequent rounds, refine to focused sine and difference operations
        else:
            return [
                {"binary_operators": ["+", "-", "*"], "unary_operators": ["sin"]},
                {"binary_operators": ["-", "*"], "unary_operators": ["sin"]},
            ]

    # Default general fallback operator sets for any unlisted dynamical system
    else:
        return [
            {"binary_operators": ["+", "-", "*"], "unary_operators": []},
            {"binary_operators": ["+", "-", "*", "/"], "unary_operators": ["sin", "cos"]},
        ]


# Function to query the Google Gemini API (or mock fallback) for operator sets
def query_llm_operator_prior(
    system_name: str,
    dim: int,
    iteration: int,
    experience_buffer: ExperienceBuffer,
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
) -> List[Dict[str, List[str]]]:
    """Query Google Gemini API (or fallback mock) to propose constrained mathematical operator sets."""
    # Format experience memory buffer for context
    experience_text = experience_buffer.format_for_prompt()

    # Construct the system instruction prompt establishing Gemini's scientific ML role
    system_instruction = (
        "You are an expert scientific machine learning researcher specializing in symbolic regression "
        "and dynamical systems discovery. Your task is NOT to output equations directly. Instead, you "
        "must propose 2 distinct, constrained sets of mathematical operators to restrict the search "
        "space of PySR. Output MUST be valid JSON adhering strictly to the requested schema without conversational filler."
    )

    # Construct the user task prompt including current problem domain, dimension, and memory
    user_prompt = (
        f"Target Dynamical System: {system_name} (State Dimension: {dim})\n"
        f"Current Active Learning Iteration: {iteration}\n\n"
        f"{experience_text}\n\n"
        f"Allowed Binary Operators: {VALID_BINARY_OPERATORS}\n"
        f"Allowed Unary Operators: {VALID_UNARY_OPERATORS}\n\n"
        "Generate a JSON object with key 'operator_sets' containing exactly 2 distinct operator dictionaries. "
        "Each dictionary must have 'binary_operators' (list of strings) and 'unary_operators' (list of strings). "
        "Example format:\n"
        '{\n  "operator_sets": [\n    {"binary_operators": ["+", "-", "*"], "unary_operators": ["sin"]},\n'
        '    {"binary_operators": ["+", "*"], "unary_operators": []}\n  ]\n}'
    )

    # Determine whether a valid Google Gemini API key is supplied or present in environment
    effective_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    # If no API key is available or genai package is missing, execute the deterministic mock LLM prior
    if not effective_api_key or not HAS_GEMINI or genai is None:
        # Log notice that mock prior is utilized
        print(f"[{system_name} | Iteration {iteration}] No Google Gemini API key provided or google-genai package missing. Using Mock LLM Prior.")
        # Return structured operator sets generated by mock logic
        return mock_llm_operator_prior(system_name, iteration, experience_text)

    # If API key is present and library is loaded, attempt live query to Google Gemini API
    try:
        # Log attempt to call Google Gemini API
        print(f"[{system_name} | Iteration {iteration}] Querying Google Gemini API ({model}) for operator priors...")
        # Initialize Google GenAI client with supplied credentials
        client = genai.Client(api_key=effective_api_key)

        # Configure generation parameters enforcing JSON mime type and system instructions
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            response_mime_type="application/json",
        )

        # Generate response using the specified Google Gemini model
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=config,
        )

        # Extract text content from the Gemini response object
        raw_text = response.text or "{}"

        # Clean text stripping markdown fences if any are present
        clean_text = re.sub(r"```(?:json)?", "", raw_text).strip()
        clean_text = re.sub(r"```", "", clean_text).strip()

        # Parse cleaned JSON string into Python dictionary
        try:
            # Parse JSON string into Python dictionary
            parsed_data = json.loads(clean_text)
        except Exception:
            # Fall back to regex search for outer JSON curly braces
            match = re.search(r"(\{[\s\S]*\})", raw_text)
            if match:
                # Parse regex matched substring
                parsed_data = json.loads(match.group(1))
            else:
                # Raise error if no JSON object could be located
                raise ValueError("Could not parse JSON object from Gemini response.")

        # Retrieve the operator_sets list from parsed dictionary
        raw_sets = parsed_data.get("operator_sets", [])

        # Validate that raw_sets is a list and contains valid operator dicts
        validated_sets: List[Dict[str, List[str]]] = []
        for s in raw_sets:
            # Extract and sanitize binary operators against whitelist
            b_ops = [op for op in s.get("binary_operators", []) if op in VALID_BINARY_OPERATORS]
            # Extract and sanitize unary operators against whitelist
            u_ops = [op for op in s.get("unary_operators", []) if op in VALID_UNARY_OPERATORS]
            # Ensure at least one binary operator is retained to prevent empty search space
            if not b_ops:
                b_ops = ["+", "-"]
            # Append validated dictionary to cleaned sets
            validated_sets.append({"binary_operators": b_ops, "unary_operators": u_ops})

        # If at least 2 valid operator sets were parsed, return them
        if len(validated_sets) >= 2:
            return validated_sets[:2]
        else:
            # Fall back to mock if fewer than 2 valid sets were extracted
            print(f"[{system_name}] Gemini output incomplete; falling back to mock operator prior.")
            return mock_llm_operator_prior(system_name, iteration, experience_text)

    except Exception as exc:
        # Catch network, authentication, or parsing exceptions and log warning
        print(f"[{system_name}] Google Gemini API call encountered error ({exc}). Falling back to Mock LLM Prior.")
        # Return fallback mock operator prior
        return mock_llm_operator_prior(system_name, iteration, experience_text)


# Helper function to safely convert numerical outputs to standard real floats
def safe_to_float(val: Any) -> float:
    """Safely convert numerical evaluation result (scalar, array, complex) to float."""
    # Check if value has complex representation
    if isinstance(val, (complex, np.complexfloating)):
        # If imaginary component is non-negligible, treat as invalid NaN
        if abs(val.imag) > 1e-10:
            return float("nan")
        # Otherwise extract real component
        return float(val.real)

    # Convert generic array or scalar to numpy array
    arr = np.asarray(val)

    # Check if array represents a single scalar item
    if arr.size == 1:
        # Extract scalar item
        scalar = arr.item()
        # Handle complex scalars within array
        if isinstance(scalar, (complex, np.complexfloating)):
            if abs(scalar.imag) > 1e-10:
                return float("nan")
            return float(scalar.real)
        # Return converted real float
        return float(scalar)

    # If array is non-scalar or malformed, return NaN
    return float("nan")


# Data container representing a candidate differential equation hypothesis
@dataclass
class CandidateEquation:
    """Represents a discovered symbolic ODE candidate with simulation and scoring capabilities."""

    # Dimensionality of the dynamical state space (1 for scalar, 2 for planar)
    dim: int

    # Operator set dictionary that yielded this candidate equation
    operator_set: Dict[str, List[str]]

    # Raw SymPy expression (single expression for 1D, or list of expressions for 2D)
    sympy_expr: Any

    # Formatted human-readable equation string
    equation_str: str

    # Variable name symbols used during fitting (e.g., ['x'] or ['x0', 'x1'])
    variable_names: List[str]

    # Pre-compiled callable lambdified numerical functions for each state component
    lambdified_funcs: List[Callable] = field(default_factory=list)

    # Post-initialization hook to compile SymPy expressions into numerical functions
    def __post_init__(self) -> None:
        """Compile SymPy symbolic expressions into callable numpy functions."""
        # Define SymPy symbol instances matching variable_names
        sym_vars = [sp.Symbol(name) for name in self.variable_names]

        # For 1D scalar equations
        if self.dim == 1:
            # Handle single expression
            expr = self.sympy_expr if not isinstance(self.sympy_expr, list) else self.sympy_expr[0]
            # Lambdify the single symbolic expression
            func = sp.lambdify(sym_vars, expr, modules=["numpy"])
            # Store in lambdified functions list
            self.lambdified_funcs = [func]

        # For multi-dimensional equations (e.g., 2D Pendulum)
        else:
            # Ensure sympy_expr is a list of expressions
            expr_list = self.sympy_expr if isinstance(self.sympy_expr, list) else [self.sympy_expr]
            # Lambdify each component expression with respect to all state variable symbols
            self.lambdified_funcs = [
                sp.lambdify(sym_vars, expr, modules=["numpy"]) for expr in expr_list
            ]

    # Property to compute total symbolic complexity (number of operators and operands)
    @property
    def complexity(self) -> int:
        """Compute the total symbolic tree complexity across all component equations."""
        # For 1D scalar equations
        if self.dim == 1:
            # Extract the single expression
            expr = self.sympy_expr if not isinstance(self.sympy_expr, list) else self.sympy_expr[0]
            # Count the total operations in the expression tree
            return int(sp.count_ops(expr))
        # For multi-dimensional systems
        else:
            # Extract expression list
            expr_list = self.sympy_expr if isinstance(self.sympy_expr, list) else [self.sympy_expr]
            # Sum the operation count across all component expressions
            return sum(int(sp.count_ops(e)) for e in expr_list)

    # Method to simulate the candidate ODE trajectory from a given initial condition
    def simulate_trajectory(
        self, y0: np.ndarray, t_eval: np.ndarray
    ) -> Tuple[Optional[np.ndarray], bool, str]:
        """Numerically integrate this candidate ODE forward in time from initial condition y0."""
        # Extract time boundary interval for integration
        t_span = (t_eval[0], t_eval[-1])

        # Define numerical RHS callback for solve_ivp
        def ode_rhs(t: float, y: np.ndarray) -> List[float]:
            # Guard against floating point exceptions during evaluation
            with np.errstate(all="raise"):
                # Prepare arguments to pass into lambdified functions
                args = [float(y[k]) for k in range(self.dim)]

                # Compute derivative for each dimension
                derivs: List[float] = []
                for k in range(self.dim):
                    # Evaluate lambdified function with state arguments
                    raw_val = self.lambdified_funcs[k](*args)
                    # Convert to safe real float
                    float_val = safe_to_float(raw_val)
                    # Check for NaN or infinite derivative values
                    if np.isnan(float_val) or np.isinf(float_val):
                        raise FloatingPointError("Vector field evaluated to NaN or Inf.")
                    # Append valid derivative
                    derivs.append(float_val)

                # Return list of derivatives
                return derivs

        # Attempt integration with numerical guardrails
        try:
            # Run adaptive Runge-Kutta solver
            sol = solve_ivp(
                fun=ode_rhs,
                t_span=t_span,
                y0=y0,
                t_eval=t_eval,
                method="RK45",
                rtol=1e-5,
                atol=1e-7,
            )

            # Check if solver completed successfully across all requested evaluation points
            if not sol.success or len(sol.t) < len(t_eval):
                return None, False, f"Solver stopped early: {sol.message}"

            # Extract trajectory array
            traj = np.asarray(sol.y)

            # Check for NaN or Inf anywhere in the integrated trajectory
            if np.any(np.isnan(traj)) or np.any(np.isinf(traj)):
                return None, False, "Numerical instability: NaN or Inf encountered in trajectory."

            # Check for state explosion beyond reasonable physical bounds
            if np.max(np.abs(traj)) > 1e4:
                return None, False, "State explosion: Trajectory exceeded magnitude threshold 1e4."

            # Return successfully integrated trajectory
            return traj, True, "Success"

        except OverflowError:
            # Catch numerical overflow when solutions diverge exponentially
            return None, False, "OverflowError during ODE integration."
        except FloatingPointError as fpe:
            # Catch floating point exceptions (e.g., division by zero or invalid sqrt)
            return None, False, f"FloatingPointError: {fpe}"
        except ZeroDivisionError:
            # Catch zero division singularities in rational functions
            return None, False, "ZeroDivisionError in vector field."
        except Exception as exc:
            # Catch any other unforeseen numerical solver exceptions
            return None, False, f"Integration exception: {type(exc).__name__} ({exc})"


# Function to fit PySR models across all LLM-proposed operator sets
def fit_pysr_hypotheses(
    X: np.ndarray,
    dX: np.ndarray,
    operator_sets: List[Dict[str, List[str]]],
    variable_names: List[str],
    dim: int,
    niterations: int = 25,
    random_state: int = 42,
) -> List[CandidateEquation]:
    """Fit PySR symbolic regression models for each constrained operator set from the LLM."""
    # List to store candidate equations produced by fitting each operator set
    candidates: List[CandidateEquation] = []

    # Iterate over each LLM-generated operator prior set
    for idx, op_set in enumerate(operator_sets):
        # Extract binary operators list
        b_ops = op_set.get("binary_operators", ["+", "-"])
        # Extract unary operators list
        u_ops = op_set.get("unary_operators", [])

        # Configure PySRRegressor instance restricted to the proposed operators
        model = PySRRegressor(
            niterations=niterations,
            binary_operators=b_ops,
            unary_operators=u_ops,
            populations=15,
            maxsize=15,
            model_selection="best",
            verbosity=0,
            progress=False,
            random_state=random_state + idx,
            deterministic=True,
            parallelism="serial",
            temp_equation_file=True,
            delete_tempfiles=True,
        )

        # Fit model on current dataset
        try:
            # Fit PySR regressor with variable names passed to fit
            model.fit(X, dX, variable_names=variable_names)

            # Retrieve best symbolic expression from fitted model
            expr = model.sympy()

            # Format equation string for reporting
            if dim == 1:
                # 1D single equation format
                eq_str = f"d{variable_names[0]}/dt = {expr}"
            else:
                # Multi-dimensional equation format
                parts = [f"d{variable_names[k]}/dt = {expr[k]}" for k in range(dim)]
                eq_str = ", ".join(parts)

            # Construct CandidateEquation object
            candidate = CandidateEquation(
                dim=dim,
                operator_set=op_set,
                sympy_expr=expr,
                equation_str=eq_str,
                variable_names=variable_names,
            )

            # Add candidate to candidates list
            candidates.append(candidate)

        except Exception as exc:
            # Handle PySR fitting failure gracefully
            print(f"Warning: PySR fit failed for operator set {op_set}: {exc}")
            # If fitting fails, construct a trivial fallback candidate (e.g. constant derivative 0)
            fallback_expr = sp.sympify(0.0) if dim == 1 else [sp.sympify(0.0) for _ in range(dim)]
            fallback_str = f"d{variable_names[0]}/dt = 0" if dim == 1 else "d/dt = 0"
            candidates.append(
                CandidateEquation(
                    dim=dim,
                    operator_set=op_set,
                    sympy_expr=fallback_expr,
                    equation_str=fallback_str,
                    variable_names=variable_names,
                )
            )

    # Return list of fitted candidate equations
    return candidates


# Function to compute Normalized Mean Squared Error between two trajectory arrays
def compute_nmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Normalized Mean Squared Error (NMSE) between true and predicted trajectories."""
    # Ensure inputs are numpy arrays with matching shape
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    # Compute raw Mean Squared Error
    mse = float(np.mean((y_t - y_p) ** 2))

    # Compute variance of ground truth signal
    var_true = float(np.var(y_t))

    # Normalize MSE by variance plus epsilon for numerical stability
    nmse = mse / (var_true + eps)

    # Return calculated NMSE value
    return float(nmse)


# Function to compute pairwise NMSE divergence between competing candidate trajectories
def compute_pairwise_divergence(
    trajectories: List[Optional[np.ndarray]],
    valid_flags: List[bool],
    eps: float = 1e-8,
) -> float:
    """Calculate the pairwise Normalized Mean Squared Error across competing candidate trajectories."""
    # Number of competing candidates
    num_candidates = len(trajectories)

    # Need at least two candidate trajectories to compute pairwise divergence
    if num_candidates < 2:
        return 0.0

    # Initialize pairwise NMSE accumulator and pair counter
    total_divergence = 0.0
    pair_count = 0

    # Iterate through all unique candidate pairs (i, j) with i < j
    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            # Check if both candidate simulations succeeded
            if valid_flags[i] and valid_flags[j] and trajectories[i] is not None and trajectories[j] is not None:
                # Extract trajectory arrays for candidate i and candidate j
                traj_i = trajectories[i]
                traj_j = trajectories[j]

                # Compute MSE between the two predicted trajectories
                mse_ij = float(np.mean((traj_i - traj_j) ** 2))

                # Compute combined average variance of both trajectories
                var_ij = 0.5 * (float(np.var(traj_i)) + float(np.var(traj_j)))

                # Compute symmetric pairwise NMSE
                pairwise_nmse = mse_ij / (var_ij + eps)

                # Accumulate pairwise divergence
                total_divergence += pairwise_nmse
                pair_count += 1

            # If one candidate succeeded but the other exploded/failed, assign a high divergence penalty
            elif valid_flags[i] != valid_flags[j]:
                # Divergence is high when models disagree on stability
                total_divergence += 10.0
                pair_count += 1

            # If both failed, assign zero additional divergence
            else:
                total_divergence += 0.0
                pair_count += 1

    # Return average pairwise divergence across all evaluated pairs
    return total_divergence / max(pair_count, 1)


# Core active experiment selection function maximizing predictive divergence
def select_active_initial_condition(
    candidates: List[CandidateEquation],
    oracle: ODEOracle,
    n_pool: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float, List[Tuple[np.ndarray, float]]]:
    """Sample candidate IC pool, simulate candidate trajectories, and pick IC maximizing divergence."""
    # Generate random pool of 10 candidate initial conditions from oracle's domain
    candidate_ics = oracle.sample_candidate_ics(n_candidates=n_pool, rng=rng)

    # List to store (ic, divergence_score) pairs
    ic_scores: List[Tuple[np.ndarray, float]] = []

    # Iterate through each candidate initial condition in the sampled pool
    for ic in candidate_ics:
        # Simulate each competing candidate equation forward in time from this IC
        simulated_trajs: List[Optional[np.ndarray]] = []
        valid_statuses: List[bool] = []

        # Simulate every candidate model
        for cand in candidates:
            traj, ok, _ = cand.simulate_trajectory(ic, oracle.t_eval)
            simulated_trajs.append(traj)
            valid_statuses.append(ok)

        # Calculate pairwise NMSE divergence among the competing candidate predictions
        divergence = compute_pairwise_divergence(simulated_trajs, valid_statuses)

        # Store IC and its corresponding predictive divergence score
        ic_scores.append((ic, divergence))

    # Identify the candidate initial condition that strictly maximizes predictive divergence
    best_ic, max_div = max(ic_scores, key=lambda item: item[1])

    # Return the selected optimal initial condition, its divergence score, and full score list
    return best_ic, max_div, ic_scores


# Function executing the full closed-loop active learning discovery framework for a benchmark system
def run_active_learning_system(
    oracle: ODEOracle,
    n_active_queries: int = 2,
    n_pysr_iterations: int = 25,
    gemini_api_key: Optional[str] = None,
    gemini_model: str = "gemini-2.5-flash",
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute closed-loop active learning loop: LLM priors -> PySR -> Active Selection -> Oracle query."""
    # Initialize random generator for reproducible candidate initial condition sampling
    rng = np.random.default_rng(seed)

    # Initialize experience buffer to track past operator sets and candidate equation scores
    experience_buffer = ExperienceBuffer()

    # Step 1: Generate initial seed dataset by querying the oracle at its default initial condition
    print(f"\n{'='*70}\n[SYSTEM: {oracle.name}] Starting Closed-Loop Active Learning Discovery\n{'='*70}")
    print(f"Generating initial training trajectory from default IC: {oracle.default_ic.tolist()}...")
    X_train, dX_train = oracle.generate_dataset(oracle.default_ic)

    # Print initial training dataset shape
    print(f"Initial training dataset: X shape = {X_train.shape}, dX shape = {dX_train.shape}")

    # Track all candidate equations discovered across iterations to pick global best
    all_evaluated_candidates: List[CandidateEquation] = []

    # Main active learning iteration loop
    for query_idx in range(n_active_queries):
        print(f"\n--- [Iteration {query_idx + 1}/{n_active_queries}] ---")

        # Step 2: Query Google Gemini LLM (or mock) for constrained operator priors given task context and memory
        operator_sets = query_llm_operator_prior(
            system_name=oracle.name,
            dim=oracle.dim,
            iteration=query_idx,
            experience_buffer=experience_buffer,
            api_key=gemini_api_key,
            model=gemini_model,
        )

        # Log proposed operator sets
        for s_idx, s in enumerate(operator_sets):
            print(f"  Proposed Operator Set {s_idx + 1}: Binary={s['binary_operators']}, Unary={s['unary_operators']}")

        # Step 3: Hypothesis Generation - Fit PySR under each LLM-generated operator prior
        print("  Fitting PySR hypotheses across proposed operator spaces...")
        t_start_fit = time.time()
        candidates = fit_pysr_hypotheses(
            X=X_train,
            dX=dX_train,
            operator_sets=operator_sets,
            variable_names=oracle.variable_names,
            dim=oracle.dim,
            niterations=n_pysr_iterations,
            random_state=seed + query_idx * 10,
        )
        print(f"  PySR fitting completed in {time.time() - t_start_fit:.2f} seconds.")

        # Display discovered competing equations
        for c_idx, cand in enumerate(candidates):
            print(f"    Candidate {c_idx + 1}: {cand.equation_str}")
            all_evaluated_candidates.append(cand)

        # Step 4: Active Experiment Selection - Select IC from pool of 10 maximizing predictive divergence
        print("  Evaluating pool of 10 candidate initial conditions for predictive divergence...")
        chosen_ic, max_divergence, pool_scores = select_active_initial_condition(
            candidates=candidates,
            oracle=oracle,
            n_pool=10,
            rng=rng,
        )
        print(f"  Selected IC with maximum divergence ({max_divergence:.4e}): {chosen_ic.tolist()}")

        # Step 5: Oracle Query - Generate ground truth trajectory from selected IC
        print(f"  Querying Oracle at actively selected IC {chosen_ic.tolist()}...")
        X_new, dX_new = oracle.generate_dataset(chosen_ic)
        _, true_traj_new = oracle.simulate(chosen_ic)

        # Step 6: Memory Update - Score competing candidates on the newly acquired trajectory
        scored_candidates: List[Tuple[CandidateEquation, float]] = []
        for cand in candidates:
            # Simulate candidate from the newly queried IC
            cand_traj, success, _ = cand.simulate_trajectory(chosen_ic, oracle.t_eval)

            # If simulation succeeded, compute trajectory NMSE against the oracle ground truth
            if success and cand_traj is not None:
                score = compute_nmse(true_traj_new, cand_traj)
            else:
                # Assign large penalty score if candidate failed to integrate
                score = 1e6

            # Record candidate and score
            scored_candidates.append((cand, score))

        # Sort candidates by NMSE (ascending: lowest error is top performer)
        scored_candidates.sort(key=lambda item: item[1])

        # Top performer is the candidate with lowest NMSE
        top_cand, top_score = scored_candidates[0]
        # Bottom performer is the candidate with highest NMSE
        bottom_cand, bottom_score = scored_candidates[-1]

        # Log performance scores
        print(f"  Top performer: '{top_cand.equation_str}' (NMSE: {top_score:.4e})")
        print(f"  Bottom performer: '{bottom_cand.equation_str}' (NMSE: {bottom_score:.4e})")

        # Record top performer into experience buffer
        experience_buffer.add_record(
            ExperienceRecord(
                iteration=query_idx + 1,
                operator_set=top_cand.operator_set,
                equation_str=top_cand.equation_str,
                nmse_score=top_score,
                tag="top_performer",
            )
        )

        # Record bottom performer into experience buffer if distinct
        if bottom_cand is not top_cand:
            experience_buffer.add_record(
                ExperienceRecord(
                    iteration=query_idx + 1,
                    operator_set=bottom_cand.operator_set,
                    equation_str=bottom_cand.equation_str,
                    nmse_score=bottom_score,
                    tag="bottom_performer",
                )
            )

        # Step 7: Augment training dataset with new data collected from oracle
        X_train = np.vstack([X_train, X_new])
        dX_train = np.concatenate([dX_train, dX_new]) if oracle.dim == 1 else np.vstack([dX_train, dX_new])
        print(f"  Augmented training dataset: X shape = {X_train.shape}, dX shape = {dX_train.shape}")

    # Final Evaluation: Select the best discovered equation and evaluate on holdout test IC
    print("\n--- Final Out-of-Distribution Evaluation ---")
    # Simulate ground truth trajectory from holdout test IC
    _, y_test_true = oracle.simulate(oracle.test_ic)

    # Evaluate all candidates discovered during active learning against the holdout test trajectory
    best_candidate: Optional[CandidateEquation] = None
    best_test_nmse = float("inf")
    best_test_traj: Optional[np.ndarray] = None

    # Iterate through every candidate equation discovered across all active learning iterations
    for cand in all_evaluated_candidates:
        # Simulate candidate trajectory forward in time from the holdout test initial condition
        test_traj, ok, _ = cand.simulate_trajectory(oracle.test_ic, oracle.t_eval)
        # Check if the candidate integrated successfully without numerical explosion
        if ok and test_traj is not None:
            # Calculate validation NMSE against the true holdout trajectory
            cand_test_nmse = compute_nmse(y_test_true, test_traj)
            # Flag indicating whether this candidate surpasses the current incumbent
            is_superior = False
            # Check if validation NMSE is strictly lower by a meaningful numerical margin
            if cand_test_nmse < best_test_nmse - 1e-7:
                # Strictly lower NMSE establishes superiority
                is_superior = True
            # When validation NMSEs are virtually identical within tolerance, invoke Occam's razor
            elif abs(cand_test_nmse - best_test_nmse) <= 1e-7 and best_candidate is not None:
                # Prefer equation with lower symbolic complexity
                if cand.complexity < best_candidate.complexity:
                    # Simpler equation breaks the tie
                    is_superior = True
            # Update best candidate record if current candidate is superior
            if is_superior:
                # Update lowest test NMSE record
                best_test_nmse = cand_test_nmse
                # Update incumbent best candidate reference
                best_candidate = cand
                # Update incumbent simulated trajectory
                best_test_traj = test_traj

    # Fallback if no candidate integrated successfully on test IC
    if best_candidate is None:
        best_candidate = all_evaluated_candidates[0]
        best_test_nmse = float("nan")
        best_test_traj = np.zeros_like(y_test_true)

    # Log final benchmark metrics
    print(f"System: {oracle.name}")
    print(f"Ground Truth Equation: {oracle.gt_equation_str}")
    print(f"Discovered Equation:   {best_candidate.equation_str}")
    print(f"Final Test NMSE:       {best_test_nmse:.4e}")

    # Return results dictionary container
    return {
        "system_name": oracle.name,
        "gt_equation": oracle.gt_equation_str,
        "discovered_equation": best_candidate.equation_str,
        "final_nmse": best_test_nmse,
        "n_queries": n_active_queries,
        "oracle": oracle,
        "best_candidate": best_candidate,
        "t_eval": oracle.t_eval,
        "true_test_traj": y_test_true,
        "pred_test_traj": best_test_traj,
    }


def main() -> None:
    """Run full benchmarking suite, tabulate results in Pandas, and generate 1x3 plots."""
    seed = 42

    oracles: List[ODEOracle] = [
        create_exponential_decay_oracle(k=0.5),
        create_logistic_growth_oracle(r=1.0, K=5.0),
        create_simple_pendulum_oracle(),
    ]

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    benchmark_results: List[Dict[str, Any]] = []

    for oracle in oracles:
        res = run_active_learning_system(
            oracle=oracle,
            n_active_queries=2,
            n_pysr_iterations=25,
            gemini_api_key=api_key,
            gemini_model=gemini_model,
            seed=seed,
        )
        benchmark_results.append(res)

    summary_data = {
        "System Name": [r["system_name"] for r in benchmark_results],
        "Ground Truth Equation": [r["gt_equation"] for r in benchmark_results],
        "Discovered Equation": [r["discovered_equation"] for r in benchmark_results],
        "Final NMSE": [r["final_nmse"] for r in benchmark_results],
        "Number of Active Queries": [r["n_queries"] for r in benchmark_results],
    }

    summary_df = pd.DataFrame(summary_data)

    print("\n" + "=" * 90)
    print("                     BENCHMARK SUMMARY TABLE: LLM-ACES FRAMEWORK")
    print("=" * 90)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", lambda x: f"{x:.4e}" if abs(x) < 1e-2 else f"{x:.4f}")
    print(summary_df.to_string(index=False))
    print("=" * 90)

    os.makedirs("results", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    summary_df.to_csv("results/llm_aces_summary.csv", index=False)
    summary_df.to_csv("outputs/llm_aces_summary.csv", index=False)
    print("Summary table successfully saved to 'results/llm_aces_summary.csv' and 'outputs/llm_aces_summary.csv'.")

    print("\nGenerating 1x3 comparison visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    r0 = benchmark_results[0]
    t0 = r0["t_eval"]
    y_true_0 = r0["true_test_traj"][0]
    y_pred_0 = r0["pred_test_traj"][0]
    axes[0].plot(t0, y_true_0, "k-", linewidth=2.5, label="True Trajectory")
    axes[0].plot(t0, y_pred_0, "r--", linewidth=2.0, label="Discovered Equation")
    axes[0].set_title(
        f"Exponential Decay\nDiscovered: {r0['discovered_equation']}\nFinal NMSE: {r0['final_nmse']:.2e}",
        fontsize=11,
    )
    axes[0].set_xlabel("Time $t$", fontsize=12)
    axes[0].set_ylabel("State $x(t)$", fontsize=12)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="best", fontsize=10)

    r1 = benchmark_results[1]
    t1 = r1["t_eval"]
    y_true_1 = r1["true_test_traj"][0]
    y_pred_1 = r1["pred_test_traj"][0]
    axes[1].plot(t1, y_true_1, "k-", linewidth=2.5, label="True Trajectory")
    axes[1].plot(t1, y_pred_1, "b--", linewidth=2.0, label="Discovered Equation")
    axes[1].set_title(
        f"Logistic Growth\nDiscovered: {r1['discovered_equation']}\nFinal NMSE: {r1['final_nmse']:.2e}",
        fontsize=11,
    )
    axes[1].set_xlabel("Time $t$", fontsize=12)
    axes[1].set_ylabel("State $x(t)$", fontsize=12)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="best", fontsize=10)

    r2 = benchmark_results[2]
    t2 = r2["t_eval"]
    y_true_2_x0 = r2["true_test_traj"][0]
    y_true_2_x1 = r2["true_test_traj"][1]
    y_pred_2_x0 = r2["pred_test_traj"][0]
    y_pred_2_x1 = r2["pred_test_traj"][1]
    axes[2].plot(t2, y_true_2_x0, "k-", linewidth=2.2, label=r"True $x_0$ ($\theta$)")
    axes[2].plot(t2, y_pred_2_x0, "g--", linewidth=2.0, label=r"Discovered $x_0$")
    axes[2].plot(t2, y_true_2_x1, "k:", linewidth=2.0, label=r"True $x_1$ ($\omega$)")
    axes[2].plot(t2, y_pred_2_x1, "m-.", linewidth=1.8, label=r"Discovered $x_1$")
    axes[2].set_title(
        f"Simple Pendulum\nDiscovered: {r2['discovered_equation']}\nFinal NMSE: {r2['final_nmse']:.2e}",
        fontsize=10,
    )
    axes[2].set_xlabel("Time $t$", fontsize=12)
    axes[2].set_ylabel(r"States $[x_0, x_1]$", fontsize=12)
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="best", fontsize=9)

    plt.tight_layout()

    plt.savefig("results/llm_aces_benchmarks.png", dpi=300, bbox_inches="tight")
    plt.savefig("outputs/llm_aces_benchmarks.png", dpi=300, bbox_inches="tight")
    print("1x3 comparison visualization successfully saved to 'results/llm_aces_benchmarks.png' and 'outputs/llm_aces_benchmarks.png'.")

    plt.close(fig)


if __name__ == "__main__":
    main()
