from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize


@dataclass
class TwissParams:
    beta: float
    alpha: float


@dataclass
class TwissParamsXY:
    x: TwissParams
    y: TwissParams


@dataclass
class QuadrupoleSettings:
    k1: float
    k2: float
    k3: float
    k4: float


@dataclass
class BeamlineConfig:
    drift_length: float
    emit_x: float
    emit_y: float


@dataclass
class BeamlineElement:
    type: str
    position: float
    length: float
    k: float | None = None
    label: str | None = None


def calculate_gamma(beta: float, alpha: float) -> float:
    return (1 + alpha * alpha) / beta


def drift_matrix(L: float) -> np.ndarray:
    return np.array([[1, L], [0, 1]])


def quad_matrix(k: float) -> np.ndarray:
    return np.array([[1, 0], [-k, 1]])


def multiply_matrices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


def propagate_twiss(twiss: TwissParams, M: np.ndarray) -> TwissParams:
    gamma = calculate_gamma(twiss.beta, twiss.alpha)
    m11, m12 = M[0, 0], M[0, 1]
    m21, m22 = M[1, 0], M[1, 1]

    beta2 = m11 * m11 * twiss.beta - 2 * m11 * m12 * twiss.alpha + m12 * m12 * gamma
    alpha2 = (
        -m11 * m21 * twiss.beta
        + (m11 * m22 + m12 * m21) * twiss.alpha
        - m12 * m22 * gamma
    )

    return TwissParams(beta=beta2, alpha=alpha2)


def create_beamline(
    config: BeamlineConfig, quads: QuadrupoleSettings
) -> List[BeamlineElement]:
    return [
        BeamlineElement(type="quad", position=0, length=0.05, k=quads.k1, label="Q1"),
        BeamlineElement(type="drift", position=0.05, length=config.drift_length),
        BeamlineElement(
            type="quad",
            position=0.05 + config.drift_length,
            length=0.05,
            k=quads.k2,
            label="Q2",
        ),
        BeamlineElement(
            type="drift",
            position=0.05 + config.drift_length + 0.05,
            length=config.drift_length,
        ),
        BeamlineElement(
            type="quad",
            position=0.05 + config.drift_length + 0.05 + config.drift_length,
            length=0.05,
            k=quads.k3,
            label="Q3",
        ),
        BeamlineElement(
            type="drift",
            position=0.05 + config.drift_length + 0.05 + config.drift_length + 0.05,
            length=config.drift_length,
        ),
        BeamlineElement(
            type="quad",
            position=0.05
            + config.drift_length
            + 0.05
            + config.drift_length
            + 0.05
            + config.drift_length,
            length=0.05,
            k=quads.k4,
            label="Q4",
        ),
    ]


def get_total_length(config: BeamlineConfig) -> float:
    return 3 * config.drift_length


def propagate_through_beamline_x(
    twiss_in: TwissParams,
    config: BeamlineConfig,
    quads: QuadrupoleSettings,
    points_per_drift: int = 50,
) -> Tuple[TwissParams, List[Tuple[float, TwissParams]]]:
    history = []
    current_twiss = TwissParams(beta=twiss_in.beta, alpha=twiss_in.alpha)
    s = 0.0

    # Q1
    history.append((s, TwissParams(beta=current_twiss.beta, alpha=current_twiss.alpha)))
    current_twiss = propagate_twiss(current_twiss, quad_matrix(quads.k1))
    s += 0.001

    # Drift 1
    for i in range(points_per_drift + 1):
        ds = (config.drift_length * i) / points_per_drift
        twiss_at_s = propagate_twiss(current_twiss, drift_matrix(ds))
        history.append((s + ds, twiss_at_s))
    current_twiss = propagate_twiss(current_twiss, drift_matrix(config.drift_length))
    s += config.drift_length

    # Q2
    current_twiss = propagate_twiss(current_twiss, quad_matrix(quads.k2))
    s += 0.001

    # Drift 2
    for i in range(points_per_drift + 1):
        ds = (config.drift_length * i) / points_per_drift
        twiss_at_s = propagate_twiss(current_twiss, drift_matrix(ds))
        history.append((s + ds, twiss_at_s))
    current_twiss = propagate_twiss(current_twiss, drift_matrix(config.drift_length))
    s += config.drift_length

    # Q3
    current_twiss = propagate_twiss(current_twiss, quad_matrix(quads.k3))
    s += 0.001

    # Drift 3
    for i in range(points_per_drift + 1):
        ds = (config.drift_length * i) / points_per_drift
        twiss_at_s = propagate_twiss(current_twiss, drift_matrix(ds))
        history.append((s + ds, twiss_at_s))
    current_twiss = propagate_twiss(current_twiss, drift_matrix(config.drift_length))
    s += config.drift_length

    # Q4
    current_twiss = propagate_twiss(current_twiss, quad_matrix(quads.k4))

    return current_twiss, history


def propagate_through_beamline_y(
    twiss_in: TwissParams,
    config: BeamlineConfig,
    quads: QuadrupoleSettings,
    points_per_drift: int = 50,
) -> Tuple[TwissParams, List[Tuple[float, TwissParams]]]:
    quads_y = QuadrupoleSettings(k1=-quads.k1, k2=-quads.k2, k3=-quads.k3, k4=-quads.k4)
    return propagate_through_beamline_x(twiss_in, config, quads_y, points_per_drift)


def mismatch(beta_t, alpha_t, beta_a, alpha_a):
    """t = target, a = actual"""
    gamma_t = (1 + alpha_t**2) / beta_t
    gamma_a = (1 + alpha_a**2) / beta_a
    return 0.5 * (beta_t * gamma_a - 2 * alpha_t * alpha_a + gamma_t * beta_a) - 1


def loss(twiss_out, twiss_target, twiss_history_x, twiss_history_y, use_penalty=False, beta_limit=10.0, penalty_weight=0.01):
    # Основное условие согласования
    Mx = mismatch(
        twiss_target.x.beta, twiss_target.x.alpha, twiss_out.x.beta, twiss_out.x.alpha
    )
    My = mismatch(
        twiss_target.y.beta, twiss_target.y.alpha, twiss_out.y.beta, twiss_out.y.alpha
    )

    penalty = 0.0
    if use_penalty:
        # Штраф за большие пики β-функции
        beta_x_max = max(t.beta for _, t in twiss_history_x)
        beta_y_max = max(t.beta for _, t in twiss_history_y)

        penalty = (max(0, beta_x_max - beta_limit) / beta_limit) ** 2 + (
            max(0, beta_y_max - beta_limit) / beta_limit
        ) ** 2

    return Mx**2 + My**2 + penalty_weight * penalty


def calculate_matching_error(
    twiss_out: TwissParamsXY, twiss_target: TwissParamsXY
) -> float:
    error_x = (twiss_out.x.beta - twiss_target.x.beta) ** 2 + (
        twiss_out.x.alpha - twiss_target.x.alpha
    ) ** 2
    error_y = (twiss_out.y.beta - twiss_target.y.beta) ** 2 + (
        twiss_out.y.alpha - twiss_target.y.alpha
    ) ** 2
    return error_x + error_y


def optimize_quadrupoles(
    twiss_in: TwissParamsXY,
    twiss_target: TwissParamsXY,
    config: BeamlineConfig,
    optimize_drift: bool = True,  # новый параметр
    use_penalty: bool = False,
    beta_limit: float = 10.0,
    penalty_weight: float = 0.01,
) -> dict:

    def objective(x: np.ndarray) -> float:
        quads = QuadrupoleSettings(k1=x[0], k2=x[1], k3=x[2], k4=x[3])
        cfg = BeamlineConfig(
            drift_length=x[4], emit_x=config.emit_x, emit_y=config.emit_y
        )

        # points_per_drift=10 ускоряет оптимизацию в 5 раз
        result_x, history_x = propagate_through_beamline_x(
            twiss_in.x, cfg, quads, points_per_drift=10
        )
        result_y, history_y = propagate_through_beamline_y(
            twiss_in.y, cfg, quads, points_per_drift=10
        )

        twiss_out = TwissParamsXY(x=result_x, y=result_y)
        Mx = mismatch(
            twiss_target.x.beta,
            twiss_target.x.alpha,
            twiss_out.x.beta,
            twiss_out.x.alpha,
        )
        My = mismatch(
            twiss_target.y.beta,
            twiss_target.y.alpha,
            twiss_out.y.beta,
            twiss_out.y.alpha,
        )

        penalty = 0.0
        if use_penalty:
            beta_x_max = max(t.beta for _, t in history_x)
            beta_y_max = max(t.beta for _, t in history_y)

            penalty = (max(0, beta_x_max - beta_limit) / beta_limit) ** 2 + (
                max(0, beta_y_max - beta_limit) / beta_limit
            ) ** 2

        return Mx**2 + My**2 + penalty_weight * penalty

    # Границы: 4 квадруполя + длина дрейфа
    bounds = [
        (-10.0, 10.0),  # k1
        (-10.0, 10.0),  # k2
        (-10.0, 10.0),  # k3
        (-10.0, 10.0),  # k4
        (0.3, 3.0),  # drift_length
    ]

    # Шаг 1: глобальный поиск
    res_global = differential_evolution(
        objective,
        bounds,
        maxiter=3000,
        tol=1e-14,
        seed=42,
        popsize=25,
    )

    # Шаг 2: локальная доводка
    res_local = minimize(
        objective,
        res_global.x,
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 50000},
    )

    k1, k2, k3, k4, L = res_local.x
    return {
        "quads": QuadrupoleSettings(k1=k1, k2=k2, k3=k3, k4=k4),
        "drift_length": L,
        "error": res_local.fun,
        "success": res_local.fun < 1e-10,
    }


def generate_phase_space_ellipse(
    twiss: TwissParams, emittance: float, num_points: int = 100
) -> List[Tuple[float, float]]:
    points = []
    gamma = calculate_gamma(twiss.beta, twiss.alpha)
    sqrt_eps = np.sqrt(emittance)
    sqrt_beta = np.sqrt(twiss.beta)

    for i in range(num_points):
        theta = (2 * np.pi * i) / num_points
        x = sqrt_eps * sqrt_beta * np.cos(theta)
        xp = sqrt_eps * (
            -twiss.alpha / sqrt_beta * np.cos(theta) + 1 / sqrt_beta * np.sin(theta)
        )
        points.append((x, xp))

    return points


def calculate_envelope(
    twiss_history: List[Tuple[float, TwissParams]], emittance: float
) -> List[Tuple[float, float, float]]:
    return [
        (s, np.sqrt(twiss.beta * emittance), -np.sqrt(twiss.beta * emittance))
        for s, twiss in twiss_history
    ]


DEFAULT_CONFIG = BeamlineConfig(drift_length=1.0, emit_x=10e-9, emit_y=2e-9)

DEFAULT_TWISS_IN = TwissParamsXY(
    x=TwissParams(beta=5.0, alpha=-0.5), y=TwissParams(beta=2.5, alpha=0.3)
)

DEFAULT_TWISS_TARGET = TwissParamsXY(
    x=TwissParams(beta=8.0, alpha=0.0), y=TwissParams(beta=4.0, alpha=0.0)
)
