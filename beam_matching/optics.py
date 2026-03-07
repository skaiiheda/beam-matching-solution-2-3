from dataclasses import dataclass, field
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
    # Физическая длина каждого квадруполя (м).
    # По умолчанию 0.1 м — реалистичное значение для согласующей секции.
    quad_length: float = 0.1


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


def quad_matrix_thick(k: float, L: float) -> np.ndarray:
    """
    Матрица переноса толстого квадруполя длиной L с градиентом k (м⁻²).

    Для k > 0 (фокусировка в X):
        M = [[cos(phi),        sin(phi)/sqrt(k)],
             [-sqrt(k)*sin(phi), cos(phi)      ]]
        где phi = sqrt(k) * L

    Для k < 0 (дефокусировка в X):
        M = [[cosh(phi),       sinh(phi)/sqrt(|k|)],
             [sqrt(|k|)*sinh(phi), cosh(phi)      ]]
        где phi = sqrt(|k|) * L

    При k ≈ 0 возвращает матрицу дрейфа (предел).
    """
    if abs(k) < 1e-10:
        return drift_matrix(L)

    if k > 0:
        phi = np.sqrt(k) * L
        sqk = np.sqrt(k)
        return np.array(
            [
                [np.cos(phi), np.sin(phi) / sqk],
                [-sqk * np.sin(phi), np.cos(phi)],
            ]
        )
    else:
        phi = np.sqrt(-k) * L
        sqk = np.sqrt(-k)
        return np.array(
            [
                [np.cosh(phi), np.sinh(phi) / sqk],
                [sqk * np.sinh(phi), np.cosh(phi)],
            ]
        )


def quad_matrix_thick_defoc(k: float, L: float) -> np.ndarray:
    """Матрица для дефокусирующей плоскости (знак k инвертирован)."""
    return quad_matrix_thick(-k, L)


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
    ql = config.quad_length
    dl = config.drift_length
    pos = 0.0
    elements = []

    for i, k in enumerate([quads.k1, quads.k2, quads.k3, quads.k4], start=1):
        elements.append(
            BeamlineElement(type="quad", position=pos, length=ql, k=k, label=f"Q{i}")
        )
        pos += ql
        if i < 4:
            elements.append(BeamlineElement(type="drift", position=pos, length=dl))
            pos += dl

    return elements


def get_total_length(config: BeamlineConfig) -> float:
    return 4 * config.quad_length + 3 * config.drift_length


def propagate_through_beamline_x(
    twiss_in: TwissParams,
    config: BeamlineConfig,
    quads: QuadrupoleSettings,
    points_per_drift: int = 50,
) -> Tuple[TwissParams, List[Tuple[float, TwissParams]]]:
    """
    Распространение в фокусирующей плоскости (X) с толстыми квадруполями.
    Возвращает (твисс на выходе, история[(s, TwissParams)]).
    """
    history: List[Tuple[float, TwissParams]] = []
    tw = TwissParams(beta=twiss_in.beta, alpha=twiss_in.alpha)
    s = 0.0
    ql = config.quad_length
    dl = config.drift_length
    ks = [quads.k1, quads.k2, quads.k3, quads.k4]
    quad_pts = max(5, points_per_drift // 5)

    for i, k in enumerate(ks):
        # --- квадруполь ---
        for j in range(quad_pts + 1):
            ds = ql * j / quad_pts
            history.append((s + ds, propagate_twiss(tw, quad_matrix_thick(k, ds))))
        tw = propagate_twiss(tw, quad_matrix_thick(k, ql))
        s += ql

        # --- дрейф (кроме последнего элемента) ---
        if i < 3:
            for j in range(points_per_drift + 1):
                ds = dl * j / points_per_drift
                history.append((s + ds, propagate_twiss(tw, drift_matrix(ds))))
            tw = propagate_twiss(tw, drift_matrix(dl))
            s += dl

    return tw, history


def propagate_through_beamline_y(
    twiss_in: TwissParams,
    config: BeamlineConfig,
    quads: QuadrupoleSettings,
    points_per_drift: int = 50,
) -> Tuple[TwissParams, List[Tuple[float, TwissParams]]]:
    """
    Распространение в дефокусирующей плоскости (Y) — знаки k инвертированы.
    """
    history: List[Tuple[float, TwissParams]] = []
    tw = TwissParams(beta=twiss_in.beta, alpha=twiss_in.alpha)
    s = 0.0
    ql = config.quad_length
    dl = config.drift_length
    ks = [quads.k1, quads.k2, quads.k3, quads.k4]
    quad_pts = max(5, points_per_drift // 5)

    for i, k in enumerate(ks):
        # --- квадруполь (Y: дефокусирующий) ---
        for j in range(quad_pts + 1):
            ds = ql * j / quad_pts
            history.append(
                (s + ds, propagate_twiss(tw, quad_matrix_thick_defoc(k, ds)))
            )
        tw = propagate_twiss(tw, quad_matrix_thick_defoc(k, ql))
        s += ql

        # --- дрейф ---
        if i < 3:
            for j in range(points_per_drift + 1):
                ds = dl * j / points_per_drift
                history.append((s + ds, propagate_twiss(tw, drift_matrix(ds))))
            tw = propagate_twiss(tw, drift_matrix(dl))
            s += dl

    return tw, history


def mismatch(beta_t, alpha_t, beta_a, alpha_a):
    """
    Параметр несогласования Курана–Тенга.
    Равен 0 при полном согласовании, > 0 иначе.
    """
    gamma_t = (1 + alpha_t**2) / beta_t
    gamma_a = (1 + alpha_a**2) / beta_a
    return 0.5 * (beta_t * gamma_a - 2 * alpha_t * alpha_a + gamma_t * beta_a) - 1


def loss(
    twiss_out,
    twiss_target,
    twiss_history_x,
    twiss_history_y,
    use_penalty=False,
    beta_limit=6.0,
    penalty_weight=0.1,
):
    """
    Функция потерь = mismatch²(X) + mismatch²(Y) + штраф за β-пики.

    Штраф нормирован: ox = max(0, beta_max - beta_limit) / beta_limit,
    поэтому при beta_max = 2·beta_limit вклад штрафа равен 1.0 *penalty_weight,
    что сопоставимо с единичным несогласованием.
    """
    Mx = mismatch(
        twiss_target.x.beta, twiss_target.x.alpha, twiss_out.x.beta, twiss_out.x.alpha
    )
    My = mismatch(
        twiss_target.y.beta, twiss_target.y.alpha, twiss_out.y.beta, twiss_out.y.alpha
    )

    penalty = 0.0
    if use_penalty and twiss_history_x and twiss_history_y:
        beta_x_max = max(t.beta for _, t in twiss_history_x)
        beta_y_max = max(t.beta for _, t in twiss_history_y)
        ox = max(0.0, beta_x_max - beta_limit) / beta_limit
        oy = max(0.0, beta_y_max - beta_limit) / beta_limit
        penalty = ox**2 + oy**2

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
    optimize_drift: bool = True,
    use_penalty: bool = True,
    beta_limit: float = 6.0,
    penalty_weight: float = 0.1,
) -> dict:
    """
    Двухэтапная оптимизация квадруполей.

    Этап 1: differential_evolution с штрафом за β-пики.
      Глобальный поиск ищет решение, которое одновременно согласовано
      И имеет умеренные β-максимумы.

    Этап 2: Nelder-Mead — сначала со штрафом, затем чистый mismatch,
      для максимальной точности согласования.
    """
    PTS = 20

    def _propagate(x: np.ndarray):
        quads = QuadrupoleSettings(k1=x[0], k2=x[1], k3=x[2], k4=x[3])
        cfg = (
            BeamlineConfig(
                drift_length=x[4],
                emit_x=config.emit_x,
                emit_y=config.emit_y,
                quad_length=config.quad_length,
            )
            if optimize_drift
            else config
        )
        rx, hx = propagate_through_beamline_x(twiss_in.x, cfg, quads, PTS)
        ry, hy = propagate_through_beamline_y(twiss_in.y, cfg, quads, PTS)
        return TwissParamsXY(x=rx, y=ry), hx, hy

    def obj_full(x):
        twiss_out, hx, hy = _propagate(x)
        return loss(
            twiss_out,
            twiss_target,
            hx,
            hy,
            use_penalty=use_penalty,
            beta_limit=beta_limit,
            penalty_weight=penalty_weight,
        )

    def obj_nomatch(x):
        twiss_out, hx, hy = _propagate(x)
        return loss(twiss_out, twiss_target, hx, hy, use_penalty=False)

    bounds = [
        (-10.0, 10.0),  # k1
        (-10.0, 10.0),  # k2
        (-10.0, 10.0),  # k3
        (-10.0, 10.0),  # k4
        *([(0.3, 3.0)] if optimize_drift else []),  # drift_length
    ]

    # Этап 1: глобальный поиск
    res_global = differential_evolution(
        obj_full,
        bounds,
        maxiter=3000,
        tol=1e-14,
        seed=42,
        popsize=25,
    )

    # Этап 2а: локальная доводка со штрафом
    res_local = minimize(
        obj_full,
        res_global.x,
        method="Nelder-Mead",
        options={"xatol": 1e-11, "fatol": 1e-11, "maxiter": 200000},
    )

    # Этап 2б: финальная доводка только по mismatch
    res_final = minimize(
        obj_nomatch,
        res_local.x,
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 200000},
    )

    if optimize_drift:
        k1, k2, k3, k4, L = res_final.x
    else:
        k1, k2, k3, k4 = res_final.x
        L = config.drift_length

    return {
        "quads": QuadrupoleSettings(k1=k1, k2=k2, k3=k3, k4=k4),
        "drift_length": L,
        "error": res_final.fun,
        "success": res_final.fun < 1e-10,
    }


def generate_phase_space_ellipse(
    twiss: TwissParams, emittance: float, num_points: int = 100
) -> List[Tuple[float, float]]:
    points = []
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


DEFAULT_CONFIG = BeamlineConfig(
    drift_length=1.0,
    emit_x=10e-9,
    emit_y=2e-9,
    quad_length=0.1,
)

DEFAULT_TWISS_IN = TwissParamsXY(
    x=TwissParams(beta=5.0, alpha=-0.5), y=TwissParams(beta=2.5, alpha=0.3)
)

DEFAULT_TWISS_TARGET = TwissParamsXY(
    x=TwissParams(beta=8.0, alpha=0.0), y=TwissParams(beta=4.0, alpha=0.0)
)
