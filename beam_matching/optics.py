from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize

# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------


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
    """Настройки квадруполей — поддерживает 4 или 5 линз."""

    k1: float
    k2: float
    k3: float
    k4: float
    k5: float = 0.0  # Используется только в режиме 5 квадруполей

    def to_list(self, n_quads: int) -> List[float]:
        return [self.k1, self.k2, self.k3, self.k4, self.k5][:n_quads]


@dataclass
class BeamlineConfig:
    drift_length: float
    emit_x: float
    emit_y: float
    quad_length: float = 0.1
    n_quads: int = 4  # 4 или 5


# ---------------------------------------------------------------------------
# Матрицы переноса (кэшируются через lru_cache по числовым ключам)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _drift_matrix_cached(L_key: int) -> np.ndarray:
    """L_key = round(L * 1e9) — целочисленный ключ для кэша."""
    L = L_key * 1e-9
    return np.array([[1.0, L], [0.0, 1.0]])


@lru_cache(maxsize=4096)
def _quad_matrix_cached(k_key: int, L_key: int) -> np.ndarray:
    """Матрица толстого квадруполя. Ключи — целые (round * 1e9)."""
    k = k_key * 1e-9
    L = L_key * 1e-9
    if abs(k) < 1e-10:
        return np.array([[1.0, L], [0.0, 1.0]])
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


def drift_matrix(L: float) -> np.ndarray:
    return _drift_matrix_cached(round(L * 1e9))


def quad_matrix_thick(k: float, L: float) -> np.ndarray:
    return _quad_matrix_cached(round(k * 1e9), round(L * 1e9))


def quad_matrix_thick_defoc(k: float, L: float) -> np.ndarray:
    return _quad_matrix_cached(round(-k * 1e9), round(L * 1e9))


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def calculate_gamma(beta: float, alpha: float) -> float:
    return (1.0 + alpha * alpha) / beta


def propagate_twiss(twiss: TwissParams, M: np.ndarray) -> TwissParams:
    gamma = calculate_gamma(twiss.beta, twiss.alpha)
    m11, m12 = M[0, 0], M[0, 1]
    m21, m22 = M[1, 0], M[1, 1]
    beta2 = m11 * m11 * twiss.beta - 2.0 * m11 * m12 * twiss.alpha + m12 * m12 * gamma
    alpha2 = (
        -m11 * m21 * twiss.beta
        + (m11 * m22 + m12 * m21) * twiss.alpha
        - m12 * m22 * gamma
    )
    return TwissParams(beta=beta2, alpha=alpha2)


def get_total_length(config: BeamlineConfig) -> float:
    n = config.n_quads
    return n * config.quad_length + (n - 1) * config.drift_length


# ---------------------------------------------------------------------------
# Элементы пучкового канала
# ---------------------------------------------------------------------------


@dataclass
class BeamlineElement:
    type: str
    position: float
    length: float
    k: float | None = None
    label: str | None = None


def create_beamline(
    config: BeamlineConfig, quads: QuadrupoleSettings
) -> List[BeamlineElement]:
    ql = config.quad_length
    dl = config.drift_length
    n = config.n_quads
    ks = quads.to_list(n)
    pos = 0.0
    elements = []
    for i, k in enumerate(ks, start=1):
        elements.append(
            BeamlineElement(type="quad", position=pos, length=ql, k=k, label=f"Q{i}")
        )
        pos += ql
        if i < n:
            elements.append(BeamlineElement(type="drift", position=pos, length=dl))
            pos += dl
    return elements


# ---------------------------------------------------------------------------
# Распространение Твисса — общая функция для X и Y
# ---------------------------------------------------------------------------


def _propagate_plane(
    twiss_in: TwissParams,
    config: BeamlineConfig,
    quads: QuadrupoleSettings,
    defocusing: bool,
    points_per_drift: int = 50,
) -> Tuple[TwissParams, List[Tuple[float, TwissParams]]]:
    """
    Универсальная функция распространения для X (defocusing=False)
    и Y (defocusing=True) плоскостей.
    """
    history: List[Tuple[float, TwissParams]] = []
    tw = TwissParams(beta=twiss_in.beta, alpha=twiss_in.alpha)
    s = 0.0
    ql = config.quad_length
    dl = config.drift_length
    n = config.n_quads
    ks = quads.to_list(n)
    quad_pts = max(5, points_per_drift // 5)

    mat_fn = quad_matrix_thick_defoc if defocusing else quad_matrix_thick

    for i, k in enumerate(ks):
        for j in range(quad_pts + 1):
            ds = ql * j / quad_pts
            history.append((s + ds, propagate_twiss(tw, mat_fn(k, ds))))
        tw = propagate_twiss(tw, mat_fn(k, ql))
        s += ql

        if i < n - 1:
            for j in range(points_per_drift + 1):
                ds = dl * j / points_per_drift
                history.append((s + ds, propagate_twiss(tw, drift_matrix(ds))))
            tw = propagate_twiss(tw, drift_matrix(dl))
            s += dl

    return tw, history


def propagate_through_beamline_x(
    twiss_in: TwissParams,
    config: BeamlineConfig,
    quads: QuadrupoleSettings,
    points_per_drift: int = 50,
) -> Tuple[TwissParams, List[Tuple[float, TwissParams]]]:
    return _propagate_plane(
        twiss_in, config, quads, defocusing=False, points_per_drift=points_per_drift
    )


def propagate_through_beamline_y(
    twiss_in: TwissParams,
    config: BeamlineConfig,
    quads: QuadrupoleSettings,
    points_per_drift: int = 50,
) -> Tuple[TwissParams, List[Tuple[float, TwissParams]]]:
    return _propagate_plane(
        twiss_in, config, quads, defocusing=True, points_per_drift=points_per_drift
    )


# ---------------------------------------------------------------------------
# Функции потерь и согласования
# ---------------------------------------------------------------------------


def mismatch(beta_t, alpha_t, beta_a, alpha_a):
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


# ---------------------------------------------------------------------------
# Оптимизация
# ---------------------------------------------------------------------------


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
    Двухэтапная оптимизация квадруполей (4 или 5 линз).

    Этап 1: differential_evolution — глобальный поиск.
    Этап 2a: Nelder-Mead со штрафом.
    Этап 2б: Nelder-Mead только по mismatch — максимальная точность.

    Настройки DE масштабируются под число степеней свободы, чтобы
    производительность оставалась сопоставимой для 4 и 5 квадруполей.
    """
    PTS = 20
    n = config.n_quads

    def _propagate(x: np.ndarray):
        ks = x[:n]
        q = QuadrupoleSettings(
            k1=ks[0],
            k2=ks[1],
            k3=ks[2],
            k4=ks[3],
            k5=ks[4] if n == 5 else 0.0,
        )
        cfg = (
            BeamlineConfig(
                drift_length=x[n],
                emit_x=config.emit_x,
                emit_y=config.emit_y,
                quad_length=config.quad_length,
                n_quads=n,
            )
            if optimize_drift
            else config
        )
        rx, hx = propagate_through_beamline_x(twiss_in.x, cfg, q, PTS)
        ry, hy = propagate_through_beamline_y(twiss_in.y, cfg, q, PTS)
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

    bounds = [(-10.0, 10.0)] * n  # k1..kN
    if optimize_drift:
        bounds.append((0.3, 3.0))  # drift_length

    # Размер популяции DE: чуть меньше для 5 квадруполей, чтобы сохранить скорость.
    # Формула: 15 + (n-4)*5 → 4 quad=15, 5 quad=20 (доп. степень свободы даёт больше опций)
    popsize = 15 + (n - 4) * 5

    res_global = differential_evolution(
        obj_full,
        bounds,
        maxiter=500,
        tol=1e-14,
        seed=42,
        popsize=popsize,
        workers=1,  # детерминированность; можно workers=-1 для параллельности
    )

    res_local = minimize(
        obj_full,
        res_global.x,
        method="Nelder-Mead",
        options={"xatol": 1e-11, "fatol": 1e-11, "maxiter": 1000},
    )

    res_final = minimize(
        obj_nomatch,
        res_local.x,
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 1000},
    )

    ks = res_final.x[:n]
    L = res_final.x[n] if optimize_drift else config.drift_length

    return {
        "quads": QuadrupoleSettings(
            k1=ks[0],
            k2=ks[1],
            k3=ks[2],
            k4=ks[3],
            k5=ks[4] if n == 5 else 0.0,
        ),
        "drift_length": L,
        "error": res_final.fun,
        "success": res_final.fun < 1e-10,
    }


# ---------------------------------------------------------------------------
# Вспомогательные функции для визуализации
# ---------------------------------------------------------------------------


def generate_phase_space_ellipse(
    twiss: TwissParams, emittance: float, num_points: int = 100
) -> List[Tuple[float, float]]:
    points = []
    sqrt_eps = np.sqrt(emittance)
    sqrt_beta = np.sqrt(twiss.beta)
    for i in range(num_points):
        theta = 2 * np.pi * i / num_points
        x = sqrt_eps * sqrt_beta * np.cos(theta)
        xp = sqrt_eps * (
            -twiss.alpha / sqrt_beta * np.cos(theta) + np.sin(theta) / sqrt_beta
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


# ---------------------------------------------------------------------------
# Дефолтные значения
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = BeamlineConfig(
    drift_length=1.0,
    emit_x=10e-9,
    emit_y=2e-9,
    quad_length=0.1,
    n_quads=4,
)

DEFAULT_TWISS_IN = TwissParamsXY(
    x=TwissParams(beta=5.0, alpha=-0.5),
    y=TwissParams(beta=2.5, alpha=0.3),
)

DEFAULT_TWISS_TARGET = TwissParamsXY(
    x=TwissParams(beta=8.0, alpha=0.0),
    y=TwissParams(beta=4.0, alpha=0.0),
)
