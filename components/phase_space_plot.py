from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go

from beam_matching.optics import (
    DEFAULT_TWISS_TARGET,
    BeamlineConfig,
    QuadrupoleSettings,
    TwissParams,
    TwissParamsXY,
    generate_phase_space_ellipse,
    propagate_through_beamline_x,
    propagate_through_beamline_y,
)


def _create_phase_space_panel(
    title: str,
    input_ellipse: List[Tuple[float, float]],
    target_ellipse: List[Tuple[float, float]],
    current_ellipse: List[Tuple[float, float]],
    x_label: str,
    xp_label: str,
) -> go.Figure:
    """
    Панель фазового пространства с автоматическим масштабированием осей.

    Масштаб подбирается индивидуально под каждую плоскость (X или Y),
    так что эллипсы всегда занимают большую часть области графика.
    """

    def _coords(ellipse):
        xs = [p[0] for p in ellipse] + [ellipse[0][0]]
        ys = [p[1] for p in ellipse] + [ellipse[0][1]]
        return xs, ys

    x_in, xp_in = _coords(input_ellipse)
    x_target, xp_target = _coords(target_ellipse)
    x_current, xp_current = _coords(current_ellipse)

    # --- Автосайз: берём максимум по всем трём эллипсам этой плоскости ---
    all_x = x_in + x_target + x_current
    all_xp = xp_in + xp_target + xp_current

    x_max = max(abs(v) for v in all_x) or 1.0
    xp_max = max(abs(v) for v in all_xp) or 1.0

    # Небольшой отступ (15 %) чтобы эллипсы не упирались в края
    x_pad = x_max * 1.15
    xp_pad = xp_max * 1.15

    fig = go.Figure()

    # Входной эллипс — оранжевый пунктир
    fig.add_trace(
        go.Scatter(
            x=x_in,
            y=xp_in,
            mode="lines",
            name="Вход",
            line=dict(color="#f59e0b", width=2, dash="dash"),
            hoverinfo="skip",
        )
    )

    # Целевой эллипс — зелёный точечный
    fig.add_trace(
        go.Scatter(
            x=x_target,
            y=xp_target,
            mode="lines",
            name="Цель",
            line=dict(color="#10b981", width=2, dash="dot"),
            hoverinfo="skip",
        )
    )

    # Текущая позиция — красный сплошной (заливка)
    fig.add_trace(
        go.Scatter(
            x=x_current,
            y=xp_current,
            mode="lines",
            name="Текущая позиция",
            fill="toself",
            fillcolor="rgba(244, 63, 94, 0.08)",
            line=dict(color="#f43f5e", width=2.5),
            hovertemplate=f"{x_label}: %{{x:.3g}}<br>{xp_label}: %{{y:.3g}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=14, color="#1e293b"),
        ),
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=12)),
            range=[-x_pad, x_pad],
            showgrid=True,
            gridcolor="#f1f5f9",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="#cbd5e1",
            zerolinewidth=1.5,
            tickformat=".2g",
            showline=True,
            linecolor="#e2e8f0",
            mirror=True,
        ),
        yaxis=dict(
            title=dict(text=xp_label, font=dict(size=12)),
            range=[-xp_pad, xp_pad],
            showgrid=True,
            gridcolor="#f1f5f9",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="#cbd5e1",
            zerolinewidth=1.5,
            tickformat=".2g",
            showline=True,
            linecolor="#e2e8f0",
            mirror=True,
            scaleanchor="x",  # квадратный аспект — эллипс не деформируется
            scaleratio=1,
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=70, r=30, t=70, b=60),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        autosize=True,  # растягивается под контейнер Streamlit
        height=420,
        font=dict(family="Inter, Arial, sans-serif", size=11),
    )

    return fig


def create_phase_space_plot(
    twiss_in: TwissParamsXY,
    config: BeamlineConfig,
    quads: QuadrupoleSettings,
    twiss_target: TwissParamsXY = None,
    selected_position_index: int = 0,
    points_per_drift: int = 50,
) -> Tuple[go.Figure, go.Figure]:
    """
    Фазовые портреты для плоскостей X и Y.

    Позиции (selected_position_index):
        0              — вход
        1 … n_quads    — после Q_i
        n_quads + 1    — выход (конец)
    """
    if twiss_target is None:
        twiss_target = DEFAULT_TWISS_TARGET

    n = config.n_quads

    _, history_x = propagate_through_beamline_x(
        twiss_in.x, config, quads, points_per_drift
    )
    _, history_y = propagate_through_beamline_y(
        twiss_in.y, config, quads, points_per_drift
    )

    def _twiss_at(pos_idx: int, history: List) -> TwissParams:
        """
        Возвращает параметры Твисса в нужной точке истории.

        pos_idx == 0          → начало
        pos_idx == i (1..n)   → конец i-го квадруполя
        pos_idx >= n+1        → конец трассировки
        """
        if pos_idx <= 0:
            return history[0][1]
        if pos_idx > n:
            return history[-1][1]

        # Ищем ближайшую точку истории по s-координате
        # Конец i-го квадруполя: s = i * quad_length + (i-1) * drift_length
        ql = config.quad_length
        dl = config.drift_length
        s_target = pos_idx * ql + (pos_idx - 1) * dl

        # Бинарный поиск по s
        s_vals = [h[0] for h in history]
        idx = int(np.searchsorted(s_vals, s_target))
        idx = min(idx, len(history) - 1)
        return history[idx][1]

    twiss_x = _twiss_at(selected_position_index, history_x)
    twiss_y = _twiss_at(selected_position_index, history_y)

    # Эллипсы X
    ell_x_in = generate_phase_space_ellipse(twiss_in.x, config.emit_x, 120)
    ell_x_target = generate_phase_space_ellipse(twiss_target.x, config.emit_x, 120)
    ell_x_current = generate_phase_space_ellipse(twiss_x, config.emit_x, 120)

    # Эллипсы Y
    ell_y_in = generate_phase_space_ellipse(twiss_in.y, config.emit_y, 120)
    ell_y_target = generate_phase_space_ellipse(twiss_target.y, config.emit_y, 120)
    ell_y_current = generate_phase_space_ellipse(twiss_y, config.emit_y, 120)

    fig_x = _create_phase_space_panel(
        title="Фазовое пространство — плоскость X",
        input_ellipse=ell_x_in,
        target_ellipse=ell_x_target,
        current_ellipse=ell_x_current,
        x_label="x (м)",
        xp_label="x′ (рад)",
    )

    fig_y = _create_phase_space_panel(
        title="Фазовое пространство — плоскость Y",
        input_ellipse=ell_y_in,
        target_ellipse=ell_y_target,
        current_ellipse=ell_y_current,
        x_label="y (м)",
        xp_label="y′ (рад)",
    )

    return fig_x, fig_y


if __name__ == "__main__":
    from optics import DEFAULT_CONFIG, DEFAULT_TWISS_IN

    quads = QuadrupoleSettings(k1=0.5, k2=-0.3, k3=0.4, k4=-0.2)
    fig_x, fig_y = create_phase_space_plot(DEFAULT_TWISS_IN, DEFAULT_CONFIG, quads)
    fig_x.show()
    fig_y.show()
    print("OK")
