from typing import List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from beam_matching.optics import (
    DEFAULT_TWISS_TARGET,
    BeamlineConfig,
    QuadrupoleSettings,
    TwissParams,
    TwissParamsXY,
    propagate_through_beamline_x,
    propagate_through_beamline_y,
)

# Цветовая палитра
_C = dict(
    beta_x="#f43f5e",  # rose-500
    beta_y="#3b82f6",  # blue-500
    alpha_x="#fb923c",  # orange-400
    alpha_y="#06b6d4",  # cyan-500
    quad_ref="#94a3b8",  # slate-400
    target_x="#fda4af",  # rose-300  (пунктир цели)
    target_y="#93c5fd",  # blue-300
    zero="#e2e8f0",  # slate-200
    grid="#f8fafc",  # slate-50
    bg="#ffffff",
)


def _add_quad_bands(
    fig,
    quad_positions: List[float],
    quad_length: float,
    y_max: float,
    row: int,
    col: int = 1,
):
    """Полупрозрачные полосы на месте квадруполей + подписи."""
    for i, pos in enumerate(quad_positions):
        # Серая полоса
        fig.add_shape(
            type="rect",
            x0=pos,
            x1=pos + quad_length,
            y0=0,
            y1=y_max,
            fillcolor="rgba(148, 163, 184, 0.12)",
            line=dict(width=0),
            layer="below",
            row=row,
            col=col,
        )
        # Тонкая линия по центру квадруполя
        fig.add_shape(
            type="line",
            x0=pos + quad_length / 2,
            x1=pos + quad_length / 2,
            y0=0,
            y1=y_max,
            line=dict(color=_C["quad_ref"], width=1, dash="dot"),
            layer="below",
            row=row,
            col=col,
        )
        if row == 1:
            fig.add_annotation(
                x=pos + quad_length / 2,
                y=y_max,
                text=f"<b>Q{i + 1}</b>",
                showarrow=False,
                font=dict(size=10, color="#64748b"),
                yanchor="bottom",
                yshift=4,
                row=row,
                col=col,
            )


def create_beta_plot(
    twiss_in: TwissParamsXY,
    config: BeamlineConfig,
    quads: QuadrupoleSettings,
    twiss_target: TwissParamsXY = None,
) -> go.Figure:
    """
    Два связанных subplot'а:
      • Верхний  — β-функции (βx, βy) + горизонтальные линии целевых значений
      • Нижний   — α-функции (αx, αy) + нулевая линия

    Квадруполи обозначены полупрозрачными полосами.
    Легенда — снизу (горизонтальная).
    """
    if twiss_target is None:
        twiss_target = DEFAULT_TWISS_TARGET

    _, history_x = propagate_through_beamline_x(twiss_in.x, config, quads)
    _, history_y = propagate_through_beamline_y(twiss_in.y, config, quads)

    s = [h[0] for h in history_x]
    beta_x = [h[1].beta for h in history_x]
    beta_y = [h[1].beta for h in history_y]
    alph_x = [h[1].alpha for h in history_x]
    alph_y = [h[1].alpha for h in history_y]

    n = config.n_quads
    ql = config.quad_length
    dl = config.drift_length
    quad_positions = [i * (ql + dl) for i in range(n)]

    beta_max = (
        max(max(beta_x), max(beta_y), twiss_target.x.beta, twiss_target.y.beta) * 1.15
    )
    alph_abs = (
        max(max(abs(v) for v in alph_x), max(abs(v) for v in alph_y)) * 1.2 or 1.0
    )

    # -----------------------------------------------------------------------
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.06,
    )

    # === β-функции ===
    fig.add_trace(
        go.Scatter(
            x=s,
            y=beta_x,
            name="βx",
            mode="lines",
            line=dict(color=_C["beta_x"], width=2.5),
            legendgroup="beta_x",
            hovertemplate="s = %{x:.3f} м<br>βx = %{y:.4f} м<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=s,
            y=beta_y,
            name="βy",
            mode="lines",
            line=dict(color=_C["beta_y"], width=2.5),
            legendgroup="beta_y",
            hovertemplate="s = %{x:.3f} м<br>βy = %{y:.4f} м<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Целевые β — пунктир + аннотация
    s_max = max(s)
    for val, color, name, leg in [
        (
            twiss_target.x.beta,
            _C["target_x"],
            f"βx* = {twiss_target.x.beta:.2f} м",
            "beta_x",
        ),
        (
            twiss_target.y.beta,
            _C["target_y"],
            f"βy* = {twiss_target.y.beta:.2f} м",
            "beta_y",
        ),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[0, s_max],
                y=[val, val],
                name=name,
                mode="lines",
                line=dict(color=color, width=1.5, dash="dash"),
                legendgroup=leg,
                showlegend=True,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    # Полосы квадруполей — верхний subplot
    _add_quad_bands(fig, quad_positions, ql, beta_max, row=1)

    # === α-функции ===
    fig.add_trace(
        go.Scatter(
            x=s,
            y=alph_x,
            name="αx",
            mode="lines",
            line=dict(color=_C["alpha_x"], width=2, dash="solid"),
            legendgroup="alpha_x",
            hovertemplate="s = %{x:.3f} м<br>αx = %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=s,
            y=alph_y,
            name="αy",
            mode="lines",
            line=dict(color=_C["alpha_y"], width=2, dash="solid"),
            legendgroup="alpha_y",
            hovertemplate="s = %{x:.3f} м<br>αy = %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # α = 0 — целевое (горизонтальная линия)
    fig.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dot"), row=2, col=1)

    # Полосы квадруполей — нижний subplot
    _add_quad_bands(fig, quad_positions, ql, alph_abs, row=2)
    _add_quad_bands(fig, quad_positions, ql, -alph_abs, row=2)

    # -----------------------------------------------------------------------
    # Общий layout
    fig.update_layout(
        title=dict(
            text="<b>Оптические функции вдоль пучкового канала</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=17, color="#0f172a"),
        ),
        # Легенда — снизу, горизонтальная
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(size=12),
            itemsizing="constant",
            tracegroupgap=8,
        ),
        margin=dict(l=70, r=80, t=70, b=100),
        plot_bgcolor=_C["bg"],
        paper_bgcolor=_C["bg"],
        autosize=True,
        height=560,
        hovermode="x unified",
        font=dict(family="Inter, Arial, sans-serif", size=11),
    )

    # Оси β-subplot
    fig.update_xaxes(
        showgrid=True,
        gridcolor=_C["grid"],
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor="#e2e8f0",
        mirror=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title=dict(text="β (м)", font=dict(size=12)),
        range=[0, beta_max],
        showgrid=True,
        gridcolor=_C["grid"],
        zeroline=True,
        zerolinecolor=_C["zero"],
        zerolinewidth=1,
        showline=True,
        linecolor="#e2e8f0",
        mirror=True,
        row=1,
        col=1,
    )

    # Оси α-subplot
    fig.update_xaxes(
        title=dict(text="s (м)", font=dict(size=12)),
        showgrid=True,
        gridcolor=_C["grid"],
        zeroline=False,
        showline=True,
        linecolor="#e2e8f0",
        mirror=True,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title=dict(text="α", font=dict(size=12)),
        range=[-alph_abs, alph_abs],
        showgrid=True,
        gridcolor=_C["grid"],
        zeroline=True,
        zerolinecolor=_C["zero"],
        zerolinewidth=1,
        showline=True,
        linecolor="#e2e8f0",
        mirror=True,
        row=2,
        col=1,
    )

    return fig


if __name__ == "__main__":
    from optics import DEFAULT_CONFIG, DEFAULT_TWISS_IN

    quads = QuadrupoleSettings(k1=0.5, k2=-0.3, k3=0.4, k4=-0.2)
    fig = create_beta_plot(DEFAULT_TWISS_IN, DEFAULT_CONFIG, quads)
    fig.show()
    print("OK")
