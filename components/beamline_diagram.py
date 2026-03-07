from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go

from beam_matching.optics import BeamlineConfig, QuadrupoleSettings


def create_beamline_diagram(
    quads: QuadrupoleSettings,
    config: BeamlineConfig,
) -> go.Figure:
    """
    Plotly-визуализация пучкового канала.

    Поддерживает 4 или 5 квадруполей в зависимости от config.n_quads.
    """
    n = config.n_quads
    drift_length = config.drift_length
    k_values = quads.to_list(n)
    labels = [f"Q{i + 1}" for i in range(n)]

    # Позиции квадруполей: Q_i начинается в i * drift_length
    quad_positions = [i * drift_length for i in range(n)]
    total_length = (n - 1) * drift_length

    fig = go.Figure()

    def get_color(k: float) -> str:
        if abs(k) < 0.001:
            return "#9ca3af"
        return "#10b981" if k > 0 else "#f43f5e"

    def get_lens_type(k: float) -> str:
        if abs(k) < 0.001:
            return "нейтральный"
        return "фокусировка" if k > 0 else "дефокусировка"

    bar_width = 0.06 * max(drift_length, 1.0)
    max_k = max(abs(k) for k in k_values) if k_values else 1.0
    scale_height = 2.0 / max(max_k, 0.1)

    for i, (pos, k, label) in enumerate(zip(quad_positions, k_values, labels)):
        color = get_color(k)
        lens_type = get_lens_type(k)
        height = max(abs(k) * scale_height, 0.5)

        quad_x = [
            pos - bar_width / 2,
            pos + bar_width / 2,
            pos + bar_width / 2,
            pos - bar_width / 2,
            pos - bar_width / 2,
        ]
        quad_y = [-height / 2, -height / 2, height / 2, height / 2, -height / 2]

        fig.add_trace(
            go.Scatter(
                x=quad_x,
                y=quad_y,
                fill="toself",
                fillcolor=color,
                line=dict(color=color, width=2),
                mode="lines",
                name=f"{label} ({lens_type})",
                showlegend=False,
                hovertemplate=f"<b>{label}</b><br>k = {k:.4f} м⁻²<br>{lens_type}<extra></extra>",
            )
        )

        fig.add_annotation(
            x=pos,
            y=height / 2 + 0.3,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=13, family="Arial Black"),
            yanchor="bottom",
        )
        fig.add_annotation(
            x=pos,
            y=-height / 2 - 0.25,
            text=f"k={k:.3f}",
            showarrow=False,
            font=dict(size=10, color="#64748b"),
            yanchor="top",
        )

    # Дрейфы между квадруполями
    for i in range(n - 1):
        x0 = quad_positions[i] + bar_width / 2
        x1 = quad_positions[i + 1] - bar_width / 2
        fig.add_shape(
            type="line",
            x0=x0,
            y0=0,
            x1=x1,
            y1=0,
            line=dict(color="#e2e8f0", width=4),
        )
        mid = (quad_positions[i] + quad_positions[i + 1]) / 2
        fig.add_annotation(
            x=mid,
            y=-1.1,
            text=f"L={drift_length:.2f}м",
            showarrow=False,
            font=dict(size=9, color="#94a3b8"),
            yanchor="top",
        )

    # Стрелка направления пучка
    arrow_y = 1.6
    fig.add_annotation(
        x=total_length / 2,
        y=arrow_y,
        text="<b>Направление пучка →</b>",
        showarrow=False,
        font=dict(size=12, color="#64748b"),
        yanchor="bottom",
    )

    # Легенда: цветные маркеры
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="#10b981", size=14, symbol="square"),
            name="Фокусировка X (k > 0)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="#f43f5e", size=14, symbol="square"),
            name="Дефокусировка X (k < 0)",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Компоновка пучкового канала ({n} квадруполя)</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=17),
        ),
        xaxis=dict(
            title=dict(text="Позиция вдоль пучкового канала (м)", standoff=10),
            range=[-0.3 * drift_length, total_length + 0.3 * drift_length],
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            tickmode="linear",
            tick0=0,
            dtick=drift_length,
        ),
        yaxis=dict(
            title=dict(text="Сила квадруполя (норм.)", standoff=10),
            range=[-2.0, 2.2],
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=True,
            zerolinecolor="#94a3b8",
            zerolinewidth=1,
            showticklabels=False,
            fixedrange=True,
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        margin=dict(l=60, r=60, t=80, b=80),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=400,
        font=dict(family="Arial", size=11),
    )
    return fig


if __name__ == "__main__":
    from beam_matching.optics import DEFAULT_CONFIG

    quads = QuadrupoleSettings(k1=0.5, k2=-0.3, k3=0.4, k4=-0.2, k5=0.3)
    config5 = BeamlineConfig(
        drift_length=1.0, emit_x=10e-9, emit_y=2e-9, quad_length=0.1, n_quads=5
    )
    fig = create_beamline_diagram(quads, config5)
    fig.show()
    print("OK: 5 квадруполей")

    fig4 = create_beamline_diagram(quads, DEFAULT_CONFIG)
    fig4.show()
    print("OK: 4 квадруполя")
