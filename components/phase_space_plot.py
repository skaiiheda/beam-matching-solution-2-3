from typing import Tuple, List
import plotly.graph_objects as go
import numpy as np

from beam_matching import (
    TwissParams, TwissParamsXY, BeamlineConfig, QuadrupoleSettings,
    propagate_through_beamline_x, propagate_through_beamline_y,
    generate_phase_space_ellipse, DEFAULT_TWISS_TARGET
)


def _create_phase_space_panel(
    title: str,
    input_ellipse: List[Tuple[float, float]],
    target_ellipse: List[Tuple[float, float]],
    current_ellipse: List[Tuple[float, float]],
    x_label: str,
    xp_label: str,
    scale: float
) -> go.Figure:
    """
    Create a single phase space plot panel.

    Args:
        title: Panel title
        input_ellipse: List of (x, x') points for input ellipse
        target_ellipse: List of (x, x') points for target ellipse
        current_ellipse: List of (x, x') points for current position ellipse
        x_label: X-axis label
        xp_label: Y-axis label
        scale: Axis scale

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Extract x and xp coordinates
    x_in = [p[0] for p in input_ellipse]
    xp_in = [p[1] for p in input_ellipse]
    x_target = [p[0] for p in target_ellipse]
    xp_target = [p[1] for p in target_ellipse]
    x_current = [p[0] for p in current_ellipse]
    xp_current = [p[1] for p in current_ellipse]

    # Close the ellipses by adding first point at the end
    x_in_closed = x_in + [x_in[0]]
    xp_in_closed = xp_in + [xp_in[0]]
    x_target_closed = x_target + [x_target[0]]
    xp_target_closed = xp_target + [xp_target[0]]
    x_current_closed = x_current + [x_current[0]]
    xp_current_closed = xp_current + [xp_current[0]]

    # Input ellipse (dashed orange)
    fig.add_trace(go.Scatter(
        x=x_in_closed,
        y=xp_in_closed,
        mode='lines',
        name='Вход (пунктир)',
        line=dict(color='#f59e0b', width=2, dash='5,5'),
        hoverinfo='skip'
    ))

    # Target ellipse (dotted green)
    fig.add_trace(go.Scatter(
        x=x_target_closed,
        y=xp_target_closed,
        mode='lines',
        name='Цель (точки)',
        line=dict(color='#10b981', width=1.5, dash='1,3'),
        hoverinfo='skip'
    ))

    # Current ellipse (solid rose)
    fig.add_trace(go.Scatter(
        x=x_current_closed,
        y=xp_current_closed,
        mode='lines',
        name='Текущая позиция',
        line=dict(color='#f43f5e', width=2),
        hovertemplate=f'{x_label}: %{{x:.2f}} мкм<br>{xp_label}: %{{y:.4f}} мрад<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=14)
        ),
        xaxis=dict(
            title=x_label,
            range=[-scale, scale],
            showgrid=True,
            gridcolor='#f1f5f9',
            zeroline=True,
            zerolinecolor='#94a3b8'
        ),
        yaxis=dict(
            title=xp_label,
            range=[-scale * 0.001, scale * 0.001],
            showgrid=True,
            gridcolor='#f1f5f9',
            zeroline=True,
            zerolinecolor='#94a3b8'
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e2e8f0',
            borderwidth=1
        ),
        margin=dict(l=60, r=60, t=60, b=60),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        height=400,
        font=dict(family='Arial', size=11)
    )
    
    return fig


def create_phase_space_plot(
    twiss_in: TwissParamsXY,
    config: BeamlineConfig,
    quads: QuadrupoleSettings,
    twiss_target: TwissParamsXY = None,
    selected_position_index: int = 5,
    points_per_drift: int = 50
) -> Tuple[go.Figure, go.Figure]:
    """
    Create phase space plots for X and Y planes.

    Returns two figures (one for X-plane, one for Y-plane).
    These should be displayed using Streamlit tabs.

    Args:
        twiss_in: Input Twiss parameters
        config: Beamline configuration
        quads: Quadrupole settings
        twiss_target: Target Twiss parameters (optional)
        selected_position_index: Position to display (0=Input, 1=After Q1, 2=After Q2, 3=After Q3, 4=After Q4, 5=Output)
        points_per_drift: Number of points per drift in history

    Returns:
        Tuple of (x_plane_figure, y_plane_figure)
    """
    if twiss_target is None:
        twiss_target = DEFAULT_TWISS_TARGET

    # Propagate through beamline and capture history
    twiss_out_x, history_x = propagate_through_beamline_x(
        twiss_in.x, config, quads, points_per_drift
    )
    twiss_out_y, history_y = propagate_through_beamline_y(
        twiss_in.y, config, quads, points_per_drift
    )

    def get_twiss_at_position(pos_idx: int, history: List, twiss_out: TwissParams) -> TwissParams:
        """Get Twiss parameters at specified position along beamline."""
        if pos_idx == 0:
            return history[0][1]
        elif pos_idx == 1:
            return history[1][1]
        elif pos_idx == 2:
            idx = 1 + (points_per_drift + 1)
            return history[idx][1]
        elif pos_idx == 3:
            idx = 1 + 2 * (points_per_drift + 1)
            return history[idx][1]
        elif pos_idx == 4 or pos_idx == 5:
            return twiss_out
        else:
            return twiss_out

    # Get Twiss parameters at selected position
    current_twiss_x = get_twiss_at_position(selected_position_index, history_x, twiss_out_x)
    current_twiss_y = get_twiss_at_position(selected_position_index, history_y, twiss_out_y)

    # Generate ellipses for X-plane
    ellipse_x_in = generate_phase_space_ellipse(twiss_in.x, config.emit_x, 100)
    ellipse_x_target = generate_phase_space_ellipse(twiss_target.x, config.emit_x, 100)
    ellipse_x_current = generate_phase_space_ellipse(current_twiss_x, config.emit_x, 100)

    # Generate ellipses for Y-plane
    ellipse_y_in = generate_phase_space_ellipse(twiss_in.y, config.emit_y, 100)
    ellipse_y_target = generate_phase_space_ellipse(twiss_target.y, config.emit_y, 100)
    ellipse_y_current = generate_phase_space_ellipse(current_twiss_y, config.emit_y, 100)

    # Calculate axis scale from all ellipses
    all_ellipses = [
        ellipse_x_in, ellipse_x_target, ellipse_x_current,
        ellipse_y_in, ellipse_y_target, ellipse_y_current
    ]
    max_x = max(abs(p[0]) for ellipse in all_ellipses for p in ellipse)
    max_xp = max(abs(p[1]) for ellipse in all_ellipses for p in ellipse)
    scale = max(max_x, max_xp / 0.001) * 1.2

    # Create X-plane figure
    fig_x = _create_phase_space_panel(
        title='Фазовое пространство плоскости X',
        input_ellipse=ellipse_x_in,
        target_ellipse=ellipse_x_target,
        current_ellipse=ellipse_x_current,
        x_label='x (мкм)',
        xp_label="x' (мрад)",
        scale=scale
    )

    # Create Y-plane figure
    fig_y = _create_phase_space_panel(
        title='Фазовое пространство плоскости Y',
        input_ellipse=ellipse_y_in,
        target_ellipse=ellipse_y_target,
        current_ellipse=ellipse_y_current,
        x_label='y (мкм)',
        xp_label="y' (мрад)",
        scale=scale
    )

    return fig_x, fig_y


if __name__ == '__main__':
    # Test the phase space plot with default settings
    from beam_matching import DEFAULT_CONFIG, DEFAULT_TWISS_IN, QuadrupoleSettings
    
    quads = QuadrupoleSettings(k1=0.5, k2=-0.3, k3=0.4, k4=-0.2)
    config = DEFAULT_CONFIG
    twiss_in = DEFAULT_TWISS_IN
    
    fig_x, fig_y = create_phase_space_plot(twiss_in, config, quads)
    fig_x.show()
    fig_y.show()
    print("Phase space plot test passed!")
