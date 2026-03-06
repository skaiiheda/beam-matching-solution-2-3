from typing import Tuple, List
import plotly.graph_objects as go
import numpy as np

from beam_matching import (
    TwissParams, TwissParamsXY, BeamlineConfig, QuadrupoleSettings,
    propagate_through_beamline_x, propagate_through_beamline_y, DEFAULT_TWISS_TARGET
)


def create_beta_plot(
    twiss_in: TwissParamsXY,
    config: BeamlineConfig,
    quads: QuadrupoleSettings,
    twiss_target: TwissParamsXY = None
) -> go.Figure:
    """
    Create a combined beta and alpha functions plot along the beamline.
    
    Main chart: Beta functions (βx and βy)
    Sub chart: Alpha functions (αx and αy)
    
    Args:
        twiss_in: Input Twiss parameters for X and Y planes
        config: Beamline configuration
        quads: Quadrupole settings
        twiss_target: Target Twiss parameters (optional, uses DEFAULT if None)
    
    Returns:
        Plotly Figure with two subplots (beta and alpha)
    """
    if twiss_target is None:
        twiss_target = DEFAULT_TWISS_TARGET
    
    # Propagate through beamline
    twiss_out_x, history_x = propagate_through_beamline_x(twiss_in.x, config, quads)
    twiss_out_y, history_y = propagate_through_beamline_y(twiss_in.y, config, quads)
    
    # Extract s values and Twiss parameters
    s_values = [h[0] for h in history_x]
    beta_x = [h[1].beta for h in history_x]
    alpha_x = [h[1].alpha for h in history_x]
    beta_y = [h[1].beta for h in history_y]
    alpha_y = [h[1].alpha for h in history_y]
    
    # Quadrupole positions for reference lines
    drift_length = config.drift_length
    quad_positions = [0, drift_length, 2*drift_length, 3*drift_length]
    quad_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    
    # Create subplot with two rows
    fig = go.Figure()
    
    # Beta functions plot (main chart)
    fig.add_trace(go.Scatter(
        x=s_values,
        y=beta_x,
        mode='lines',
        name='βx',
        line=dict(color='#f43f5e', width=2),  # rose-500
        legendgroup='beta',
        hovertemplate='s: %{x:.3f} m<br>βx: %{y:.4f} m<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=s_values,
        y=beta_y,
        mode='lines',
        name='βy',
        line=dict(color='#0ea5e9', width=2),  # sky-500
        legendgroup='beta',
        hovertemplate='s: %{x:.3f} m<br>βy: %{y:.4f} m<extra></extra>'
    ))
    
    # Add reference lines for quadrupole positions
    for pos, label in zip(quad_positions, quad_labels):
        fig.add_shape(
            type='line',
            x0=pos, y0=0,
            x1=pos, y1=max(max(beta_x), max(beta_y)),
            line=dict(color='#94a3b8', width=1, dash='5,5'),
            layer='below'
        )
        fig.add_annotation(
            x=pos,
            y=max(max(beta_x), max(beta_y)),
            text=label,
            showarrow=False,
            font=dict(size=9, color='#64748b'),
            yanchor='bottom',
            yshift=5
        )
    
    # Add horizontal reference lines for target beta values
    max_beta = max(max(beta_x), max(beta_y), twiss_target.x.beta, twiss_target.y.beta) * 1.2
    
    fig.add_shape(
        type='line',
        x0=0, y0=twiss_target.x.beta,
        x1=max(s_values), y1=twiss_target.x.beta,
        line=dict(color='#f43f5e', width=2, dash='10,5'),
        layer='below'
    )
    fig.add_annotation(
        x=max(s_values),
        y=twiss_target.x.beta,
        text=f'βx*={twiss_target.x.beta}',
        showarrow=False,
        font=dict(size=9, color='#f43f5e'),
        xanchor='left',
        xshift=5
    )
    
    fig.add_shape(
        type='line',
        x0=0, y0=twiss_target.y.beta,
        x1=max(s_values), y1=twiss_target.y.beta,
        line=dict(color='#0ea5e9', width=2, dash='10,5'),
        layer='below'
    )
    fig.add_annotation(
        x=max(s_values),
        y=twiss_target.y.beta,
        text=f'βy*={twiss_target.y.beta}',
        showarrow=False,
        font=dict(size=9, color='#0ea5e9'),
        xanchor='left',
        xshift=5
    )
    
    # Update layout for beta plot
    fig.update_layout(
        title=dict(
            text='<b>β-функции вдоль пучкового канала</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='s (м)',
            showgrid=True,
            gridcolor='#f1f5f9',
            zeroline=False
        ),
        yaxis=dict(
            title='β (м)',
            range=[0, max_beta],
            showgrid=True,
            gridcolor='#f1f5f9',
            zeroline=True,
            zerolinecolor='#e2e8f0'
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=60, r=60, t=80, b=60),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        height=450,
        font=dict(family='Arial', size=11)
    )
    
    return fig


if __name__ == '__main__':
    # Test the beta plot with default settings
    from beam_matching import DEFAULT_CONFIG, DEFAULT_TWISS_IN, QuadrupoleSettings
    
    quads = QuadrupoleSettings(k1=0.5, k2=-0.3, k3=0.4, k4=-0.2)
    config = DEFAULT_CONFIG
    twiss_in = DEFAULT_TWISS_IN
    
    fig = create_beta_plot(twiss_in, config, quads)
    fig.show()
    print("Beta plot test passed!")
