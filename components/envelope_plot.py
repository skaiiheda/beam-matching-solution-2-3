from typing import List, Tuple
import plotly.graph_objects as go

from beam_matching import (
    TwissParams, TwissParamsXY, BeamlineConfig, QuadrupoleSettings,
    propagate_through_beamline_x, propagate_through_beamline_y, calculate_envelope
)


def create_envelope_plot(
    twiss_in: TwissParamsXY,
    config: BeamlineConfig,
    quads: QuadrupoleSettings
) -> go.Figure:
    """
    Create beam envelope plot showing σ = √(β·ε) along the beamline.
    
    Shows filled areas for both X and Y envelopes.
    
    Args:
        twiss_in: Input Twiss parameters
        config: Beamline configuration
        quads: Quadrupole settings
    
    Returns:
        Plotly Figure
    """
    # Propagate through beamline
    twiss_out_x, history_x = propagate_through_beamline_x(twiss_in.x, config, quads)
    twiss_out_y, history_y = propagate_through_beamline_y(twiss_in.y, config, quads)
    
    # Calculate envelopes
    envelope_x = calculate_envelope(history_x, config.emit_x)
    envelope_y = calculate_envelope(history_y, config.emit_y)
    
    # Extract data for X envelope
    s_values = [p[0] for p in envelope_x]
    sigma_x = [p[1] * 1e6 for p in envelope_x]  # Convert to μm
    sigma_x_neg = [p[2] * 1e6 for p in envelope_x]
    
    # Extract data for Y envelope
    sigma_y = [p[1] * 1e6 for p in envelope_y]  # Convert to μm
    sigma_y_neg = [p[2] * 1e6 for p in envelope_y]
    
    # Quadrupole positions for reference lines
    drift_length = config.drift_length
    quad_positions = [0, drift_length, 2*drift_length, 3*drift_length]
    quad_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    
    fig = go.Figure()
    
    # X envelope - filled area (upper and lower)
    # Create a closed polygon for the filled area
    s_x_rev = list(reversed(s_values))
    sigma_x_neg_rev = list(reversed(sigma_x_neg))
    
    fig.add_trace(go.Scatter(
        x=s_values + s_x_rev,
        y=sigma_x + sigma_x_neg_rev,
        fill='toself',
        fillcolor='rgba(244, 63, 94, 0.2)',  # rose-500 with opacity
        line=dict(color='#f43f5e', width=1.5),  # rose-500
        name='σx',
        hovertemplate='s: %{x:.3f} m<br>σx: %{y:.2f} μm<extra></extra>',
        legendgroup='x'
    ))
    
    # Y envelope - filled area
    sigma_y_neg_rev = list(reversed(sigma_y_neg))
    
    fig.add_trace(go.Scatter(
        x=s_values + s_x_rev,
        y=sigma_y + sigma_y_neg_rev,
        fill='toself',
        fillcolor='rgba(14, 165, 233, 0.2)',  # sky-500 with opacity
        line=dict(color='#0ea5e9', width=1.5),  # sky-500
        name='σy',
        hovertemplate='s: %{x:.3f} m<br>σy: %{y:.2f} μm<extra></extra>',
        legendgroup='y'
    ))
    
    # Add reference lines for quadrupole positions
    max_sigma = max(abs(x) for x in sigma_x + sigma_y + sigma_x_neg + sigma_y_neg)
    min_sigma = min(sigma_x_neg + sigma_y_neg)
    
    for pos, label in zip(quad_positions, quad_labels):
        fig.add_shape(
            type='line',
            x0=pos, y0=min_sigma,
            x1=pos, y1=max_sigma,
            line=dict(color='#94a3b8', width=1, dash='5,5'),
            layer='below'
        )
        fig.add_annotation(
            x=pos,
            y=max_sigma,
            text=label,
            showarrow=False,
            font=dict(size=9, color='#64748b'),
            yanchor='bottom',
            yshift=5
        )
    
    # Add zero line
    fig.add_shape(
        type='line',
        x0=min(s_values), y0=0,
        x1=max(s_values), y1=0,
        line=dict(color='#94a3b8', width=1),
        layer='below'
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Огибающая пучка (σ = √(β·ε))</b>',
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
            title='σ (мкм)',
            range=[min_sigma * 1.1, max_sigma * 1.1],
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
        margin=dict(l=60, r=60, t=80, b=80),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        height=350,
        font=dict(family='Arial', size=11)
    )
    
    return fig


if __name__ == '__main__':
    # Test the envelope plot with default settings
    from beam_matching import DEFAULT_CONFIG, DEFAULT_TWISS_IN, QuadrupoleSettings
    
    quads = QuadrupoleSettings(k1=0.5, k2=-0.3, k3=0.4, k4=-0.2)
    config = DEFAULT_CONFIG
    twiss_in = DEFAULT_TWISS_IN
    
    fig = create_envelope_plot(twiss_in, config, quads)
    fig.show()
    print("Envelope plot test passed!")
