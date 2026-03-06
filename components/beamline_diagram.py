from dataclasses import dataclass
from typing import List, Tuple
import plotly.graph_objects as go
import numpy as np

from beam_matching import QuadrupoleSettings, BeamlineConfig


def create_beamline_diagram(
    quads: QuadrupoleSettings,
    config: BeamlineConfig
) -> go.Figure:
    """
    Create a Plotly-based beamline schematic visualization.
    
    Shows Q1 → Drift → Q2 → Drift → Q3 → Drift → Q4 layout.
    
    Args:
        quads: Quadrupole settings with k1, k2, k3, k4 strengths
        config: Beamline configuration with drift_length
    
    Returns:
        Plotly Figure object
    """
    drift_length = config.drift_length
    k_values = [quads.k1, quads.k2, quads.k3, quads.k4]
    labels = ['Q1', 'Q2', 'Q3', 'Q4']
    
    # Calculate positions
    # Q1 at 0, Q2 at drift_length, Q3 at 2*drift_length, Q4 at 3*drift_length
    quad_positions = [0, drift_length, 2*drift_length, 3*drift_length]
    
    fig = go.Figure()
    
    # Determine lens type and colors
    def get_lens_color(k: float) -> str:
        if abs(k) < 0.001:
            return '#9ca3af'  # gray-400 (neutral)
        return '#10b981' if k > 0 else '#f43f5e'  # green (focusing) or red (defocusing)
    
    def get_lens_type(k: float) -> str:
        if abs(k) < 0.001:
            return 'neutral'
        return 'focusing' if k > 0 else 'defocusing'
    
    # Draw quadrupoles as vertical bars
    bar_width = 0.08  # meters
    max_height = max(abs(k) for k in k_values) if k_values else 1.0
    scale_height = 2.0 / max(max_height, 0.1)  # Normalize heights
    
    for i, (pos, k, label) in enumerate(zip(quad_positions, k_values, labels)):
        color = get_lens_color(k)
        lens_type = get_lens_type(k)
        
        # Height proportional to strength, with minimum height
        height = abs(k) * scale_height
        height = max(height, 0.5)
        
        # Draw quad as a filled rectangle using polygon
        quad_x = [pos - bar_width/2, pos + bar_width/2, pos + bar_width/2, pos - bar_width/2]
        quad_y = [-height/2, -height/2, height/2, height/2]
        
        fig.add_trace(go.Scatter(
            x=quad_x,
            y=quad_y,
            fill='toself',
            fillcolor=color,
            line=dict(color=color, width=2),
            mode='lines',
            name=f'{label} ({lens_type})',
            showlegend=i == 0  # Only show legend for first quad
        ))
        
        # Add label above quad
        fig.add_annotation(
            x=pos,
            y=height/2 + 0.3,
            text=f'<b>{label}</b>',
            showarrow=False,
            font=dict(size=14, family='Arial Black'),
            yanchor='bottom'
        )
        
        # Add k value below quad
        fig.add_annotation(
            x=pos,
            y=-height/2 - 0.3,
            text=f'k={k:.3f}',
            showarrow=False,
            font=dict(size=10, color='#64748b'),
            yanchor='top'
        )
    
    # Draw drift spaces as horizontal lines
    total_length = 3 * drift_length
    
    # Drift 1: between Q1 and Q2
    drift_y = 0
    fig.add_shape(
        type='line',
        x0=bar_width/2, y0=drift_y,
        x1=drift_length - bar_width/2, y1=drift_y,
        line=dict(color='#e2e8f0', width=4)
    )
    
    # Drift 2: between Q2 and Q3
    fig.add_shape(
        type='line',
        x0=drift_length + bar_width/2, y0=drift_y,
        x1=2*drift_length - bar_width/2, y1=drift_y,
        line=dict(color='#e2e8f0', width=4)
    )
    
    # Drift 3: between Q3 and Q4
    fig.add_shape(
        type='line',
        x0=2*drift_length + bar_width/2, y0=drift_y,
        x1=3*drift_length - bar_width/2, y1=drift_y,
        line=dict(color='#e2e8f0', width=4)
    )
    
    # Add drift length labels
    drift_label_y = -0.8
    fig.add_annotation(
        x=drift_length / 2,
        y=drift_label_y,
        text=f'L={drift_length:.1f}m',
        showarrow=False,
        font=dict(size=9, color='#94a3b8'),
        yanchor='top'
    )
    fig.add_annotation(
        x=1.5 * drift_length,
        y=drift_label_y,
        text=f'L={drift_length:.1f}m',
        showarrow=False,
        font=dict(size=9, color='#94a3b8'),
        yanchor='top'
    )
    fig.add_annotation(
        x=2.5 * drift_length,
        y=drift_label_y,
        text=f'L={drift_length:.1f}m',
        showarrow=False,
        font=dict(size=9, color='#94a3b8'),
        yanchor='top'
    )
    
    # Add beam direction arrow
    arrow_y = 1.5
    arrow_x_start = 0.2
    arrow_x_end = total_length - 0.2
    
    fig.add_annotation(
        x=total_length / 2,
        y=arrow_y,
        text='<b>Направление пучка →</b>',
        showarrow=False,
        font=dict(size=12, color='#64748b'),
        yanchor='bottom'
    )
    
    # Draw arrow line
    fig.add_shape(
        type='line',
        x0=arrow_x_start, y0=arrow_y - 0.15,
        x1=arrow_x_end - 0.1, y1=arrow_y - 0.15,
        line=dict(color='#64748b', width=2)
    )
    
    # Draw arrow head
    fig.add_shape(
        type='path',
        path=f'M {arrow_x_end - 0.1},{arrow_y - 0.15} L {arrow_x_end - 0.2},{arrow_y - 0.25} L {arrow_x_end - 0.2},{arrow_y - 0.05} Z',
        line=dict(color='#64748b', width=2),
        fillcolor='#64748b'
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Компоновка пучкового канала</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis=dict(
            title=dict(text='Позиция вдоль пучкового канала (м)', standoff=10),
            range=[-0.3, total_length + 0.3],
            showgrid=True,
            gridcolor='#f1f5f9',
            zeroline=False,
            tickmode='linear',
            tick0=0,
            dtick=drift_length
        ),
        yaxis=dict(
            title=dict(text='Сила квадруполя (нормализованная)', standoff=10),
            range=[-2, 2],
            showgrid=True,
            gridcolor='#f1f5f9',
            zeroline=True,
            zerolinecolor='#94a3b8',
            zerolinewidth=1,
            showticklabels=False,
            fixedrange=True
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
        margin=dict(l=60, r=60, t=80, b=80),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        height=400,
        font=dict(family='Arial', size=11)
    )
    
    # Add legend entries manually
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color='#10b981', size=15, symbol='square'),
        name='Фокусировка (k > 0) в плоскости X',
        legendgroup='focusing'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color='#f43f5e', size=15, symbol='square'),
        name='Дефокусировка (k < 0) в плоскости X',
        legendgroup='defocusing'
    ))
    
    return fig


if __name__ == '__main__':
    # Test the diagram with default settings
    from beam_matching import DEFAULT_CONFIG, QuadrupoleSettings
    
    quads = QuadrupoleSettings(k1=0.5, k2=-0.3, k3=0.4, k4=-0.2)
    config = DEFAULT_CONFIG
    
    fig = create_beamline_diagram(quads, config)
    fig.show()
    print("Beamline diagram test passed!")
