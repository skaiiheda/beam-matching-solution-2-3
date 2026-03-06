from typing import Dict, Any
import pandas as pd

from beam_matching import (
    TwissParams, TwissParamsXY, BeamlineConfig, QuadrupoleSettings,
    propagate_through_beamline_x, propagate_through_beamline_y,
    calculate_matching_error, format_number, percent_error, DEFAULT_TWISS_TARGET
)


def create_statistics_table(
    twiss_in: TwissParamsXY,
    config: BeamlineConfig,
    quads: QuadrupoleSettings,
    twiss_target: TwissParamsXY = None
) -> pd.DataFrame:
    """
    Create statistics table comparing input, output, and target Twiss parameters.
    
    Args:
        twiss_in: Input Twiss parameters
        config: Beamline configuration
        quads: Quadrupole settings
        twiss_target: Target Twiss parameters (optional)
    
    Returns:
        pandas DataFrame with statistics
    """
    if twiss_target is None:
        twiss_target = DEFAULT_TWISS_TARGET
    
    # Calculate output Twiss parameters
    twiss_out_x, _ = propagate_through_beamline_x(twiss_in.x, config, quads)
    twiss_out_y, _ = propagate_through_beamline_y(twiss_in.y, config, quads)
    
    twiss_out = TwissParamsXY(x=twiss_out_x, y=twiss_out_y)
    
    # Create table data
    table_data = [
        {
            'Parameter': 'βx',
            'Unit': 'м',
            'Input': twiss_in.x.beta,
            'Output': twiss_out.x.beta,
            'Target': twiss_target.x.beta,
            'Error %': percent_error(twiss_out.x.beta, twiss_target.x.beta)
        },
        {
            'Parameter': 'αx',
            'Unit': '',
            'Input': twiss_in.x.alpha,
            'Output': twiss_out.x.alpha,
            'Target': twiss_target.x.alpha,
            'Error %': percent_error(twiss_out.x.alpha, twiss_target.x.alpha)
        },
        {
            'Parameter': 'βy',
            'Unit': 'м',
            'Input': twiss_in.y.beta,
            'Output': twiss_out.y.beta,
            'Target': twiss_target.y.beta,
            'Error %': percent_error(twiss_out.y.beta, twiss_target.y.beta)
        },
        {
            'Parameter': 'αy',
            'Unit': '',
            'Input': twiss_in.y.alpha,
            'Output': twiss_out.y.alpha,
            'Target': twiss_target.y.alpha,
            'Error %': percent_error(twiss_out.y.alpha, twiss_target.y.alpha)
        }
    ]
    
    df = pd.DataFrame(table_data)
    
    # Format columns
    df['Input'] = df['Input'].apply(lambda x: format_number(x, 4))
    df['Output'] = df['Output'].apply(lambda x: format_number(x, 4))
    df['Target'] = df['Target'].apply(lambda x: format_number(x, 4))
    df['Error %'] = df['Error %'].apply(lambda x: f"{x:.2f}%")
    
    return df


def create_quadrupole_summary(quads: QuadrupoleSettings) -> Dict[str, Any]:
    """
    Create summary statistics for quadrupole strengths.
    
    Args:
        quads: Quadrupole settings
    
    Returns:
        Dictionary with summary statistics
    """
    k_values = [quads.k1, quads.k2, quads.k3, quads.k4]
    
    return {
        'k1': {'value': quads.k1, 'formatted': format_number(quads.k1, 4)},
        'k2': {'value': quads.k2, 'formatted': format_number(quads.k2, 4)},
        'k3': {'value': quads.k3, 'formatted': format_number(quads.k3, 4)},
        'k4': {'value': quads.k4, 'formatted': format_number(quads.k4, 4)},
        'avg': sum(abs(k) for k in k_values) / 4,
        'max': max(abs(k) for k in k_values),
    }


def calculate_matching_statistics(
    twiss_out: TwissParamsXY,
    twiss_target: TwissParamsXY
) -> Dict[str, Any]:
    """
    Calculate overall matching statistics.
    
    Args:
        twiss_out: Output Twiss parameters
        twiss_target: Target Twiss parameters
    
    Returns:
        Dictionary with statistics
    """
    errors = [
        percent_error(twiss_out.x.beta, twiss_target.x.beta),
        percent_error(twiss_out.x.alpha, twiss_target.x.alpha),
        percent_error(twiss_out.y.beta, twiss_target.y.beta),
        percent_error(twiss_out.y.alpha, twiss_target.y.alpha)
    ]
    
    avg_error = sum(errors) / len(errors)
    max_error = max(errors)
    is_matched = avg_error < 5  # Less than 5% average error
    
    return {
        'avg_error': avg_error,
        'max_error': max_error,
        'is_matched': is_matched,
        'error': calculate_matching_error(twiss_out, twiss_target)
    }


if __name__ == '__main__':
    # Test the statistics table with default settings
    from beam_matching import DEFAULT_CONFIG, DEFAULT_TWISS_IN, QuadrupoleSettings
    
    quads = QuadrupoleSettings(k1=0.5, k2=-0.3, k3=0.4, k4=-0.2)
    config = DEFAULT_CONFIG
    twiss_in = DEFAULT_TWISS_IN
    
    df = create_statistics_table(twiss_in, config, quads)
    print("Statistics Table:")
    print(df.to_string(index=False))
    
    # Test quadrupole summary
    quad_summary = create_quadrupole_summary(quads)
    print("\nQuadrupole Summary:")
    print(f"k1: {quad_summary['k1']['formatted']} m⁻²")
    print(f"k2: {quad_summary['k2']['formatted']} m⁻²")
    print(f"k3: {quad_summary['k3']['formatted']} m⁻²")
    print(f"k4: {quad_summary['k4']['formatted']} m⁻²")
    
    # Test matching statistics
    twiss_out_x, _ = propagate_through_beamline_x(twiss_in.x, config, quads)
    twiss_out_y, _ = propagate_through_beamline_y(twiss_in.y, config, quads)
    twiss_out = TwissParamsXY(x=twiss_out_x, y=twiss_out_y)
    
    stats = calculate_matching_statistics(twiss_out, DEFAULT_TWISS_TARGET)
    print("\nMatching Statistics:")
    print(f"Average Error: {stats['avg_error']:.2f}%")
    print(f"Max Error: {stats['max_error']:.2f}%")
    print(f"Matched: {stats['is_matched']}")
    print(f"Error Value: {stats['error']:.8e}")
    
    print("\nStatistics table test passed!")
