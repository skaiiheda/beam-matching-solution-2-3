#!/usr/bin/env python3
"""
Test script to verify beam optics physics calculations
"""
import sys
sys.path.insert(0, '.')

from beam_matching import (
    TwissParams,
    TwissParamsXY,
    QuadrupoleSettings,
    BeamlineConfig,
    calculate_gamma,
    drift_matrix,
    quad_matrix,
    multiply_matrices,
    propagate_twiss,
    propagate_through_beamline_x,
    propagate_through_beamline_y,
    calculate_matching_error,
    optimize_quadrupoles,
    generate_phase_space_ellipse,
    calculate_envelope,
    DEFAULT_CONFIG,
    DEFAULT_TWISS_IN,
    DEFAULT_TWISS_TARGET,
    format_number,
    percent_error,
)


def test_basic_calculations():
    print("=" * 60)
    print("Testing Basic Calculations")
    print("=" * 60)
    
    # Test gamma calculation
    beta, alpha = 5.0, -0.5
    gamma = calculate_gamma(beta, alpha)
    expected_gamma = (1 + alpha * alpha) / beta
    print(f"Gamma calculation: beta={beta}, alpha={alpha}")
    print(f"  Result: {gamma:.6f}, Expected: {expected_gamma:.6f}")
    assert abs(gamma - expected_gamma) < 1e-10, "Gamma calculation failed"
    print("  ✓ PASSED")
    
    # Test drift matrix
    L = 1.0
    M_drift = drift_matrix(L)
    print(f"\nDrift matrix (L={L}):")
    print(f"  [[{M_drift[0,0]}, {M_drift[0,1]}],")
    print(f"   [{M_drift[1,0]}, {M_drift[1,1]}]]")
    assert M_drift[0, 0] == 1 and M_drift[0, 1] == L and M_drift[1, 0] == 0 and M_drift[1, 1] == 1
    print("  ✓ PASSED")
    
    # Test quad matrix
    k = 0.5
    M_quad = quad_matrix(k)
    print(f"\nQuad matrix (k={k}):")
    print(f"  [[{M_quad[0,0]}, {M_quad[0,1]}],")
    print(f"   [{M_quad[1,0]}, {M_quad[1,1]}]]")
    assert M_quad[0, 0] == 1 and M_quad[0, 1] == 0 and M_quad[1, 0] == -k and M_quad[1, 1] == 1
    print("  ✓ PASSED")


def test_twiss_propagation():
    print("\n" + "=" * 60)
    print("Testing Twiss Parameter Propagation")
    print("=" * 60)
    
    twiss = TwissParams(beta=5.0, alpha=-0.5)
    M = drift_matrix(1.0)
    
    twiss_out = propagate_twiss(twiss, M)
    print(f"Input Twiss: β={twiss.beta}, α={twiss.alpha}")
    print(f"After drift L=1.0: β={twiss_out.beta:.6f}, α={twiss_out.alpha:.6f}")
    
    # Manual calculation check
    gamma = calculate_gamma(twiss.beta, twiss.alpha)
    m11, m12 = 1.0, 1.0
    m21, m22 = 0.0, 1.0
    
    beta2_check = m11*m11*twiss.beta - 2*m11*m12*twiss.alpha + m12*m12*gamma
    alpha2_check = -m11*m21*twiss.beta + (m11*m22 + m12*m21)*twiss.alpha - m12*m22*gamma
    
    assert abs(twiss_out.beta - beta2_check) < 1e-10
    assert abs(twiss_out.alpha - alpha2_check) < 1e-10
    print("  ✓ PASSED")


def test_beamline_propagation():
    print("\n" + "=" * 60)
    print("Testing Full Beamline Propagation")
    print("=" * 60)
    
    twiss_in = TwissParams(beta=5.0, alpha=-0.5)
    config = BeamlineConfig(drift_length=1.0, emit_x=10e-9, emit_y=2e-9)
    quads = QuadrupoleSettings(k1=0.1, k2=-0.1, k3=0.1, k4=-0.1)
    
    twiss_out, history = propagate_through_beamline_x(twiss_in, config, quads)
    
    print(f"Input Twiss: β={twiss_in.beta}, α={twiss_in.alpha}")
    print(f"Output Twiss: β={twiss_out.beta:.6f}, α={twiss_out.alpha:.6f}")
    print(f"Number of history points: {len(history)}")
    print(f"Final position: {history[-1][0]:.3f} m")
    
    # Check that we have points
    assert len(history) > 0
    assert history[0][0] == 0
    assert abs(history[-1][0] - 3.0) < 0.1  # Approximately 3m
    print("  ✓ PASSED")


def test_matching_error():
    print("\n" + "=" * 60)
    print("Testing Matching Error Calculation")
    print("=" * 60)
    
    twiss_out = TwissParamsXY(
        x=TwissParams(beta=7.9, alpha=0.01),
        y=TwissParams(beta=4.1, alpha=-0.01)
    )
    
    error = calculate_matching_error(twiss_out, DEFAULT_TWISS_TARGET)
    print(f"Target Twiss: βx*={DEFAULT_TWISS_TARGET.x.beta}, αx*={DEFAULT_TWISS_TARGET.x.alpha}")
    print(f"             βy*={DEFAULT_TWISS_TARGET.y.beta}, αy*={DEFAULT_TWISS_TARGET.y.alpha}")
    print(f"Output Twiss: βx={twiss_out.x.beta}, αx={twiss_out.x.alpha}")
    print(f"             βy={twiss_out.y.beta}, αy={twiss_out.y.alpha}")
    print(f"Matching error: {error:.8f}")
    
    assert error >= 0
    print("  ✓ PASSED")


def test_optimization():
    print("\n" + "=" * 60)
    print("Testing Quadrupole Optimization")
    print("=" * 60)
    
    result = optimize_quadrupoles(
        DEFAULT_TWISS_IN,
        DEFAULT_TWISS_TARGET,
        DEFAULT_CONFIG,
        use_penalty=False
    )

    print(f"Optimization successful: {result['success']}")
    print(f"Final error: {result['error']:.8e}")
    print(f"\nOptimal quadrupole strengths:")
    print(f"  k1 = {result['quads'].k1:.6f} m⁻²")
    print(f"  k2 = {result['quads'].k2:.6f} m⁻²")
    print(f"  k3 = {result['quads'].k3:.6f} m⁻²")
    print(f"  k4 = {result['quads'].k4:.6f} m⁻²")
    
    # Verify error is small
    assert result['error'] < 1e-3
    print("  ✓ PASSED")


def test_phase_space_ellipse():
    print("\n" + "=" * 60)
    print("Testing Phase Space Ellipse Generation")
    print("=" * 60)
    
    twiss = TwissParams(beta=5.0, alpha=-0.5)
    emittance = 10e-9
    
    ellipse = generate_phase_space_ellipse(twiss, emittance, num_points=100)
    
    print(f"Twiss: β={twiss.beta}, α={twiss.alpha}")
    print(f"Emittance: {emittance:.2e} m·rad")
    print(f"Number of ellipse points: {len(ellipse)}")
    print(f"Sample points (first 3):")
    for i in range(3):
        x, xp = ellipse[i]
        print(f"  {i+1}. x={x*1e6:.2f} μm, x'={xp*1e3:.4f} mrad")
    
    assert len(ellipse) == 100
    # Check that ellipse is closed (first and last points should be close)
    assert abs(ellipse[0][0] - ellipse[-1][0]) < 1e-10
    assert abs(ellipse[0][1] - ellipse[-1][1]) < 1e-10
    print("  ✓ PASSED")


def test_envelope():
    print("\n" + "=" * 60)
    print("Testing Beam Envelope Calculation")
    print("=" * 60)
    
    config = BeamlineConfig(drift_length=1.0, emit_x=10e-9, emit_y=2e-9)
    quads = QuadrupoleSettings(k1=0.1, k2=-0.1, k3=0.1, k4=-0.1)
    
    twiss_out, history = propagate_through_beamline_x(
        TwissParams(beta=5.0, alpha=-0.5),
        config,
        quads
    )
    
    envelope = calculate_envelope(history, config.emit_x)
    
    print(f"Number of envelope points: {len(envelope)}")
    print(f"Sample points (first 3):")
    for i in range(3):
        s, sigma, sigma_neg = envelope[i]
        print(f"  {i+1}. s={s:.3f} m, σ={sigma*1e6:.2f} μm, σ⁻={sigma_neg*1e6:.2f} μm")
    
    assert len(envelope) == len(history)
    # Check sigma is positive and sigma_neg is negative
    for s, sigma, sigma_neg in envelope:
        assert sigma > 0
        assert sigma_neg < 0
        assert abs(sigma + sigma_neg) < 1e-10
    print("  ✓ PASSED")


def test_utils():
    print("\n" + "=" * 60)
    print("Testing Utility Functions")
    print("=" * 60)
    
    # Test format_number
    test_cases = [
        (0.0, 4, '0'),
        (5.123456, 4, '5.1235'),
        (12345.678, 4, '1.2346e+04'),
        (0.000123, 4, '1.230e-04'),
        (5.0, 4, '5.0000'),
    ]
    
    for value, decimals, expected in test_cases:
        result = format_number(value, decimals)
        print(f"format_number({value}, {decimals}) = '{result}' (expected: '{expected}')")
        assert result == expected, f"Expected '{expected}', got '{result}'"
    
    print("  format_number: ✓ PASSED")
    
    # Test percent_error
    error_cases = [
        (5.0, 5.0, 0.0),
        (6.0, 5.0, 20.0),
        (4.0, 5.0, 20.0),
        (0.5, 0.0, 50.0),  # When target is 0, use absolute
    ]
    
    for actual, target, expected in error_cases:
        result = percent_error(actual, target)
        print(f"percent_error({actual}, {target}) = {result:.2f}% (expected: {expected:.2f}%)")
        assert abs(result - expected) < 0.01
    
    print("  percent_error: ✓ PASSED")


def main():
    print("\n" + "=" * 60)
    print("BEAM OPTICS PHYSICS TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_basic_calculations()
        test_twiss_propagation()
        test_beamline_propagation()
        test_matching_error()
        test_optimization()
        test_phase_space_ellipse()
        test_envelope()
        test_utils()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
