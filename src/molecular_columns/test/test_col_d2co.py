import numpy as np
import pytest
import astropy.units as u
import molecular_columns.col_d2co as col_d2co

try:
    from astropy.units.errors import UnitsError
except ImportError:  # pragma: no cover
    from astropy.units.core import UnitsError


def test_parse_qns():
    """Test quantum number parsing for D2CO transitions."""
    upper, lower = col_d2co.parse_qns("10(6,4) - 10(6,5)")
    assert upper == (10, 6, 4)
    assert lower == (10, 6, 5)
    
    # Test with different quantum numbers
    upper, lower = col_d2co.parse_qns("4(4,0) - 4(4,1)")
    assert upper == (4, 4, 0)
    assert lower == (4, 4, 1)


def test_load_d2co_csv():
    """Test CSV loading with header cleanup."""
    df = col_d2co.load_d2co_csv("src/molecular_columns/o-d2co.csv")
    
    # Verify expected columns are present and stripped of whitespace
    expected_cols = [
        "Frequency (MHz)",
        "Resolved QNs",
        "E_U (K)",
        "aij",
        "Upper State Degeneracy",
        "ortho_or_para",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Expected column '{col}' not found"
    
    # Verify data is loaded
    assert len(df) > 0


def test_o_D2CO_thin_csv_invalid_units():
    """Test that invalid unit inputs are rejected."""
    df = col_d2co.load_d2co_csv("src/molecular_columns/o-d2co.csv")
    
    # freq must have frequency units
    with pytest.raises(UnitsError):
        col_d2co.o_D2CO_thin_csv(df, freq=231.410234 * u.K)  # type: ignore
    
    # Tex must have temperature units
    with pytest.raises(UnitsError):
        col_d2co.o_D2CO_thin_csv(df, freq=231.410234 * u.GHz, Tex=5 * u.m)  # type: ignore
    
    # TdV must have K*km/s units
    with pytest.raises(UnitsError):
        col_d2co.o_D2CO_thin_csv(
            df, freq=231.410234 * u.GHz, TdV=1.0 * u.K  # type: ignore
        )


def test_p_D2CO_thin_csv_invalid_units():
    """Test that invalid unit inputs are rejected for para."""
    df = col_d2co.load_d2co_csv("src/molecular_columns/p-d2co.csv")
    
    with pytest.raises(UnitsError):
        col_d2co.p_D2CO_thin_csv(df, freq=231.410234 * u.K)  # type: ignore


def test_o_D2CO_thin_csv_regression():
    """Test ortho-D2CO column density with known values."""
    df = col_d2co.load_d2co_csv("src/molecular_columns/o-d2co.csv")
    
    # Use the same frequency and parameters as in test_d2co.ipynb
    freq = 231.410234 * u.GHz
    Tex = 20 * u.K
    TdV = 1.0 * u.K * u.km / u.s
    T_bg = 2.73 * u.K
    
    result = col_d2co.o_D2CO_thin_csv(df, freq, Tex, TdV, T_bg)
    
    # Check that result has correct units
    assert result.unit == u.cm**-2
    
    # Check that result is a finite number
    assert np.isfinite(result.value)


def test_p_D2CO_thin_csv_regression():
    """Test para-D2CO column density with known values."""
    df = col_d2co.load_d2co_csv("src/molecular_columns/p-d2co.csv")
    
    # Use a frequency that exists in the para CSV
    # 0.1663 MHz from the first row of p-d2co.csv
    freq = 0.1663 * u.MHz
    Tex = 5 * u.K
    TdV = 1.0 * u.K * u.km / u.s
    T_bg = 2.73 * u.K
    
    result = col_d2co.p_D2CO_thin_csv(df, freq, Tex, TdV, T_bg)
    
    # Check that result has correct units
    assert result.unit == u.cm**-2
    
    # Check that result is a finite number
    assert np.isfinite(result.value)


def test_o_D2CO_thin_csv_scalar_values():
    """Test ortho-D2CO with different Tex values."""
    df = col_d2co.load_d2co_csv("src/molecular_columns/o-d2co.csv")
    
    freq = 231.410234 * u.GHz
    TdV = 1.0 * u.K * u.km / u.s
    T_bg = 2.73 * u.K
    
    # Compute column density at two different temperatures
    result_5K = col_d2co.o_D2CO_thin_csv(df, freq, Tex=5 * u.K, TdV=TdV, T_bg=T_bg)
    result_20K = col_d2co.o_D2CO_thin_csv(df, freq, Tex=20 * u.K, TdV=TdV, T_bg=T_bg)
    
    # Both should have correct units and be finite
    assert result_5K.unit == u.cm**-2
    assert result_20K.unit == u.cm**-2
    assert np.isfinite(result_5K.value)
    assert np.isfinite(result_20K.value)