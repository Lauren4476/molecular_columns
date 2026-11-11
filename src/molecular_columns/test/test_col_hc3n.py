import numpy as np
import pytest
import astropy.units as u

try:
    from astropy.units.errors import UnitsError
except ImportError:  # pragma: no cover
    from astropy.units.core import UnitsError
import molecular_columns.col_hc3n as col_hc3n


def test_col_hc3n_Q_HC3N():
    value = col_hc3n.Q_HC3N(Tex=5 * u.K)  # type: ignore
    np.testing.assert_almost_equal(value, 23.2385, decimal=4)

    value = col_hc3n.Q_HC3N(Tex=[5.0, 5.0, 5.0] * u.K)  # type: ignore
    np.testing.assert_almost_equal(value, [23.2385, 23.2385, 23.2385], decimal=4)


def test_col_hc3n_invalid_units():
    with pytest.raises(UnitsError):
        col_hc3n.Q_HC3N(Tex=5 * u.m)  # type: ignore


def test_col_hc3n_thin():
    result = col_hc3n.HC3N_thin(
        J_up=37, Tex=5 * u.K, TdV=1.0 * u.K * u.km / u.s  # type: ignore
    )  # type: ignore
    assert np.isnan(result.value)
    result = col_hc3n.HC3N_thin(
        J_up=1, Tex=5 * u.K, TdV=1.0 * u.K * u.km / u.s  # type: ignore
    )  # type: ignore
    assert pytest.approx(result.to(u.cm**-2).value, rel=0.001) == 7.05144e13  # type: ignore
    result = col_hc3n.HC3N_thin(
        J_up=10, Tex=10 * u.K, TdV=1.0 * u.K * u.km / u.s  # type: ignore
    )  # type: ignore
    assert pytest.approx(result.to(u.cm**-2).value, rel=0.001) == 7.79681e12  # type: ignore
