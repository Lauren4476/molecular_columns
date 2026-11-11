import numpy as np
import pytest
import astropy.units as u

try:
    from astropy.units.errors import UnitsError
except ImportError:  # pragma: no cover
    from astropy.units.core import UnitsError
import molecular_columns.col_hcop as col_hcop


def test_col_hcop_Q_HCOp():
    value = col_hcop.Q_HCOp(Tex=5 * u.K)  # type: ignore
    np.testing.assert_almost_equal(value, 2.7008, decimal=4)

    value = col_hcop.Q_HCOp(Tex=[5.0, 5.0, 5.0] * u.K)  # type: ignore
    np.testing.assert_almost_equal(value, [2.7008, 2.7008, 2.7008], decimal=4)


def test_col_hcop_invalid_units():
    with pytest.raises(UnitsError):
        col_hcop.Q_HCOp(Tex=5 * u.m)  # type: ignore


def test_col_hcop_thin():
    result = col_hcop.HCOp_thin(
        J_up=30, Tex=5 * u.K, TdV=1.0 * u.K * u.km / u.s  # type: ignore
    )  # type: ignore
    assert np.isnan(result.value)
    result = col_hcop.HCOp_thin(
        J_up=1, Tex=5 * u.K, TdV=1.0 * u.K * u.km / u.s  # type: ignore
    )  # type: ignore
    assert pytest.approx(result.to(u.cm**-2).value, rel=0.001) == 1.19781e12  # type: ignore
