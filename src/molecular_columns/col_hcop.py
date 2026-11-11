import numpy as np
import astropy.units as u
from numpy.typing import NDArray
from astropy.constants import c, k_B, h  # type: ignore

from .common_functions import J_nu

# g_u and E_u values obtained from LAMDA database
# https://home.strw.leidenuniv.nl/~moldata/datafiles/hco+@xpol.dat
gu_list = np.array(
    [
        1.0,
        3.0,
        5.0,
        7.0,
        9.0,
        11.0,
        13.0,
        15.0,
        17.0,
        19.0,
        21.0,
        23.0,
        25.0,
        27.0,
        29.0,
        31.0,
        33.0,
        35.0,
        37.0,
        39.0,
        41.0,
        43.0,
        45.0,
        47.0,
        49.0,
        51.0,
        53.0,
        55.0,
        57.0,
        59.0,
        61.0,
    ]
)
E_u_list = (
    np.array(
        [
            0.0000,
            2.9750,
            8.9250,
            17.8497,
            29.7491,
            44.6228,
            62.4706,
            83.2919,
            107.0864,
            133.8536,
            163.5927,
            196.3033,
            231.9844,
            270.6354,
            312.2554,
            356.8434,
            404.3984,
            454.9193,
            508.4050,
            564.8543,
            624.2659,
            686.6383,
            751.9703,
            820.2602,
            891.5066,
            965.7077,
            1042.8617,
            1122.9670,
            1206.0216,
            1292.0236,
            1380.9708,
        ]
    )
    * (h * c / k_B)
    / u.cm  # type: ignore
).to(
    u.K  # type: ignore
)
full_index = np.arange(np.size(E_u_list))

freq_list = (
    np.array(
        [
            89.1885230,
            178.3750650,
            267.5576190,
            356.7342880,
            445.9029960,
            535.0617755,
            624.2086733,
            713.3420900,
            802.4583290,
            891.5579242,
            980.6374000,
            1069.6938000,
            1158.7276478,
            1247.7348251,
            1336.7138728,
            1425.6627131,
            1514.5792539,
            1603.4613882,
            1692.3069926,
            1781.1139268,
            1869.8800326,
            1958.6031327,
            2047.2810302,
            2135.9115072,
            2224.4923240,
            2313.0212183,
            2401.4959041,
            2489.9140708,
            2578.2733822,
            2666.5714759,
        ]
    )
    * u.GHz  # type: ignore
)

Aij_list = (
    np.array(
        [
            4.2512e-05,
            4.0810e-04,
            1.4757e-03,
            3.6269e-03,
            7.2449e-03,
            1.2711e-02,
            2.0402e-02,
            3.0711e-02,
            4.4006e-02,
            6.0667e-02,
            8.1070e-02,
            1.0562e-01,
            1.3467e-01,
            1.6860e-01,
            2.0777e-01,
            2.5255e-01,
            3.0338e-01,
            3.6050e-01,
            4.2444e-01,
            4.9549e-01,
            5.7405e-01,
            6.6034e-01,
            7.5493e-01,
            8.5794e-01,
            9.6990e-01,
            1.0912e+00,
            1.2221e+00,
            1.3630e+00,
            1.5144e+00,
            1.6763e+00,
        ]
    )
    / u.s  # type: ignore
)


@u.quantity_input
def Q_HCOp_i(
    index: int | NDArray[np.int_],
    Tex: u.K = 5 * u.K,  # type: ignore
) -> float | NDArray[np.float64]:
    """
    The function returns the individual elements of the partition function:
    the occupancy of each level dependent on degeneracy and energy level
    for a given excitation temperature.

    Parameters
    ----------
    index : int | NDArray[np.int_]
        The index of the energy level.
    Tex : u.K
        The excitation temperature.
    Returns
    -------
    float | NDArray[np.float64]
        The occupancy of the level.
    """
    return gu_list[index] * np.exp(-E_u_list[index] / Tex)


@u.quantity_input
def Q_HCOp(
    Tex: u.K = 5 * u.K,  # type: ignore
) -> float | NDArray[np.float64]:
    """
    It returns the partition function for HCO^+ with an excitation
    temperature.
    It uses all 31 energy levels (J=0-30).

    Parameters
    ----------
    Tex : u.K
        The excitation temperature(s).
    Returns
    -------
    Q_HCOp_all : float | NDArray[np.float64]
        The partition function(s).
    """
    if Tex.size == 1:
        return np.sum(Q_HCOp_i(full_index, Tex=Tex))
    else:
        Q_HCOp_all = np.zeros_like(Tex.value)
        for i in range(Tex.size):
            Q_HCOp_all[i] = np.sum(Q_HCOp_i(full_index, Tex=Tex[i]))
        return Q_HCOp_all


@u.quantity_input
def HCOp_thin(
    J_up: int = 1,
    Tex: u.K = 5 * u.K,  # type: ignore
    TdV: u.K * u.km / u.s = 1.0 * u.K * u.km / u.s,  # type: ignore
    T_bg: u.K = 2.73 * u.K,  # type: ignore
) -> u.cm**-2:  # type: ignore
    """
    Total column density determination from the HCO+ J_up -> J_up-1 transition.
    The A_ul, frequency and Einstein coefficient are obtained from LAMBDA database.

    Parameters
    ----------
    J_up : int
        The upper level of the transition.
    Tex : u.K
        The excitation temperature.
    TdV : u.K * u.km / u.s
        The integrated intensity of the transition.
    T_bg : u.K
        The background temperature.

    Returns
    -------
    Ncol : u.cm**-2
        The column density.
    """
    if J_up < np.size(Aij_list):
        freq = freq_list[J_up - 1]
        A_ul = Aij_list[J_up - 1]
    else:
        print("J_up is not available")
        return np.nan * u.cm**-2  # type: ignore
    Jex = J_nu(Tex=Tex, freq=freq)
    Jbg = J_nu(Tex=T_bg, freq=freq)
    Ncol = (
        (8 * np.pi * freq**3 / c**3)
        * Q_HCOp(Tex=Tex)
        / A_ul
        / Q_HCOp_i(J_up, Tex=Tex)
        / (np.exp(h * freq / k_B / Tex) - 1)
        * TdV
        / (Jex - Jbg)
    )
    return Ncol
