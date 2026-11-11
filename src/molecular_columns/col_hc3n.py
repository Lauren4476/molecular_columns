import numpy as np
import astropy.units as u
from numpy.typing import NDArray
from astropy.constants import c, k_B, h  # type: ignore

from .common_functions import J_nu

# g_u and E_u values obtained from LAMDA database
# https://home.strw.leidenuniv.nl/~moldata/datafiles/hc3n-h2.dat
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
        63.0,
        65.0,
        67.0,
        69.0,
        71.0,
        73.0,
        75.0,
    ]
)
E_u_list = (
    np.array(
        [
            0.0000,
            0.3035,
            0.9104,
            1.8209,
            3.0348,
            4.5522,
            6.3731,
            8.4974,
            10.9252,
            13.6565,
            16.6912,
            20.0294,
            23.6710,
            27.6161,
            31.8647,
            36.4166,
            41.2720,
            46.4308,
            51.8930,
            57.6587,
            63.7277,
            70.1001,
            76.7759,
            83.7551,
            91.0376,
            98.6235,
            106.5127,
            114.7053,
            123.2011,
            132.0003,
            141.1027,
            150.5085,
            160.2175,
            170.2297,
            180.5452,
            191.1639,
            202.0858,
            213.3109,
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
            9.09812,
            18.19623,
            27.29429,
            36.39232,
            45.49031,
            54.58825,
            63.68605,
            72.78382,
            81.88147,
            90.97902,
            100.07639,
            109.17363,
            118.27074,
            127.36767,
            136.46441,
            145.56096,
            154.65728,
            163.75339,
            172.84930,
            181.94492,
            191.04030,
            200.13539,
            209.23023,
            218.32472,
            227.41890,
            236.51279,
            245.60632,
            254.69950,
            263.79231,
            272.88475,
            281.97679,
            291.06843,
            300.15965,
            309.25043,
            318.34078,
            327.43067,
            336.52008,
        ]
    )
    * u.GHz  # type: ignore
)

Aij_list = (
    np.array(
        [
            4.070e-08,
            3.910e-07,
            1.410e-06,
            3.470e-06,
            6.940e-06,
            1.220e-05,
            1.950e-05,
            2.940e-05,
            4.210e-05,
            5.810e-05,
            7.770e-05,
            1.010e-04,
            1.290e-04,
            1.620e-04,
            1.990e-04,
            2.420e-04,
            2.910e-04,
            3.460e-04,
            4.080e-04,
            4.760e-04,
            5.520e-04,
            6.350e-04,
            7.260e-04,
            8.260e-04,
            9.350e-04,
            1.050e-03,
            1.180e-03,
            1.320e-03,
            1.460e-03,
            1.620e-03,
            1.790e-03,
            1.970e-03,
            2.160e-03,
            2.360e-03,
            2.580e-03,
            2.810e-03,
            3.050e-03,
        ]
    )
    / u.s  # type: ignore
)


@u.quantity_input
def Q_HC3N_i(
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
def Q_HC3N(
    Tex: u.K = 5 * u.K,  # type: ignore
) -> float | NDArray[np.float64]:
    """
    It returns the partition function for HC_3N with an excitation
    temperature.
    It uses all 38 energy levels (J=0-37).

    Parameters
    ----------
    Tex : u.K
        The excitation temperature(s).
    Returns
    -------
    Q_HC3N_all : float | NDArray[np.float64]
        The partition function(s).
    """
    if Tex.size == 1:
        return np.sum(Q_HC3N_i(full_index, Tex=Tex))
    else:
        Q_HC3N_all = np.zeros_like(Tex.value)
        for i in range(Tex.size):
            Q_HC3N_all[i] = np.sum(Q_HC3N_i(full_index, Tex=Tex[i]))
        return Q_HC3N_all


@u.quantity_input
def HC3N_thin(
    J_up: int = 1,
    Tex: u.K = 5 * u.K,  # type: ignore
    TdV: u.K * u.km / u.s = 1.0 * u.K * u.km / u.s,  # type: ignore
    T_bg: u.K = 2.73 * u.K,  # type: ignore
) -> u.cm**-2:  # type: ignore
    """
    Total column density determination from the HC3N J_up -> J_up-1 transition.
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
        * Q_HC3N(Tex=Tex)
        / A_ul
        / Q_HC3N_i(J_up, Tex=Tex)
        / (np.exp(h * freq / k_B / Tex) - 1)
        * TdV
        / (Jex - Jbg)
    )
    return Ncol
