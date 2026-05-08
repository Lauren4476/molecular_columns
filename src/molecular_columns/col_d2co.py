# Lauren's code for D2CO. Last updated 0805026. May contain errors
# Example use:
'''
df = load_d2co_csv("src/molecular_columns/o-d2co.csv")

freq = 231.410234 * u.GHz
Tex = 20 * u.K
TdV = 1.0 * u.K * u.km / u.s
T_bg = 2.73 * u.K

o_column_density = o_D2CO_thin_csv(df, freq, Tex, TdV, T_bg)
print(f"Ortho-D2CO column density: {o_column_density:.2e}")
'''

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.constants import c, k_B, h  # type: ignore
from .common_functions import J_nu


# Load CSV
def load_d2co_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    return df


# Parse quantum numbers
def parse_qns(qn: str):
    """
    e.g. '10(6,4) - 10(6,5)' → ((10,6,4),(10,6,5))
    """
    upper, lower = qn.split("-")

    def parse(state):
        state = state.strip()
        J = int(state.split("(")[0])
        Ka, Kc = state.split("(")[1].replace(")", "").split(",")
        return (J, int(Ka), int(Kc))

    return parse(upper), parse(lower)


# Build level dictionary for partition function
def build_levels(df: pd.DataFrame):
    levels = {}

    for _, row in df.iterrows():
        upper, lower = parse_qns(row["Resolved QNs"])

        for state in (upper, lower):
            if state not in levels:
                levels[state] = {
                    "E": row["E_U (K)"] * u.K,
                    "g": row["Upper State Degeneracy"],
                }

    return levels


# Partition function
def Q(levels, Tex):
    return sum(lvl["g"] * np.exp(-(lvl["E"] / Tex).decompose().value) for lvl in levels.values())


# Common column density core
def calculate_column_density(row, levels, Tex, TdV, T_bg):
    freq = row["Frequency (MHz)"] * u.MHz
    freq_GHz = freq.to(u.GHz)

    A_ul = (10 ** row["log10_aij"]) / u.s
    E_u = row["E_U (K)"] * u.K
    g_u = row["Upper State Degeneracy"]

    upper, _ = parse_qns(row["Resolved QNs"])
    E_upper_level = levels[upper]["E"]
    g_upper_level = levels[upper]["g"]

    Q_T = Q(levels, Tex)

    Jex = J_nu(Tex=Tex, freq=freq_GHz)
    Jbg = J_nu(Tex=T_bg, freq=freq_GHz)

    Ncol = (
        (8 * np.pi * freq_GHz**3 / c**3)
        * Q_T
        / A_ul
        / (g_upper_level * np.exp(-(E_upper_level / Tex).decompose().value))
        / (np.exp(h * freq_GHz / (k_B * Tex)) - 1)
        * TdV
        / (Jex - Jbg)
    )

    return Ncol


# PARA-D2CO
@u.quantity_input
def p_D2CO_thin_csv(
    df: pd.DataFrame,
    freq: u.GHz,
    Tex: u.K = 5 * u.K,
    TdV: u.K * u.km / u.s = 1.0 * u.K * u.km / u.s,
    T_bg: u.K = 2.73 * u.K,
) -> u.cm**-2:

    df_p = df[df["ortho_or_para"].str.lower() == "p"].copy()
    levels = build_levels(df_p)

    row = df_p[np.isclose(df_p["Frequency (MHz)"], freq.to(u.MHz).value)].iloc[0]

    return calculate_column_density(row, levels, Tex, TdV, T_bg)


# ORTHO-D2CO
@u.quantity_input
def o_D2CO_thin_csv(
    df: pd.DataFrame,
    freq: u.GHz,
    Tex: u.K = 5 * u.K,
    TdV: u.K * u.km / u.s = 1.0 * u.K * u.km / u.s,
    T_bg: u.K = 2.73 * u.K,
) -> u.cm**-2:

    df_o = df[df["ortho_or_para"].str.lower() == "o"].copy()
    levels = build_levels(df_o)

    row = df_o[np.isclose(df_o["Frequency (MHz)"], freq.to(u.MHz).value)].iloc[0]

    return calculate_column_density(row, levels, Tex, TdV, T_bg)