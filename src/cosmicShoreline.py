"""
Compute the cosmic shoreline flux and distance for a planet.

The cosmic shoreline (Zahnle & Catling 2017) is a line in
escape velocity vs. cumulative XUV flux space that separates
planets with atmospheres from those without. The line is
calibrated to the Solar System.

The cosmic shoreline is defined as:
    F_XUV_cumulative / F_XUV_Earth = (v_esc / v_esc_Earth)^alpha

where alpha is approximately 1 (linear in log-log space from the
Solar System data).

The normalized cumulative XUV flux for Earth is computed using the
Ribas (2005) model with t_sat = 100 Myr and f_sat = 1e-3.
"""

import numpy as np

D_CUMULATIVE_EARTH_FLUX = 9.759583e15
D_GRAVITATIONAL_CONSTANT = 6.674e-11
D_EARTH_MASS = 5.972e24
D_EARTH_RADIUS = 6.371e6


def fdEscapeVelocity(dMass, dRadius):
    """Return the escape velocity in m/s.

    Parameters
    ----------
    dMass : float
        Planet mass in kg.
    dRadius : float
        Planet radius in meters.
    """
    return np.sqrt(2 * D_GRAVITATIONAL_CONSTANT * dMass / dRadius)


def fdEscapeVelocityEarthUnits(dMassEarth, dRadiusEarth):
    """Return escape velocity in km/s from Earth-unit inputs."""
    dMass = dMassEarth * D_EARTH_MASS
    dRadius = dRadiusEarth * D_EARTH_RADIUS
    return fdEscapeVelocity(dMass, dRadius) / 1e3


def fdShorelineFlux(dEscapeVelocityKmPerSec):
    """Return the cosmic shoreline flux in Earth-normalized units.

    The shoreline is a line in log(v_esc) vs log(F_norm) space.
    From the Solar System calibration, the slope is approximately
    such that the line passes through (v_esc_Earth, 1) with the
    observed slope from Zahnle & Catling (2017).

    Using the slope from the makeplot.py in GJ1132:
        log(F) goes from -6 to 4 as v_esc goes from 0.2 to 60 km/s
        slope = (4 - (-6)) / (log10(60) - log10(0.2)) = 10/2.477 = 4.04

    But the simpler formulation used in the paper normalizes to
    Earth's escape velocity = 11.19 km/s and Earth flux = 1.

    For GJ 1132 b, the shoreline flux is ~51 Earth units.
    """
    dEarthEscVel = 11.186
    dLogSlope = 10.0 / (np.log10(60) - np.log10(0.2))
    dLogFlux = dLogSlope * (
        np.log10(dEscapeVelocityKmPerSec) - np.log10(dEarthEscVel)
    )
    return 10 ** dLogFlux


def fdNormalizedCumulativeFlux(dCumulativeFluxWm2):
    """Normalize cumulative XUV flux to Earth units."""
    return dCumulativeFluxWm2 / D_CUMULATIVE_EARTH_FLUX


def fdShorelineDistance(dNormalizedFlux, dShorelineFlux):
    """Return the ratio of actual to shoreline flux.

    Values > 1 mean the planet is on the atmosphere-free side.
    Values < 1 mean the planet is on the atmosphere-expected side.
    """
    return dNormalizedFlux / dShorelineFlux
