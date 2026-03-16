"""
Compute stellar age distributions from the Engle & Guinan (2023)
age-rotation relationship for M dwarfs.

The model relates log10(age/Gyr) to rotation period via a two-part
linear function:

    tau = a * P_rot + b                          if P_rot < d
    tau = a * P_rot + b + c * (P_rot - d)        if P_rot >= d

Three calibrations exist:
    Early  (M0-M2):     a=0.0621, b=-1.0437, c=-0.0528, d=-23.4933
    Mid    (M2.5-M3.5): a=0.0561, b=-0.8900, c=-0.0521, d=-24.1888
    Late   (M4-M6.5):   a=0.0251, b=-0.1615, c=-0.0212, d=-25.45

References:
    Engle & Guinan (2023), ApJ, 954, 118
"""

import numpy as np

DICT_ENGLE_ROTATION_COEFFICIENTS = {
    "early": {
        "daA": (0.0621, 0.0047),
        "daB": (-1.0437, 0.0737),
        "daC": (-0.0528, 0.0047),
        "daD": (23.4933, 2.0268),
    },
    "mid": {
        "daA": (0.0561, 0.0064),
        "daB": (-0.8900, 0.0994),
        "daC": (-0.0521, 0.0064),
        "daD": (24.1888, 2.5133),
    },
    "late": {
        "daA": (0.0251, 0.0018),
        "daB": (-0.1615, 0.0303),
        "daC": (-0.0212, 0.0018),
        "daD": (25.45, 1.9079),
    },
}

D_MAX_LOG_AGE = np.log10(13)


def fdaComputeLogAgeDistribution(
    daRotationPeriod, sSpectralClass, iNumSamples=100000
):
    """Return an array of log10(age/Gyr) samples drawn from the
    Engle & Guinan (2023) age-rotation relationship.

    Parameters
    ----------
    daRotationPeriod : tuple of (float, float)
        (mean, sigma) for the stellar rotation period in days.
    sSpectralClass : str
        One of 'early', 'mid', or 'late'.
    iNumSamples : int
        Number of Monte Carlo draws.

    Returns
    -------
    daLogAge : numpy array
        Filtered log10(age/Gyr) samples with age <= 13 Gyr.
    """
    dictCoefficients = DICT_ENGLE_ROTATION_COEFFICIENTS[sSpectralClass]

    daASamples = np.random.normal(
        dictCoefficients["daA"][0], dictCoefficients["daA"][1], iNumSamples
    )
    daBSamples = np.random.normal(
        dictCoefficients["daB"][0], dictCoefficients["daB"][1], iNumSamples
    )
    daCSamples = np.random.normal(
        dictCoefficients["daC"][0], dictCoefficients["daC"][1], iNumSamples
    )
    daDSamples = np.random.normal(
        dictCoefficients["daD"][0], dictCoefficients["daD"][1], iNumSamples
    )
    daRotPerSamples = np.random.normal(
        daRotationPeriod[0], daRotationPeriod[1], iNumSamples
    )

    daLogAge = daASamples * daRotPerSamples + daBSamples
    baPiecewise = daRotPerSamples >= daDSamples
    daLogAge[baPiecewise] += (
        daCSamples[baPiecewise]
        * (daRotPerSamples[baPiecewise] - daDSamples[baPiecewise])
    )
    daLogAge = daLogAge[daLogAge <= D_MAX_LOG_AGE]

    return daLogAge


def fdaComputeAgeInYears(daLogAge):
    """Convert log10(age/Gyr) samples to age in years."""
    return 10 ** daLogAge * 1e9


def ftComputeAgeSummary(daLogAge):
    """Return (mean, lower_95, upper_95) for age in Gyr."""
    daAgeGyr = 10 ** daLogAge
    dMean = np.mean(daAgeGyr)
    dLower = np.percentile(daAgeGyr, 2.5)
    dUpper = np.percentile(daAgeGyr, 97.5)
    return dMean, dLower, dUpper


def fsSelectSpectralClass(dStellarMass):
    """Return the Engle spectral class string based on stellar mass.

    Approximate mass-spectral type boundaries for M dwarfs:
        M0-M2:     0.40 - 0.60 Msun
        M2.5-M3.5: 0.25 - 0.40 Msun
        M4-M6.5:   0.08 - 0.25 Msun

    Parameters
    ----------
    dStellarMass : float
        Stellar mass in solar masses.

    Returns
    -------
    str : 'early', 'mid', or 'late'
    """
    if dStellarMass >= 0.40:
        return "early"
    elif dStellarMass >= 0.25:
        return "mid"
    else:
        return "late"


def fsSelectXUVModel(dStellarMass):
    """Return the vplanet sXUVModel string based on stellar mass.

    The Engle (2024) XUV model has two calibrations:
        Early   (M0-M2):     Engle24Early
        MidLate (M2.6-M6.5): Engle24MidLate

    Parameters
    ----------
    dStellarMass : float
        Stellar mass in solar masses.

    Returns
    -------
    str : vplanet sXUVModel string
    """
    if dStellarMass >= 0.40:
        return "Engle24Early"
    else:
        return "Engle24MidLate"
