"""
Generate LaTeX tables for the manuscript.

Table 1: Stellar properties and ages
Table 2: Planet properties and cumulative XUV fluxes
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catalog import LIST_SYSTEMS
from engleAge import (
    fdaComputeAgeInYears,
    fdaComputeLogAgeDistribution,
    fsSelectSpectralClass,
    fsSelectXUVModel,
    ftComputeAgeSummary,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fsFormatUncertainty(dValue, dSigma, iDecimals=3):
    """Format a value with its uncertainty."""
    return f"${dValue:.{iDecimals}f} \\pm {dSigma:.{iDecimals}f}$"


def fnWriteAgeTable(sOutputPath):
    """Write LaTeX table of stellar properties and ages."""
    listLines = [
        "\\begin{deluxetable*}{lccccccc}",
        "\\tablecaption{Stellar Properties and Ages"
        "\\label{tab:ages}}",
        "\\tablehead{",
        "\\colhead{System} & \\colhead{Sp.~Type} & "
        "\\colhead{$d$ [pc]} & "
        "\\colhead{$M_\\star$ [$M_\\odot$]} & "
        "\\colhead{$P_{\\rm rot}$ [d]} & "
        "\\colhead{Calibration} & "
        "\\colhead{Age [Gyr]} & "
        "\\colhead{XUV Model}",
        "}",
        "\\startdata",
    ]

    for dictSystem in LIST_SYSTEMS:
        sSpectralClass = fsSelectSpectralClass(
            dictSystem["dStellarMass"]
        )
        sXUVModel = fsSelectXUVModel(dictSystem["dStellarMass"])

        sAgePath = os.path.join(
            REPO_ROOT, "systems", dictSystem["sSystemName"],
            "age_samples.txt",
        )
        if os.path.exists(sAgePath):
            daAgeYears = np.loadtxt(sAgePath)
            daAgeGyr = daAgeYears / 1e9
            dMeanAge = np.mean(daAgeGyr)
            dLowerAge = np.percentile(daAgeGyr, 2.5)
            dUpperAge = np.percentile(daAgeGyr, 97.5)
            sAge = (
                f"${dMeanAge:.2f}"
                f"_{{-{dMeanAge - dLowerAge:.2f}}}"
                f"^{{+{dUpperAge - dMeanAge:.2f}}}$"
            )
        else:
            sAge = "\\nodata"

        bOverride = "dAgeOverrideGyr" in dictSystem
        sCalibration = sSpectralClass.capitalize()
        if bOverride:
            sCalibration += "$^\\dagger$"

        sLine = (
            f"{dictSystem['sStarName']} & "
            f"{dictSystem['sSpectralType']} & "
            f"{dictSystem['dDistance']:.2f} & "
            f"{fsFormatUncertainty(dictSystem['dStellarMass'], dictSystem['dStellarMassSigma'])} & "
            f"{fsFormatUncertainty(dictSystem['dRotationPeriod'], dictSystem['dRotationPeriodSigma'], 1)} & "
            f"{sCalibration} & "
            f"{sAge} & "
            f"{sXUVModel} \\\\"
        )
        listLines.append(sLine)

    listLines.extend([
        "\\enddata",
        "\\tablecomments{Ages are computed from the Engle \\& Guinan "
        "(2023) rotation-age relationship with Monte Carlo sampling "
        "($10^5$ draws). Uncertainties are 95\\% confidence intervals. "
        "$^\\dagger$Age override from literature "
        "(see text).}",
        "\\end{deluxetable*}",
    ])

    with open(sOutputPath, "w") as fh:
        fh.write("\n".join(listLines))
    print(f"Age table saved to {sOutputPath}")


def fnWriteFluxTable(dictAllResults, sOutputPath):
    """Write LaTeX table of planet properties and XUV fluxes."""
    listLines = [
        "\\begin{deluxetable*}{lcccccccc}",
        "\\tablecaption{Planet Properties and Cumulative XUV Fluxes"
        "\\label{tab:fluxes}}",
        "\\tablehead{",
        "\\colhead{Planet} & "
        "\\colhead{$P_{\\rm orb}$ [d]} & "
        "\\colhead{$M_p$ [$M_\\oplus$]} & "
        "\\colhead{$v_{\\rm esc}$ [km/s]} & "
        "\\colhead{$F_{\\rm XUV}/F_\\oplus$} & "
        "\\colhead{$F_{\\rm shore}/F_\\oplus$} & "
        "\\colhead{$\\Delta_{\\rm shore}$} & "
        "\\colhead{95\\% CI} & "
        "\\colhead{Atmosphere}",
        "}",
        "\\startdata",
    ]

    for sSystemName, dictSystem in dictAllResults.items():
        for dictPlanet in dictSystem["listPlanetResults"]:
            dMeanFlux = dictPlanet["dMeanNormalizedFlux"]
            dLowerFlux = dictPlanet["dLower95NormalizedFlux"]
            dUpperFlux = dictPlanet["dUpper95NormalizedFlux"]
            dShoreDist = dictPlanet["dMeanShorelineDistance"]

            sFlux = (
                f"${dMeanFlux:.1f}"
                f"_{{-{dMeanFlux - dLowerFlux:.1f}}}"
                f"^{{+{dUpperFlux - dMeanFlux:.1f}}}$"
            )
            sCI = f"[{dLowerFlux:.1f}, {dUpperFlux:.1f}]"

            if dShoreDist > 1:
                sAtmosphere = "Vulnerable"
            else:
                sAtmosphere = "Protected"

            sMassSuffix = "$^\\ddagger$" if dictPlanet.get("bMsini", False) else ""
            sLine = (
                f"{dictSystem['sStarName']} "
                f"{dictPlanet['sPlanetName']} & "
                f"{dictPlanet['dOrbPeriod']:.4f} & "
                f"{dictPlanet['dMass']:.2f}{sMassSuffix} & "
                f"{dictPlanet['dEscapeVelocity']:.1f} & "
                f"{sFlux} & "
                f"{dictPlanet['dShorelineFlux']:.1f} & "
                f"{dShoreDist:.2f} & "
                f"{sCI} & "
                f"{sAtmosphere} \\\\"
            )
            listLines.append(sLine)

    listLines.extend([
        "\\enddata",
        "\\tablecomments{Cumulative XUV fluxes normalized to Earth's "
        "value ($F_\\oplus = 9.76 \\times 10^{15}$ W/m$^2$). "
        "$F_{\\rm shore}$ is the cosmic shoreline flux "
        "(Zahnle \\& Catling 2017). "
        "$\\Delta_{\\rm shore} = F_{\\rm XUV}/F_{\\rm shore}$; "
        "values $> 1$ indicate the planet lies on the "
        "atmosphere-vulnerable side of the shoreline. "
        "$^\\ddagger$Mass is $M \\sin i$ (minimum mass).}",
        "\\end{deluxetable*}",
    ])

    with open(sOutputPath, "w") as fh:
        fh.write("\n".join(listLines))
    print(f"Flux table saved to {sOutputPath}")


if __name__ == "__main__":
    sManuscriptDir = os.path.join(REPO_ROOT, "manuscript")
    os.makedirs(sManuscriptDir, exist_ok=True)

    fnWriteAgeTable(os.path.join(sManuscriptDir, "table_ages.tex"))

    sResultsPath = os.path.join(REPO_ROOT, "results.json")
    if os.path.exists(sResultsPath):
        with open(sResultsPath, "r") as fh:
            dictAllResults = json.load(fh)
        fnWriteFluxTable(
            dictAllResults,
            os.path.join(sManuscriptDir, "table_fluxes.tex"),
        )
    else:
        print("No results.json found; skipping flux table.")
