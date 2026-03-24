"""
Generate two results LaTeX tables for the manuscript.

Table 1 (Stellar): Star Name, Distance, Mass, Rotation Period, Age,
    References (stellar mass, rotation, age).

Table 2 (Planetary): Star Name, Planet Name, Orbital Period,
    Semi-major Axis, Cumulative XUV Flux, Cosmic Shoreline Distance,
    References (planet discovery/characterization).

Stars ordered by distance; multi-planet systems list planets by
increasing orbital period (innermost first).
"""

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catalog import LIST_SYSTEMS
from cosmicShoreline import (
    D_CUMULATIVE_EARTH_FLUX,
    fdEscapeVelocityEarthUnits,
    fdShorelineFlux,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AU_IN_METERS = 1.496e11


# -------------------------------------------------------------------
# Data extraction
# -------------------------------------------------------------------


def fdaExtractSemiMajorAxes(sSystemDir, sSystemName, sPlanetName):
    """Extract semi-major axis values from trial log files.

    Returns array of semi-major axis values in AU.
    """
    sOutputDir = os.path.join(sSystemDir, "output")
    listSMA = []
    for sTrialDir in sorted(os.listdir(sOutputDir)):
        sLogPath = os.path.join(
            sOutputDir, sTrialDir, f"{sSystemName}.log"
        )
        if not os.path.isfile(sLogPath):
            continue
        dSMA = fdParseSMAFromLog(sLogPath, sPlanetName)
        if dSMA is not None:
            listSMA.append(dSMA)
    return np.array(listSMA)


def fdParseSMAFromLog(sLogPath, sPlanetName):
    """Parse a single log file to extract SMA for a given planet."""
    sCurrentBody = None
    with open(sLogPath, "r") as fileHandle:
        for sLine in fileHandle:
            if "----- BODY:" in sLine:
                sCurrentBody = sLine.split("BODY:")[1].split()[0]
            if (
                sCurrentBody == sPlanetName
                and "SemiMajorAxis" in sLine
                and "Critical" not in sLine
            ):
                dSMA = float(sLine.split(":")[-1].strip())
                return dSMA / AU_IN_METERS
    return None


def fdaExtractFluxDistribution(sSystemDir, sPlanetName):
    """Extract cumulative XUV flux distribution from converged output.

    Returns array of normalized flux values (in Earth units).
    """
    sJsonPath = os.path.join(
        sSystemDir, "output", "Converged_Param_Dictionary.json"
    )
    with open(sJsonPath, "r") as fileHandle:
        sContent = fileHandle.read().strip()
        if sContent.startswith('"') and sContent.endswith('"'):
            sContent = sContent[1:-1].replace('\\"', '"')
        dictData = json.loads(sContent)

    sKey = f"{sPlanetName},CumulativeXUVFlux,final"
    daFlux = np.array(dictData.get(sKey, []))
    daFlux = daFlux[daFlux > 0]
    return daFlux / D_CUMULATIVE_EARTH_FLUX


def fdaExtractAgeDistribution(sSystemDir):
    """Extract age distribution from age_samples.txt.

    Returns array of ages in Gyr.
    """
    sPath = os.path.join(sSystemDir, "age_samples.txt")
    daAgeYears = np.loadtxt(sPath)
    return daAgeYears / 1e9


# -------------------------------------------------------------------
# Formatting helpers
# -------------------------------------------------------------------


def fsFormatAsymmetric(dLower, dMedian, dUpper, iDecimals):
    """Format a value with asymmetric uncertainties."""
    sFmt = f".{iDecimals}f"
    dErrLow = dMedian - dLower
    dErrHigh = dUpper - dMedian
    return (
        f"${dMedian:{sFmt}}"
        f"_{{-{dErrLow:{sFmt}}}}"
        f"^{{+{dErrHigh:{sFmt}}}}$"
    )


def fsFormatSymmetric(dValue, dSigma, iDecimals):
    """Format a value with symmetric uncertainty."""
    sFmt = f".{iDecimals}f"
    return f"${dValue:{sFmt}} \\pm {dSigma:{sFmt}}$"


def fsFormatShortRef(sRef):
    """Shorten a reference to Author+Year format."""
    sRef = sRef.strip()
    matchResult = re.match(r"(\w[\w\s&.\-]+?)\s*\((\d{4})\)", sRef)
    if matchResult:
        sAuthor = matchResult.group(1).strip()
        sYear = matchResult.group(2)
        sFirstAuthor = sAuthor.split(",")[0].split("&")[0].strip()
        if "," in sAuthor or "&" in sAuthor or "et al" in sAuthor:
            return f"{sFirstAuthor}+{sYear}"
        return f"{sFirstAuthor} {sYear}"
    return sRef[:20]


def flistDeduplicateShortRefs(listRefs):
    """Shorten and deduplicate a list of full reference strings."""
    return list(dict.fromkeys(fsFormatShortRef(r) for r in listRefs))


def flistSortSystemsByDistance(listSystems):
    """Return systems sorted by distance."""
    return sorted(listSystems, key=lambda d: d["dDistance"])


def fbSystemHasConvergedOutput(sSystemDir):
    """Return True if the system has converged vconverge output."""
    sJsonPath = os.path.join(
        sSystemDir, "output", "Converged_Param_Dictionary.json"
    )
    return os.path.exists(sJsonPath)


# -------------------------------------------------------------------
# Age column
# -------------------------------------------------------------------


def fsFormatAgeColumn(daAgeGyr):
    """Format the stellar age column from distribution."""
    if len(daAgeGyr) == 0:
        return "\\nodata"
    dLower = np.percentile(daAgeGyr, 16)
    dMedian = np.percentile(daAgeGyr, 50)
    dUpper = np.percentile(daAgeGyr, 84)
    return fsFormatAsymmetric(dLower, dMedian, dUpper, 2)


# -------------------------------------------------------------------
# Cosmic shoreline distance
# -------------------------------------------------------------------


def fdComputeShorelineFluxForPlanet(dictPlanet):
    """Compute the cosmic shoreline flux for a planet.

    Uses catalog mass and the empirical R = M^0.27 scaling.
    """
    dMass = dictPlanet.get("dMass", 1.0)
    if dMass is None:
        dMass = 1.0
    dRadius = dMass ** 0.27
    dEscVel = fdEscapeVelocityEarthUnits(dMass, dRadius)
    return fdShorelineFlux(dEscVel)


def fsFormatShorelineDistanceColumn(daFlux, dShorelineFlux):
    """Format the cosmic shoreline distance from flux distribution."""
    if len(daFlux) == 0:
        return "\\nodata"
    daDist = daFlux / dShorelineFlux
    dLower = np.percentile(daDist, 16)
    dMedian = np.percentile(daDist, 50)
    dUpper = np.percentile(daDist, 84)
    return fsFormatAsymmetric(dLower, dMedian, dUpper, 2)


# -------------------------------------------------------------------
# SMA and flux columns
# -------------------------------------------------------------------


def fsFormatSMAColumn(daSMA):
    """Format the semi-major axis column from distribution."""
    if len(daSMA) == 0:
        return "\\nodata"
    dLower = np.percentile(daSMA, 16)
    dMedian = np.percentile(daSMA, 50)
    dUpper = np.percentile(daSMA, 84)
    return fsFormatAsymmetric(dLower, dMedian, dUpper, 4)


def fsFormatFluxColumn(daFlux):
    """Format the cumulative XUV flux column from distribution."""
    if len(daFlux) == 0:
        return "\\nodata"
    dLower = np.percentile(daFlux, 16)
    dMedian = np.percentile(daFlux, 50)
    dUpper = np.percentile(daFlux, 84)
    dErrLow = dMedian - dLower
    dErrHigh = dUpper - dMedian
    return (
        f"${dMedian:.1f}"
        f"_{{-{dErrLow:.1f}}}"
        f"^{{+{dErrHigh:.1f}}}$"
    )


# -------------------------------------------------------------------
# Reference collection (split by table)
# -------------------------------------------------------------------


def flistCollectStellarReferences(dictSystem):
    """Collect unique references for the stellar properties table."""
    listRefs = []
    for sRef in [
        dictSystem.get("sMassRef", ""),
        dictSystem.get("sRotationRef", ""),
        dictSystem.get("sAgeRef", ""),
    ]:
        if sRef and sRef not in listRefs:
            listRefs.append(sRef)
    return listRefs


def flistCollectPlanetReferences(dictPlanet):
    """Collect unique references for the planet properties table."""
    sRef = dictPlanet.get("sRef", "")
    return [sRef] if sRef else []


# -------------------------------------------------------------------
# Table 1: Stellar properties
# -------------------------------------------------------------------


def flistFormatStellarTableHeader():
    """Return header lines for the stellar properties table."""
    return [
        "\\begin{longrotatetable}",
        "\\begin{deluxetable*}{lcccccl}",
        "\\tablecaption{Stellar Properties"
        "\\label{tab:stellar}}",
        "\\tabletypesize{\\scriptsize}",
        "\\tablewidth{0pt}",
        "\\tablehead{",
        "\\colhead{Star} & "
        "\\colhead{$d$ [pc]} & "
        "\\colhead{$M_\\star$ [$M_\\odot$]} & "
        "\\colhead{$P_{\\rm rot}$ [d]} & "
        "\\colhead{Age [Gyr]} & "
        "\\colhead{$N_{\\rm pl}$} & "
        "\\colhead{References}",
        "}",
        "\\startdata",
    ]


def flistFormatStellarTableFooter():
    """Return footer lines for the stellar properties table."""
    return [
        "\\enddata",
        "\\tablecomments{"
        "Stellar mass and rotation period uncertainties are "
        "$1\\sigma$ from the literature. "
        "Ages are 16\\%, 50\\%, and 84\\% confidence intervals "
        "derived from the Engle \\& Guinan (2023) gyrochronology "
        "relation, except where noted. "
        "See Table~\\ref{tab:planets} for planet properties.}",
        "\\end{deluxetable*}",
        "\\end{longrotatetable}",
    ]


def fsFormatStellarRow(dictSystem, sSystemDir):
    """Format a single row of the stellar properties table."""
    sStarCol = dictSystem["sStarName"]
    sDistCol = f"{dictSystem['dDistance']:.2f}"
    sMassCol = fsFormatSymmetric(
        dictSystem["dStellarMass"],
        dictSystem["dStellarMassSigma"],
        3,
    )
    sRotCol = fsFormatSymmetric(
        dictSystem["dRotationPeriod"],
        dictSystem["dRotationPeriodSigma"],
        1,
    )
    daAgeGyr = fdaExtractAgeDistribution(sSystemDir)
    sAgeCol = fsFormatAgeColumn(daAgeGyr)
    iNumPlanets = len(dictSystem["listPlanets"])
    listRefs = flistCollectStellarReferences(dictSystem)
    sRefCol = "; ".join(flistDeduplicateShortRefs(listRefs))
    return (
        f"{sStarCol} & {sDistCol} & {sMassCol} & "
        f"{sRotCol} & {sAgeCol} & {iNumPlanets} & {sRefCol} \\\\"
    )


# -------------------------------------------------------------------
# Table 2: Planet properties
# -------------------------------------------------------------------


def flistFormatPlanetTableHeader():
    """Return header lines for the planet properties table."""
    return [
        "\\begin{longrotatetable}",
        "\\begin{deluxetable*}{llccccccl}",
        "\\tablecaption{Planet Properties and Cumulative XUV Fluxes"
        "\\label{tab:planets}}",
        "\\tabletypesize{\\scriptsize}",
        "\\tablewidth{0pt}",
        "\\tablehead{",
        "\\colhead{Star} & "
        "\\colhead{Planet} & "
        "\\colhead{$P_{\\rm orb}$ [d]} & "
        "\\colhead{$a$ [AU]} & "
        "\\colhead{$F_{\\rm XUV}/F_\\oplus$} & "
        "\\colhead{$D_{\\rm shore}$} & "
        "\\colhead{$M_{\\rm p}$ [$M_\\oplus$]} & "
        "\\colhead{$v_{\\rm esc}$ [km/s]} & "
        "\\colhead{References}",
        "}",
        "\\startdata",
    ]


def flistFormatPlanetTableFooter():
    """Return footer lines for the planet properties table."""
    return [
        "\\enddata",
        "\\tablecomments{"
        "Cumulative XUV fluxes normalized to Earth's value "
        "($F_\\oplus = 9.76 \\times 10^{15}$ W/m$^2$). "
        "Semi-major axis, cumulative XUV flux, and cosmic "
        "shoreline distance uncertainties are 16\\%, 50\\%, "
        "and 84\\% confidence intervals from "
        "the vconverge Monte Carlo ensemble. "
        "Orbital period uncertainties are $1\\sigma$ from "
        "the literature. "
        "$D_{\\rm shore} = F_{\\rm XUV} / F_{\\rm shore}$, "
        "the ratio of cumulative XUV flux to the cosmic "
        "shoreline flux at the planet's escape velocity "
        "(Zahnle \\& Catling 2017); "
        "values $> 1$ indicate greater exposure than the "
        "shoreline prediction. "
        "Planet radii estimated via $R = M^{0.27}$ "
        "(Earth units). "
        "$^\\ddagger$Mass is $M \\sin i$ (minimum mass).}",
        "\\end{deluxetable*}",
        "\\end{longrotatetable}",
    ]


def fsFormatPlanetRow(
    dictSystem, dictPlanet, sSystemDir, bFirstPlanet
):
    """Format a single row of the planet properties table."""
    sStarCol = dictSystem["sStarName"] if bFirstPlanet else ""
    sPlanetName = dictPlanet["sName"]
    sMsiniMark = "$^\\ddagger$" if dictPlanet.get("bMsini", False) else ""
    sPlanetCol = f"{sPlanetName}{sMsiniMark}"
    sOrbPerCol = fsFormatSymmetric(
        dictPlanet["dOrbPeriod"], dictPlanet["dOrbPeriodSigma"], 4
    )

    daSMA = fdaExtractSemiMajorAxes(
        sSystemDir, dictSystem["sSystemName"], sPlanetName
    )
    daFlux = fdaExtractFluxDistribution(sSystemDir, sPlanetName)
    sSMACol = fsFormatSMAColumn(daSMA)
    sFluxCol = fsFormatFluxColumn(daFlux)

    dShoreFlux = fdComputeShorelineFluxForPlanet(dictPlanet)
    sShoreDistCol = fsFormatShorelineDistanceColumn(daFlux, dShoreFlux)

    dMass = dictPlanet.get("dMass", 1.0)
    if dMass is None:
        dMass = 1.0
    dRadius = dMass ** 0.27
    dEscVel = fdEscapeVelocityEarthUnits(dMass, dRadius)
    sMassCol = f"{dMass:.2f}"
    sEscVelCol = f"{dEscVel:.1f}"

    listRefs = flistCollectPlanetReferences(dictPlanet)
    sRefCol = "; ".join(flistDeduplicateShortRefs(listRefs))

    return (
        f"{sStarCol} & {sPlanetCol} & {sOrbPerCol} & "
        f"{sSMACol} & {sFluxCol} & {sShoreDistCol} & "
        f"{sMassCol} & {sEscVelCol} & {sRefCol} \\\\"
    )


# -------------------------------------------------------------------
# Document wrapper
# -------------------------------------------------------------------


def flistFormatDocumentHeader():
    """Return preamble lines for a standalone AASTeX document."""
    return [
        "\\documentclass{aastex631}",
        "\\begin{document}",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "",
    ]


def flistFormatDocumentFooter():
    """Return closing lines for a standalone AASTeX document."""
    return [
        "",
        "\\end{document}",
    ]


# -------------------------------------------------------------------
# Main table writers
# -------------------------------------------------------------------


def fnWriteStellarTable(listSorted, listLines):
    """Append the stellar properties table to listLines."""
    listLines.extend(flistFormatStellarTableHeader())
    for dictSystem in listSorted:
        sSystemDir = os.path.join(
            REPO_ROOT, "systems", dictSystem["sSystemName"]
        )
        if not fbSystemHasConvergedOutput(sSystemDir):
            continue
        listLines.append(fsFormatStellarRow(dictSystem, sSystemDir))
    listLines.extend(flistFormatStellarTableFooter())


def fnWritePlanetTable(listSorted, listLines):
    """Append the planet properties table to listLines."""
    listLines.extend(flistFormatPlanetTableHeader())
    for dictSystem in listSorted:
        sSystemDir = os.path.join(
            REPO_ROOT, "systems", dictSystem["sSystemName"]
        )
        if not fbSystemHasConvergedOutput(sSystemDir):
            continue
        listPlanets = sorted(
            dictSystem["listPlanets"], key=lambda p: p["dOrbPeriod"]
        )
        bFirstPlanet = True
        for dictPlanet in listPlanets:
            listLines.append(
                fsFormatPlanetRow(
                    dictSystem, dictPlanet, sSystemDir, bFirstPlanet
                )
            )
            bFirstPlanet = False
    listLines.extend(flistFormatPlanetTableFooter())


def fnWriteResultsTables(sOutputPath):
    """Write both results tables to a single LaTeX file."""
    listSorted = flistSortSystemsByDistance(LIST_SYSTEMS)
    listLines = flistFormatDocumentHeader()
    fnWriteStellarTable(listSorted, listLines)
    listLines.append("")
    fnWritePlanetTable(listSorted, listLines)
    listLines.extend(flistFormatDocumentFooter())

    with open(sOutputPath, "w") as fileHandle:
        fileHandle.write("\n".join(listLines))
    print(f"Results tables saved to {sOutputPath}")


if __name__ == "__main__":
    sManuscriptDir = os.path.join(REPO_ROOT, "manuscript")
    os.makedirs(sManuscriptDir, exist_ok=True)
    fnWriteResultsTables(
        os.path.join(sManuscriptDir, "table_results.tex")
    )
