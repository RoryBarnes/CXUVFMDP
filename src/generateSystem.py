"""
Generate VPLanet/vconverge input files for a planetary system to compute
the cumulative XUV flux distribution.

Each system gets a directory under systems/ containing:
    - vpl.in      : primary VPLanet input file
    - star.in     : host star parameters
    - <planet>.in : one per planet with P < 25 days
    - vspace.in   : parameter sweep definition
    - vconverge.in: convergence control file
    - age_samples.txt: age distribution from Engle & Guinan (2023)
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engleAge import (
    fdaComputeAgeInYears,
    fdaComputeLogAgeDistribution,
    fsSelectSpectralClass,
    fsSelectXUVModel,
    ftComputeAgeSummary,
)


def fnWriteVplIn(sOutputDirectory, sSystemName, listPlanetNames):
    """Write the primary vpl.in file."""
    sBodyFiles = "star.in\t" + "\t".join(
        [f"{sName}.in" for sName in listPlanetNames]
    )
    sContent = f"""# Primary input file for {sSystemName}
sSystemName\t{sSystemName}
iVerbose\t0
bOverwrite\t1

saBodyFiles\t{sBodyFiles}

sUnitMass\tsolar
sUnitLength\taU
sUnitTime\tYEAR
sUnitAngle\td
sUnitTemp\tK

bDoLog\t\t1
iDigits\t\t6
dMinValue\t1e-10

bDoForward\t1
bVarDt\t\t1
dEta\t\t1.0e-2
dStopTime\t8.0e9
dOutputTime\t1.0e6
"""
    sFilePath = os.path.join(sOutputDirectory, "vpl.in")
    with open(sFilePath, "w") as fileHandle:
        fileHandle.write(sContent)


def fnWriteStarIn(sOutputDirectory, dMass, sXUVModel):
    """Write the star.in file."""
    dMassForModel = dMass
    if sXUVModel == "Engle24MidLate" and dMass < 0.101:
        dMassForModel = 0.101
    elif sXUVModel == "Engle24Early" and dMass > 0.60:
        dMassForModel = 0.60
    sContent = f"""# Host star parameters
sName\t\tstar
saModules\tstellar

dMass\t\t{dMassForModel}
dRotPeriod\t-1.0
dAge\t\t1.0e6

sStellarModel\tbaraffe
sXUVModel\t{sXUVModel}
bHaltEndBaraffeGrid\t0

saOutputOrder\tTime -Luminosity -LXUVStellar -LXUVTot
"""
    sFilePath = os.path.join(sOutputDirectory, "star.in")
    with open(sFilePath, "w") as fileHandle:
        fileHandle.write(sContent)


def fnWritePlanetIn(sOutputDirectory, sPlanetName, dMass, dOrbPeriod):
    """Write a planet .in file.

    Parameters
    ----------
    dMass : float
        Planet mass in Earth masses (will be negated for vplanet).
    dOrbPeriod : float
        Orbital period in days (will be negated for vplanet).
    """
    sContent = f"""# Planet {sPlanetName} parameters
sName\t\t{sPlanetName}
saModules\tatmesc
dMass\t\t{-abs(dMass)}
dOrbPeriod\t{-abs(dOrbPeriod)}
saOutputOrder\tTime -CumulativeXUVFlux
"""
    sFilePath = os.path.join(sOutputDirectory, f"{sPlanetName}.in")
    with open(sFilePath, "w") as fileHandle:
        fileHandle.write(sContent)


def fnWriteVspaceIn(
    sOutputDirectory,
    dStellarMass,
    dStellarMassSigma,
    listPlanetDicts,
    sXUVModel,
):
    """Write the vspace.in file with Gaussian draws from posteriors.

    Parameters
    ----------
    listPlanetDicts : list of dict
        Each dict has keys: sName, dMass, dMassSigma, dOrbPeriod,
        dOrbPeriodSigma.
    """
    listLines = [
        "srcfolder .",
        "destfolder output",
        "trialname xuv_",
        "samplemode random",
        "randsize 500",
        "",
    ]

    for dictPlanet in listPlanetDicts:
        listLines.append(f"file {dictPlanet['sName']}.in")
        listLines.append("")
        if dictPlanet["dMass"] is not None and dictPlanet["dMassSigma"] is not None:
            listLines.append(
                f"dMass [{-abs(dictPlanet['dMass'])}, "
                f"{dictPlanet['dMassSigma']}, g, max0] "
                f"{dictPlanet['sName']}_mass"
            )
        listLines.append(
            f"dOrbPeriod [{-abs(dictPlanet['dOrbPeriod'])}, "
            f"{dictPlanet['dOrbPeriodSigma']}, g, max0] "
            f"{dictPlanet['sName']}_orbper"
        )
        listLines.append("")

    listLines.append("file star.in")
    listLines.append("")
    if sXUVModel == "Engle24MidLate" and dStellarMass < 0.101:
        pass
    elif sXUVModel == "Engle24MidLate":
        listLines.append(
            f"dMass [{dStellarMass}, {dStellarMassSigma}, g, min0.101] mass"
        )
    elif sXUVModel == "Engle24Early":
        listLines.append(
            f"dMass [{dStellarMass}, {dStellarMassSigma}, g, min0.40, max0.60] mass"
        )
    else:
        listLines.append(
            f"dMass [{dStellarMass}, {dStellarMassSigma}, g] mass"
        )
    listLines.append("dAge [5e6, 1e6, g, min1000000] start")

    if sXUVModel == "Engle24Early":
        listLines.append(
            "dXUVEngleEarlyA [-0.4896, 0.0773, g] engle_a"
        )
        listLines.append(
            "dXUVEngleEarlyB [-3.2128, 0.0458, g] engle_b"
        )
        listLines.append(
            "dXUVEngleEarlyC [-0.4469, 0.0835, g] engle_c"
        )
        listLines.append(
            "dXUVEngleEarlyD [-0.2985, 0.1005, g] engle_d"
        )
    else:
        listLines.append(
            "dXUVEngleMidLateA [-0.1456, 0.0911, g] engle_a"
        )
        listLines.append(
            "dXUVEngleMidLateB [-2.8876, 0.0439, g] engle_b"
        )
        listLines.append(
            "dXUVEngleMidLateC [-1.8187, 0.2412, g] engle_c"
        )
        listLines.append(
            "dXUVEngleMidLateD [0.3545, 0.0604, g] engle_d"
        )

    listLines.append("")
    listLines.append("file vpl.in")
    listLines.append("")
    listLines.append("dStopTime [age_samples.txt, txt, p, 1] age")
    listLines.append("")

    sFilePath = os.path.join(sOutputDirectory, "vspace.in")
    with open(sFilePath, "w") as fileHandle:
        fileHandle.write("\n".join(listLines))


def fnWriteVconvergeIn(sOutputDirectory, listPlanetNames):
    """Write the vconverge.in file."""
    sContent = f"""sVspaceFile vspace.in
iStepSize 100
iMaxSteps 200
sConvergenceMethod KS_statistic
fConvergenceCondition 0.004
iNumberOfConvergences 3

sObjectFile {listPlanetNames[0]}.in

saConverge final CumulativeXUVFlux
"""
    sFilePath = os.path.join(sOutputDirectory, "vconverge.in")
    with open(sFilePath, "w") as fileHandle:
        fileHandle.write(sContent)


def fnGenerateSystemFiles(dictSystem):
    """Generate all input files for a planetary system.

    Parameters
    ----------
    dictSystem : dict
        Dictionary with keys:
            sSystemName : str
            dStellarMass : float (Msun)
            dStellarMassSigma : float
            dRotationPeriod : float (days)
            dRotationPeriodSigma : float
            listPlanets : list of dict with keys:
                sName, dMass, dMassSigma, dOrbPeriod, dOrbPeriodSigma
    """
    sSystemDir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "systems",
        dictSystem["sSystemName"],
    )
    os.makedirs(sSystemDir, exist_ok=True)

    sSpectralClass = fsSelectSpectralClass(dictSystem["dStellarMass"])
    sXUVModel = fsSelectXUVModel(dictSystem["dStellarMass"])

    if "dAgeOverrideGyr" in dictSystem:
        dMeanAgeGyr = dictSystem["dAgeOverrideGyr"]
        dSigmaAgeGyr = dictSystem["dAgeOverrideSigmaGyr"]
        daAgeGyr = np.random.normal(dMeanAgeGyr, dSigmaAgeGyr, 100000)
        daAgeGyr = daAgeGyr[(daAgeGyr > 0.001) & (daAgeGyr <= 13.0)]
        daAgeYears = daAgeGyr * 1e9
        tAgeSummary = (
            np.mean(daAgeGyr),
            np.percentile(daAgeGyr, 2.5),
            np.percentile(daAgeGyr, 97.5),
        )
    else:
        daRotationPeriod = (
            dictSystem["dRotationPeriod"],
            dictSystem["dRotationPeriodSigma"],
        )
        daLogAge = fdaComputeLogAgeDistribution(
            daRotationPeriod, sSpectralClass
        )
        daAgeYears = fdaComputeAgeInYears(daLogAge)
        tAgeSummary = ftComputeAgeSummary(daLogAge)

    sAgePath = os.path.join(sSystemDir, "age_samples.txt")
    np.savetxt(sAgePath, daAgeYears)

    listPlanetNames = [d["sName"] for d in dictSystem["listPlanets"]]

    fnWriteVplIn(sSystemDir, dictSystem["sSystemName"], listPlanetNames)
    fnWriteStarIn(sSystemDir, dictSystem["dStellarMass"], sXUVModel)

    for dictPlanet in dictSystem["listPlanets"]:
        dPlanetMass = dictPlanet.get("dMass", 1.0)
        if dPlanetMass is None:
            dPlanetMass = 1.0
        fnWritePlanetIn(
            sSystemDir,
            dictPlanet["sName"],
            dPlanetMass,
            dictPlanet["dOrbPeriod"],
        )

    fnWriteVspaceIn(
        sSystemDir,
        dictSystem["dStellarMass"],
        dictSystem["dStellarMassSigma"],
        dictSystem["listPlanets"],
        sXUVModel,
    )
    fnWriteVconvergeIn(sSystemDir, listPlanetNames)

    return sSystemDir, tAgeSummary, sSpectralClass, sXUVModel
