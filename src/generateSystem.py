"""
Generate VPLanet/vconverge input files for a planetary system to compute
the cumulative XUV flux distribution.

Each system gets a directory under systems/ containing:
    - vpl.in      : primary VPLanet input file
    - star.in     : host star parameters (stellar + flare modules)
    - <planet>.in : one per planet with P < 25 days
    - vspace.in   : parameter sweep definition
    - vconverge.in: convergence control file
    - age_samples.txt: age distribution from Engle & Guinan (2023)
"""

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


def fsFormatVplContent(sSystemName, sBodyFiles):
    """Return the content string for vpl.in."""
    return (
        f"# Primary input file for {sSystemName}\n"
        f"sSystemName\t{sSystemName}\n"
        "iVerbose\t0\nbOverwrite\t1\n\n"
        f"saBodyFiles\t{sBodyFiles}\n\n"
        "sUnitMass\tsolar\nsUnitLength\taU\n"
        "sUnitTime\tYEAR\nsUnitAngle\td\nsUnitTemp\tK\n\n"
        "bDoLog\t\t1\niDigits\t\t6\ndMinValue\t1e-10\n\n"
        "bDoForward\t1\nbVarDt\t\t1\n"
        "dEta\t\t1.0e-2\ndStopTime\t8.0e9\ndOutputTime\t1.0e6\n"
    )


def fnWriteVplIn(sOutputDirectory, sSystemName, listPlanetNames):
    """Write the primary vpl.in file."""
    sBodyFiles = "star.in\t" + "\t".join(
        [f"{sName}.in" for sName in listPlanetNames]
    )
    sContent = fsFormatVplContent(sSystemName, sBodyFiles)
    sFilePath = os.path.join(sOutputDirectory, "vpl.in")
    with open(sFilePath, "w") as fileHandle:
        fileHandle.write(sContent)


def fsFormatFlareBlock():
    """Return the flare parameter block for star.in."""
    return (
        "sFlareFFD\t\tsingle\n"
        "dFlareMinEnergy\t\t-1.0e29\n"
        "dFlareMaxEnergy\t\t-1.0e34\n"
        "dEnergyBin\t\t20\n"
        "sFlareBandPass\t\tsxr\n"
        "dFlareSingleStarA1\t-0.301\n"
        "dFlareSingleStarA2\t-0.533\n"
        "dFlareSingleStarB1\t8.732\n"
        "dFlareSingleStarB2\t16.780"
    )


def fsFormatStarContent(dMass, sXUVModel, sFlareBlock):
    """Return the content string for star.in."""
    return (
        "# Host star parameters\n"
        "sName\t\tstar\nsaModules\tstellar\tflare\n\n"
        f"dMass\t\t{dMass}\ndRotPeriod\t-1.0\ndAge\t\t1.0e6\n\n"
        f"sStellarModel\tbaraffe\nsXUVModel\t{sXUVModel}\n"
        "bHaltEndBaraffeGrid\t0\n\n"
        f"{sFlareBlock}\n\n"
        "saOutputOrder\tTime -Luminosity -LXUVStellar -LXUVFlare -LXUVTot\n"
    )


def fnWriteStarIn(sOutputDirectory, dMass, sXUVModel):
    """Write the star.in file with stellar and flare parameters."""
    dClampedMass, _ = ftClampMassForModel(dMass, sXUVModel)
    sContent = fsFormatStarContent(dClampedMass, sXUVModel, fsFormatFlareBlock())
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


def flistFormatVspaceHeader():
    """Return constant header lines for vspace.in."""
    return [
        "srcfolder .",
        "destfolder output",
        "trialname xuv_",
        "samplemode random",
        "randsize 500",
        "",
    ]


def flistFormatPlanetVspaceBlock(dictPlanet):
    """Return vspace lines for a single planet's parameters."""
    listLines = [f"file {dictPlanet['sName']}.in", ""]
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
    return listLines


def ftClampMassForModel(dStellarMass, sXUVModel):
    """Clamp stellar mass to valid range for the XUV model and return bounds string."""
    if sXUVModel == "Engle24Early":
        return min(max(dStellarMass, 0.40), 0.60), "min0.40, max0.60"
    return max(dStellarMass, 0.10), "min0.10, max0.399"


def flistFormatFlareVspaceBlock():
    """Return empty list — flare FFD parameters use fixed best-fit values.

    The FFD parameters (A1, A2, B1, B2) have large, correlated uncertainties
    from the GJ 1132 MCMC fit. Sampling them independently from Gaussians
    breaks the correlation structure, producing unphysical flare rates in
    ~40% of trials and preventing KS convergence. The best-fit values are
    kept in star.in (via fsFormatFlareBlock) so the FLARE module contributes
    to LXUVTot, but without adding non-convergent variance.
    """
    return []


def flistFormatEngleCoefficients(sXUVModel):
    """Return vspace lines for Engle XUV model coefficients."""
    if sXUVModel == "Engle24Early":
        return [
            "dXUVEngleEarlyA [-0.4896, 0.0773, g] engle_a",
            "dXUVEngleEarlyB [-3.2128, 0.0458, g] engle_b",
            "dXUVEngleEarlyC [-0.4469, 0.0835, g] engle_c",
            "dXUVEngleEarlyD [-0.2985, 0.1005, g] engle_d",
        ]
    return [
        "dXUVEngleMidLateA [-0.1456, 0.0911, g] engle_a",
        "dXUVEngleMidLateB [-2.8876, 0.0439, g] engle_b",
        "dXUVEngleMidLateC [-1.8187, 0.2412, g] engle_c",
        "dXUVEngleMidLateD [0.3545, 0.0604, g] engle_d",
    ]


def flistFormatStellarVspaceBlock(dStellarMass, dStellarMassSigma, sXUVModel):
    """Return vspace lines for stellar mass, age, Engle coefficients, and flare FFD."""
    listLines = ["file star.in", ""]
    dClampedMass, sMassBounds = ftClampMassForModel(dStellarMass, sXUVModel)
    listLines.append(f"dMass [{dClampedMass}, {dStellarMassSigma}, g, {sMassBounds}] mass")
    listLines.append("dAge [5e6, 1e6, g, min1000000] start")
    listLines.extend(flistFormatEngleCoefficients(sXUVModel))
    listLines.extend(flistFormatFlareVspaceBlock())
    listLines.append("")
    return listLines


def fnWriteVspaceIn(
    sOutputDirectory,
    dStellarMass,
    dStellarMassSigma,
    listPlanetDicts,
    sXUVModel,
):
    """Write the vspace.in file with Gaussian draws from posteriors."""
    listLines = flistFormatVspaceHeader()
    for dictPlanet in listPlanetDicts:
        listLines.extend(flistFormatPlanetVspaceBlock(dictPlanet))
    listLines.extend(
        flistFormatStellarVspaceBlock(dStellarMass, dStellarMassSigma, sXUVModel)
    )
    listLines.extend(["file vpl.in", ""])
    listLines.append("dStopTime [age_samples.txt, txt, p, 1] age")
    listLines.append("")

    sFilePath = os.path.join(sOutputDirectory, "vspace.in")
    with open(sFilePath, "w") as fileHandle:
        fileHandle.write("\n".join(listLines))


def fnWriteVconvergeIn(sOutputDirectory, listPlanetNames):
    """Write the vconverge.in file monitoring all planets."""
    listLines = [
        "sVspaceFile vspace.in",
        "iStepSize 100",
        "iMaxSteps 200",
        "sConvergenceMethod KS_statistic",
        "fConvergenceCondition 0.004",
        "iNumberOfConvergences 3",
        "",
    ]
    for sPlanetName in listPlanetNames:
        listLines.append(f"sObjectFile {sPlanetName}.in")
        listLines.append("saConverge final CumulativeXUVFlux")
        listLines.append("")

    sFilePath = os.path.join(sOutputDirectory, "vconverge.in")
    with open(sFilePath, "w") as fileHandle:
        fileHandle.write("\n".join(listLines))


def ftComputeOverrideAge(dictSystem):
    """Compute age distribution from literature override values."""
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
    return daAgeYears, tAgeSummary


def ftComputeSystemAge(dictSystem, sSpectralClass):
    """Compute age distribution, returning (daAgeYears, tAgeSummary)."""
    if "dAgeOverrideGyr" in dictSystem:
        return ftComputeOverrideAge(dictSystem)
    daRotationPeriod = (
        dictSystem["dRotationPeriod"],
        dictSystem["dRotationPeriodSigma"],
    )
    daLogAge = fdaComputeLogAgeDistribution(daRotationPeriod, sSpectralClass)
    daAgeYears = fdaComputeAgeInYears(daLogAge)
    tAgeSummary = ftComputeAgeSummary(daLogAge)
    return daAgeYears, tAgeSummary


def fsComputeSystemDirectory(sSystemName):
    """Return the path to a system's directory under systems/."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "systems",
        sSystemName,
    )


def fnWriteAllPlanetFiles(sSystemDir, listPlanets):
    """Write .in files for all planets in a system."""
    for dictPlanet in listPlanets:
        dPlanetMass = dictPlanet.get("dMass", 1.0)
        if dPlanetMass is None:
            dPlanetMass = 1.0
        fnWritePlanetIn(
            sSystemDir, dictPlanet["sName"],
            dPlanetMass, dictPlanet["dOrbPeriod"],
        )


def fnGenerateSystemFiles(dictSystem):
    """Generate all input files for a planetary system."""
    sSystemDir = fsComputeSystemDirectory(dictSystem["sSystemName"])
    os.makedirs(sSystemDir, exist_ok=True)

    sSpectralClass = fsSelectSpectralClass(dictSystem["dStellarMass"])
    sXUVModel = fsSelectXUVModel(dictSystem["dStellarMass"])
    daAgeYears, tAgeSummary = ftComputeSystemAge(dictSystem, sSpectralClass)
    np.savetxt(os.path.join(sSystemDir, "age_samples.txt"), daAgeYears)

    listPlanetNames = [d["sName"] for d in dictSystem["listPlanets"]]
    fnWriteVplIn(sSystemDir, dictSystem["sSystemName"], listPlanetNames)
    fnWriteStarIn(sSystemDir, dictSystem["dStellarMass"], sXUVModel)
    fnWriteAllPlanetFiles(sSystemDir, dictSystem["listPlanets"])
    fnWriteVspaceIn(
        sSystemDir, dictSystem["dStellarMass"],
        dictSystem["dStellarMassSigma"], dictSystem["listPlanets"], sXUVModel,
    )
    fnWriteVconvergeIn(sSystemDir, listPlanetNames)
    return sSystemDir, tAgeSummary, sSpectralClass, sXUVModel
