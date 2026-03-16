"""
Generate VPLanet input files for all systems in the catalog,
then run vconverge to compute cumulative XUV flux distributions.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catalog import LIST_SYSTEMS
from cosmicShoreline import (
    D_CUMULATIVE_EARTH_FLUX,
    fdEscapeVelocityEarthUnits,
    fdShorelineDistance,
    fdShorelineFlux,
)
from generateSystem import fnGenerateSystemFiles
from runPipeline import fdictParseConvergedOutput, fnRunVconverge

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fnBuildAllSystems():
    """Generate input files for all systems and print age summaries."""
    listAgeSummaries = []
    for dictSystem in LIST_SYSTEMS:
        sSystemDir, tAgeSummary, sSpectralClass, sXUVModel = (
            fnGenerateSystemFiles(dictSystem)
        )
        dMeanAge, dLowerAge, dUpperAge = tAgeSummary
        print(
            f"  {dictSystem['sSystemName']:15s}  "
            f"Age = {dMeanAge:.2f} Gyr  "
            f"[{dLowerAge:.2f}, {dUpperAge:.2f}]  "
            f"class={sSpectralClass}  XUV={sXUVModel}"
        )
        listAgeSummaries.append(
            {
                "sSystemName": dictSystem["sSystemName"],
                "dMeanAge": dMeanAge,
                "dLowerAge": dLowerAge,
                "dUpperAge": dUpperAge,
                "sSpectralClass": sSpectralClass,
                "sXUVModel": sXUVModel,
            }
        )
    return listAgeSummaries


def fnRunAllSimulations():
    """Run vconverge for all systems, skipping already converged."""
    sSystemsDir = os.path.join(REPO_ROOT, "systems")
    for dictSystem in LIST_SYSTEMS:
        sSystemDir = os.path.join(sSystemsDir, dictSystem["sSystemName"])
        if not os.path.exists(os.path.join(sSystemDir, "vconverge.in")):
            print(f"  Skipping {dictSystem['sSystemName']}: no input files")
            continue
        sConvergedPath = os.path.join(
            sSystemDir, "output", "Converged_Param_Dictionary.json"
        )
        if os.path.exists(sConvergedPath):
            print(f"  Skipping {dictSystem['sSystemName']}: already converged")
            continue
        print(f"\n{'=' * 50}")
        print(f"Running {dictSystem['sSystemName']}")
        print(f"{'=' * 50}")
        try:
            fnRunVconverge(sSystemDir)
        except Exception as error:
            print(f"  FAILED: {error}")


def fdictCollectResults():
    """Collect results from all completed vconverge runs."""
    sSystemsDir = os.path.join(REPO_ROOT, "systems")
    dictAllResults = {}

    for dictSystem in LIST_SYSTEMS:
        sSystemDir = os.path.join(sSystemsDir, dictSystem["sSystemName"])
        sJsonPath = os.path.join(
            sSystemDir, "output", "Converged_Param_Dictionary.json"
        )
        if not os.path.exists(sJsonPath):
            continue

        dictSystemResults = {
            "sStarName": dictSystem["sStarName"],
            "dStellarMass": dictSystem["dStellarMass"],
            "dDistance": dictSystem["dDistance"],
            "listPlanetResults": [],
        }

        for dictPlanet in dictSystem["listPlanets"]:
            try:
                dictFlux = fdictParseConvergedOutput(
                    sSystemDir, dictPlanet["sName"]
                )
                dMassPlanet = dictPlanet.get("dMass", 1.0)
                if dMassPlanet is None:
                    dMassPlanet = 1.0
                dRadiusPlanet = dMassPlanet ** 0.27
                dEscVel = fdEscapeVelocityEarthUnits(
                    dMassPlanet, dRadiusPlanet
                )
                dShoreFlux = fdShorelineFlux(dEscVel)

                dMeanShorelineDist = fdShorelineDistance(
                    dictFlux["dMean"], dShoreFlux
                )
                dLowerShorelineDist = fdShorelineDistance(
                    dictFlux["dLower95"], dShoreFlux
                )
                dUpperShorelineDist = fdShorelineDistance(
                    dictFlux["dUpper95"], dShoreFlux
                )

                dictPlanetResult = {
                    "sPlanetName": dictPlanet["sName"],
                    "dOrbPeriod": dictPlanet["dOrbPeriod"],
                    "dMass": dMassPlanet,
                    "dEscapeVelocity": dEscVel,
                    "dShorelineFlux": dShoreFlux,
                    "dMeanNormalizedFlux": dictFlux["dMean"],
                    "dLower95NormalizedFlux": dictFlux["dLower95"],
                    "dUpper95NormalizedFlux": dictFlux["dUpper95"],
                    "dMeanShorelineDistance": dMeanShorelineDist,
                    "dLowerShorelineDistance": dLowerShorelineDist,
                    "dUpperShorelineDistance": dUpperShorelineDist,
                }
                dictSystemResults["listPlanetResults"].append(
                    dictPlanetResult
                )
            except Exception as error:
                print(
                    f"  Error for {dictSystem['sSystemName']} "
                    f"{dictPlanet['sName']}: {error}"
                )

        dictAllResults[dictSystem["sSystemName"]] = dictSystemResults

    return dictAllResults


def fnSaveResults(dictAllResults, sOutputPath):
    """Save results to a JSON file."""
    with open(sOutputPath, "w") as fileHandle:
        json.dump(dictAllResults, fileHandle, indent=2, default=str)
    print(f"Results saved to {sOutputPath}")


if __name__ == "__main__":
    print("Building catalog of M dwarf cumulative XUV fluxes")
    print("=" * 60)

    print("\nStep 1: Generating system input files...")
    listAgeSummaries = fnBuildAllSystems()

    print("\nStep 2: Running vconverge simulations...")
    fnRunAllSimulations()

    print("\nStep 3: Collecting results...")
    dictAllResults = fdictCollectResults()

    sResultsPath = os.path.join(REPO_ROOT, "results.json")
    fnSaveResults(dictAllResults, sResultsPath)

    print("\nDone!")
