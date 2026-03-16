"""
Orchestrate the CXUVFC pipeline: generate system files, run
vconverge simulations, collect results, and produce the manuscript
figures and tables.

Usage:
    python director.py                 # Run full pipeline
    python director.py --generate      # Generate input files only
    python director.py --simulate      # Run vconverge simulations only
    python director.py --analyze       # Collect results and make plots
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from buildCatalog import (
    fdictCollectResults,
    fnBuildAllSystems,
    fnRunAllSimulations,
    fnSaveResults,
)
from generateTables import fnWriteAgeTable, fnWriteFluxTable
from plotCosmicShoreline import fnCreateCosmicShorelineFigure
from plotDistributions import (
    fnPlotAgeDistributions,
    fnPlotFluxDistributions,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def fnGenerate():
    """Step 1: Generate all system input files."""
    print("\nStep 1: Generating system input files...")
    print("=" * 60)
    fnBuildAllSystems()


def fnSimulate():
    """Step 2: Run vconverge simulations."""
    print("\nStep 2: Running vconverge simulations...")
    print("=" * 60)
    fnRunAllSimulations()


def fnAnalyze():
    """Step 3: Collect results and produce outputs."""
    print("\nStep 3: Collecting results...")
    print("=" * 60)
    dictAllResults = fdictCollectResults()

    sManuscriptDir = os.path.join(REPO_ROOT, "manuscript")
    os.makedirs(sManuscriptDir, exist_ok=True)

    sResultsPath = os.path.join(REPO_ROOT, "results.json")
    fnSaveResults(dictAllResults, sResultsPath)

    print("\nGenerating tables...")
    fnWriteAgeTable(
        os.path.join(sManuscriptDir, "table_ages.tex")
    )
    fnWriteFluxTable(
        dictAllResults,
        os.path.join(sManuscriptDir, "table_fluxes.tex"),
    )

    print("\nGenerating figures...")
    fnPlotAgeDistributions(
        os.path.join(sManuscriptDir, "age_distributions.pdf")
    )
    fnPlotFluxDistributions(
        os.path.join(sManuscriptDir, "flux_distributions.pdf")
    )
    fnCreateCosmicShorelineFigure(
        dictAllResults,
        os.path.join(sManuscriptDir, "cosmic_shoreline.pdf"),
    )

    fnPrintResultsSummary(dictAllResults)


def fnPrintResultsSummary(dictAllResults):
    """Print a summary of results to the terminal."""
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    for sSystem, dictSystem in dictAllResults.items():
        print(f"\n  {dictSystem['sStarName']}:")
        for dictPlanet in dictSystem["listPlanetResults"]:
            dFlux = dictPlanet["dMeanNormalizedFlux"]
            dShoreDist = dictPlanet["dMeanShorelineDistance"]
            sSide = "above" if dShoreDist > 1 else "below"
            print(
                f"    {dictPlanet['sPlanetName']}: "
                f"F/F_Earth = {dFlux:.1f}, "
                f"Shore dist = {dShoreDist:.2f} ({sSide})"
            )


def main():
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="CXUVFC pipeline orchestrator"
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate input files only",
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run vconverge simulations only",
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Collect results and make plots only",
    )
    args = parser.parse_args()

    bRunAll = not (args.generate or args.simulate or args.analyze)

    print("CXUVFC: Cumulative XUV Flux Catalog")
    print("=" * 60)

    if bRunAll or args.generate:
        fnGenerate()
    if bRunAll or args.simulate:
        fnSimulate()
    if bRunAll or args.analyze:
        fnAnalyze()

    print("\nDone!")


if __name__ == "__main__":
    main()
