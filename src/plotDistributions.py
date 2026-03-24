"""
Plot cumulative XUV flux and age distributions for all systems.

Produces multi-panel histogram figures showing the probability
distributions for each system in the catalog.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catalog import LIST_SYSTEMS
from cosmicShoreline import D_CUMULATIVE_EARTH_FLUX
from runPipeline import fdictParseConvergedOutput

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

D_FONT_SIZE = 14
D_TICK_FONT = 10


def fnPlotAgeDistributions(sOutputPath):
    """Plot age distribution histograms for all systems."""
    iNumSystems = len(LIST_SYSTEMS)
    iCols = 6
    iRows = (iNumSystems + iCols - 1) // iCols

    fig, daAxes = plt.subplots(
        iRows, iCols, figsize=(20, 2.5 * iRows), squeeze=False,
    )

    for iIndex, dictSystem in enumerate(LIST_SYSTEMS):
        iRow = iIndex // iCols
        iCol = iIndex % iCols
        ax = daAxes[iRow, iCol]

        sAgePath = os.path.join(
            REPO_ROOT, "systems", dictSystem["sSystemName"],
            "age_samples.txt",
        )
        if not os.path.exists(sAgePath):
            ax.set_visible(False)
            continue

        daAgeYears = np.loadtxt(sAgePath)
        daAgeGyr = daAgeYears / 1e9

        ax.hist(daAgeGyr, bins=50, density=True, color="steelblue",
                edgecolor="white", linewidth=0.3, alpha=0.8)

        dMean = np.mean(daAgeGyr)
        dLower = np.percentile(daAgeGyr, 2.5)
        dUpper = np.percentile(daAgeGyr, 97.5)

        ax.axvline(dMean, color="k", linestyle="-", linewidth=1.5)
        ax.axvline(dLower, color="k", linestyle="--", linewidth=1)
        ax.axvline(dUpper, color="k", linestyle="--", linewidth=1)

        ax.set_title(
            f"{dictSystem['sStarName']}",
            fontsize=D_FONT_SIZE,
        )
        ax.set_xlabel("Age [Gyr]", fontsize=D_TICK_FONT)
        ax.set_ylabel("Density", fontsize=D_TICK_FONT)
        ax.tick_params(labelsize=D_TICK_FONT - 2)

    for iIndex in range(iNumSystems, iRows * iCols):
        iRow = iIndex // iCols
        iCol = iIndex % iCols
        daAxes[iRow, iCol].set_visible(False)

    fig.tight_layout()
    fig.savefig(sOutputPath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Age distribution figure saved to {sOutputPath}")


def fnPlotFluxDistributions(sOutputPath):
    """Plot cumulative XUV flux distribution histograms."""
    listPanels = []
    for dictSystem in LIST_SYSTEMS:
        sSystemDir = os.path.join(
            REPO_ROOT, "systems", dictSystem["sSystemName"],
        )
        for dictPlanet in dictSystem["listPlanets"]:
            listPanels.append({
                "sStarName": dictSystem["sStarName"],
                "sPlanetName": dictPlanet["sName"],
                "sSystemDir": sSystemDir,
            })

    iNumPanels = len(listPanels)
    iCols = 6
    iRows = (iNumPanels + iCols - 1) // iCols

    fig, daAxes = plt.subplots(
        iRows, iCols, figsize=(20, 2.5 * iRows), squeeze=False,
    )

    for iIndex, dictPanel in enumerate(listPanels):
        iRow = iIndex // iCols
        iCol = iIndex % iCols
        ax = daAxes[iRow, iCol]

        try:
            dictFlux = fdictParseConvergedOutput(
                dictPanel["sSystemDir"], dictPanel["sPlanetName"],
            )
            daNorm = dictFlux["daNormalized"]

            ax.hist(
                np.log10(daNorm), bins=50, density=True,
                color="coral", edgecolor="white",
                linewidth=0.3, alpha=0.8,
            )
            ax.axvline(
                np.log10(dictFlux["dMean"]),
                color="k", linestyle="-", linewidth=1.5,
            )
            ax.axvline(
                np.log10(dictFlux["dLower95"]),
                color="k", linestyle="--", linewidth=1,
            )
            ax.axvline(
                np.log10(dictFlux["dUpper95"]),
                color="k", linestyle="--", linewidth=1,
            )
        except Exception as error:
            ax.text(
                0.5, 0.5, "No data",
                transform=ax.transAxes, ha="center",
            )

        sTitle = (
            f"{dictPanel['sStarName']} {dictPanel['sPlanetName']}"
        )
        ax.set_title(sTitle, fontsize=D_FONT_SIZE - 2)
        ax.set_xlabel(
            r"$\log_{10}(F_{\rm XUV}/F_{\oplus})$",
            fontsize=D_TICK_FONT,
        )
        ax.set_ylabel("Density", fontsize=D_TICK_FONT)
        ax.tick_params(labelsize=D_TICK_FONT - 2)

    for iIndex in range(iNumPanels, iRows * iCols):
        iRow = iIndex // iCols
        iCol = iIndex % iCols
        daAxes[iRow, iCol].set_visible(False)

    fig.tight_layout()
    fig.savefig(sOutputPath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Flux distribution figure saved to {sOutputPath}")


if __name__ == "__main__":
    sManuscriptDir = os.path.join(REPO_ROOT, "manuscript")
    os.makedirs(sManuscriptDir, exist_ok=True)

    fnPlotAgeDistributions(
        os.path.join(sManuscriptDir, "age_distributions.pdf")
    )
    fnPlotFluxDistributions(
        os.path.join(sManuscriptDir, "flux_distributions.pdf")
    )
