"""
Plot the cosmic shoreline: escape velocity vs. normalized cumulative
XUV flux for all planets in the catalog, with Solar System reference.

Produces a log-log plot following the style of Zahnle & Catling (2017)
and Barnes et al. (GJ 1132).
"""

import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

D_MARKER_SIZE = 7
D_FONT_SIZE = 18
D_TICK_FONT = 14
D_ANNOTATION_FONT = 8

DICT_SOLAR_SYSTEM = {
    "Mercury": {"dEscVel": 4.25, "dNormFlux": 6.67},
    "Venus": {"dEscVel": 10.36, "dNormFlux": 1.91},
    "Earth": {"dEscVel": 11.19, "dNormFlux": 1.00},
    "Mars": {"dEscVel": 5.03, "dNormFlux": 0.43},
    "Jupiter": {"dEscVel": 59.5, "dNormFlux": 0.074},
    "Saturn": {"dEscVel": 35.5, "dNormFlux": 0.011},
    "Uranus": {"dEscVel": 21.3, "dNormFlux": 0.0027},
    "Neptune": {"dEscVel": 23.5, "dNormFlux": 0.0011},
}


def fdaGenerateColormap(iNumSystems):
    """Return an array of distinct colors for the given number of systems."""
    if iNumSystems <= 20:
        daColormap = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        daColormap = plt.cm.gist_ncar(np.linspace(0.05, 0.95, iNumSystems))
    return daColormap


def fnPlotShorelineLine(ax):
    """Draw the cosmic shoreline line."""
    daShorelineX = [0.2, 60]
    daShorelineY = [1e-6, 1e4]
    ax.plot(
        daShorelineX,
        daShorelineY,
        color="lightsteelblue",
        linewidth=5,
        zorder=-1,
        alpha=0.7,
    )
    ax.annotate(
        "Cosmic",
        (1.3, 0.0008),
        fontsize=D_FONT_SIZE - 2,
        rotation=42,
        color="lightsteelblue",
    )
    ax.annotate(
        "Shoreline",
        (18, 60),
        fontsize=D_FONT_SIZE - 2,
        rotation=42,
        color="lightsteelblue",
    )


def fnPlotSolarSystem(ax):
    """Plot Solar System planets as black circles."""
    for sName, dictVals in DICT_SOLAR_SYSTEM.items():
        ax.plot(
            dictVals["dEscVel"],
            dictVals["dNormFlux"],
            "o",
            color="k",
            markersize=D_MARKER_SIZE - 1,
            zorder=5,
        )


def fnPlotSolarSystemLabels(ax):
    """Annotate Solar System planet positions."""
    dictOffsets = {
        "Mercury": (1.5, 9),
        "Venus": (10.8, 2.6),
        "Earth": (12.5, 0.55),
        "Mars": (5.3, 0.18),
        "Jupiter": (32, 0.045),
        "Saturn": (37, 0.006),
        "Uranus": (14, 0.0016),
        "Neptune": (25, 0.0006),
    }
    for sName, tPos in dictOffsets.items():
        ax.annotate(sName, tPos, fontsize=D_ANNOTATION_FONT, color="0.4")


def fnPlotErrorBar(ax, dX, dY, dLower, dUpper, sColor, sLabel=None,
                   bMsini=False):
    """Plot a data point with vertical error bar."""
    daYerr = np.array([[dY - dLower], [dUpper - dY]])
    sMarkerStyle = "o" if not bMsini else "o"
    sFillColor = sColor if not bMsini else "none"
    ax.plot(
        dX, dY, sMarkerStyle, color=sColor, markerfacecolor=sFillColor,
        markersize=D_MARKER_SIZE, zorder=10, label=sLabel,
        markeredgewidth=1.5,
    )
    ax.errorbar(
        [dX], [dY], yerr=daYerr, capsize=3, capthick=1.5,
        elinewidth=1.5, fmt="none", ecolor=sColor, zorder=9,
    )


def fnCreateCosmicShorelineFigure(dictAllResults, sOutputPath):
    """Create the cosmic shoreline figure from results dictionary."""
    fig, ax = plt.subplots(figsize=(10, 8))

    fnPlotShorelineLine(ax)
    fnPlotSolarSystem(ax)
    fnPlotSolarSystemLabels(ax)

    iNumSystems = len(dictAllResults)
    daColors = fdaGenerateColormap(iNumSystems)

    iColorIndex = 0
    for sSystemName, dictSystem in dictAllResults.items():
        sColor = daColors[iColorIndex % len(daColors)]
        bFirst = True
        for dictPlanet in dictSystem["listPlanetResults"]:
            sLabel = dictSystem["sStarName"] if bFirst else None
            bMsini = dictPlanet.get("bMsini", False)
            fnPlotErrorBar(
                ax,
                dictPlanet["dEscapeVelocity"],
                dictPlanet["dMeanNormalizedFlux"],
                dictPlanet["dLower95NormalizedFlux"],
                dictPlanet["dUpper95NormalizedFlux"],
                sColor,
                sLabel,
                bMsini,
            )
            bFirst = False
        iColorIndex += 1

    ax.set_xlabel("Escape Velocity [km/s]", fontsize=D_FONT_SIZE)
    ax.set_ylabel("Normalized Cumulative XUV Flux", fontsize=D_FONT_SIZE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 100)
    ax.set_ylim(1e-4, 1e5)
    ax.tick_params(labelsize=D_TICK_FONT)
    ax.legend(
        fontsize=D_ANNOTATION_FONT - 2, loc="lower right",
        ncol=4, framealpha=0.7,
    )

    fig.savefig(sOutputPath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Cosmic shoreline figure saved to {sOutputPath}")


if __name__ == "__main__":
    sResultsPath = os.path.join(REPO_ROOT, "results.json")
    if not os.path.exists(sResultsPath):
        print("No results.json found. Run buildCatalog.py first.")
        sys.exit(1)

    with open(sResultsPath, "r") as fh:
        dictAllResults = json.load(fh)

    sOutputPath = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(REPO_ROOT, "manuscript", "cosmic_shoreline.pdf")
    )
    fnCreateCosmicShorelineFigure(dictAllResults, sOutputPath)
