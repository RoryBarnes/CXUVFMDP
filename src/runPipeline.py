"""
Run the vconverge pipeline for all systems in the catalog.

For each system directory under systems/, run:
    1. vconverge vconverge.in
    2. Parse the Converged_Param_Dictionary.json output
    3. Compute normalized cumulative XUV flux statistics
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cosmicShoreline import (
    D_CUMULATIVE_EARTH_FLUX,
    fdEscapeVelocityEarthUnits,
    fdNormalizedCumulativeFlux,
    fdShorelineDistance,
    fdShorelineFlux,
)

VPLANET_BIN_DIR = "/home/vplanet/.local/bin"


def fnBuildEnvironment():
    """Return environment dict with vplanet on PATH."""
    dictEnv = os.environ.copy()
    dictEnv["PATH"] = VPLANET_BIN_DIR + ":" + dictEnv.get("PATH", "")
    return dictEnv


def fnRunVconverge(sSystemDir):
    """Run vconverge in the given system directory."""
    sVconvergeExe = os.path.join(VPLANET_BIN_DIR, "vconverge")
    if not os.path.isfile(sVconvergeExe):
        import shutil
        sVconvergeExe = shutil.which("vconverge")
    if sVconvergeExe is None:
        raise FileNotFoundError("Cannot find vconverge executable")

    dictEnv = fnBuildEnvironment()
    print(f"  Running vconverge in {sSystemDir}...", flush=True)
    result = subprocess.run(
        [sVconvergeExe, "vconverge.in"],
        cwd=sSystemDir,
        env=dictEnv,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}", flush=True)
        raise RuntimeError(
            f"vconverge failed in {sSystemDir} with code {result.returncode}"
        )
    print(f"  Completed vconverge in {sSystemDir}", flush=True)


def fdaExtractFluxFromTrials(sSystemDir, sPlanetName, sSystemName):
    """Extract final CumulativeXUVFlux from trial forward files.

    Scans all trial output directories and reads the last line of
    each planet's .forward file.

    Returns
    -------
    numpy array of cumulative XUV flux values (W/m^2).
    """
    sOutputDir = os.path.join(sSystemDir, "output")
    listFlux = []
    for sTrialDir in sorted(os.listdir(sOutputDir)):
        sTrialPath = os.path.join(sOutputDir, sTrialDir)
        if not os.path.isdir(sTrialPath):
            continue
        sForwardFile = os.path.join(
            sTrialPath, f"{sSystemName}.{sPlanetName}.forward"
        )
        if not os.path.exists(sForwardFile):
            continue
        try:
            with open(sForwardFile, "r") as fileHandle:
                sLastLine = ""
                for sLine in fileHandle:
                    sLastLine = sLine
                dFlux = float(sLastLine.strip().split()[-1])
                if dFlux > 0:
                    listFlux.append(dFlux)
        except (ValueError, IndexError):
            continue
    return np.array(listFlux)


def fdictParseConvergedOutput(sSystemDir, sPlanetName):
    """Parse the Converged_Param_Dictionary.json output.

    If the planet is not in the converged dictionary (e.g. for
    multi-planet systems where only one planet was tracked),
    falls back to extracting flux from individual trial files.

    Returns
    -------
    dict with keys: daCumulativeFlux, dMean, dLower95, dUpper95,
                    dMeanNormalized, dLower95Normalized, dUpper95Normalized
    """
    sJsonPath = os.path.join(
        sSystemDir, "output", "Converged_Param_Dictionary.json"
    )
    if not os.path.exists(sJsonPath):
        raise FileNotFoundError(f"No converged output at {sJsonPath}")

    with open(sJsonPath, "r") as fileHandle:
        sContent = fileHandle.read().strip()
        if sContent.startswith('"') and sContent.endswith('"'):
            sContent = sContent[1:-1]
            sContent = sContent.replace('\\"', '"')
        dictData = json.loads(sContent)

    sKey = f"{sPlanetName},CumulativeXUVFlux,final"
    daCumulativeFlux = np.array(dictData.get(sKey, []))

    if len(daCumulativeFlux) == 0:
        sSystemName = os.path.basename(sSystemDir)
        daCumulativeFlux = fdaExtractFluxFromTrials(
            sSystemDir, sPlanetName, sSystemName
        )

    daCumulativeFlux = pd.Series(daCumulativeFlux).dropna().values
    daCumulativeFlux = daCumulativeFlux[daCumulativeFlux > 0]

    if len(daCumulativeFlux) == 0:
        raise ValueError(
            f"No valid CumulativeXUVFlux data for {sPlanetName}"
        )

    daNormalized = daCumulativeFlux / D_CUMULATIVE_EARTH_FLUX

    dMean = np.mean(daNormalized)
    dLower = np.percentile(daNormalized, 2.5)
    dUpper = np.percentile(daNormalized, 97.5)
    dMedian = np.median(daNormalized)

    return {
        "daCumulativeFlux": daCumulativeFlux,
        "daNormalized": daNormalized,
        "dMean": dMean,
        "dMedian": dMedian,
        "dLower95": dLower,
        "dUpper95": dUpper,
    }


def fnRunAllSystems(sSystemsDir):
    """Run vconverge for all system directories."""
    listSystemDirs = sorted(
        [
            os.path.join(sSystemsDir, sName)
            for sName in os.listdir(sSystemsDir)
            if os.path.isdir(os.path.join(sSystemsDir, sName))
        ]
    )

    dictResults = {}
    for sSystemDir in listSystemDirs:
        sSystemName = os.path.basename(sSystemDir)
        sVconvergeIn = os.path.join(sSystemDir, "vconverge.in")
        if not os.path.exists(sVconvergeIn):
            print(f"  Skipping {sSystemName}: no vconverge.in")
            continue

        print(f"\n{'=' * 50}")
        print(f"Processing {sSystemName}")
        print(f"{'=' * 50}")

        try:
            fnRunVconverge(sSystemDir)
            dictResults[sSystemName] = "completed"
        except Exception as error:
            print(f"  FAILED: {error}")
            dictResults[sSystemName] = f"failed: {error}"

    return dictResults


if __name__ == "__main__":
    sSystemsDir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "systems",
    )
    fnRunAllSystems(sSystemsDir)
