import bilby
import h5py
import corner
import re
import os
import glob
from typing import List, Dict
from bilby.gw.result import CBCResult
from .reader import GWTC1Result, BilbyResult
from .utils import get_event_name
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

print(f"bilby version: {bilby.__version__}")

ROOT = os.path.dirname(__file__)


def get_review_results() -> Dict[str, CBCResult]:
    files = glob.glob(f"{ROOT}/../review_results/*.hdf5")
    results = {}
    for f in files:
        name = get_event_name(f)
        print(f"Loading Review:{name}")
        results[name] = BilbyResult.from_hdf5(f).CBCResult
    return results


def get_lvk_results() -> Dict[str, CBCResult]:
    files = glob.glob(f"{ROOT}/../gwtc1_samples/*.hdf5")
    results = {}
    for f in files:
        name = get_event_name(f)
        print(f"Loading LVK:{name}")
        results[name] = GWTC1Result.from_hdf5(f).CBCResult
    return results


def plot_review_results(plot_dir: str = "plots"):
    os.makedirs(plot_dir, exist_ok=True)

    lvk_results = get_lvk_results()
    review_results = get_review_results()

    parameter_sets = dict(
        mass=["mass_1", "mass_2", "mass_ratio", "chirp_mass"],
        skypos=["luminosity_distance", "ra", "dec"],
        spin=["a_1", "a_2", "cos_tilt_1", "cos_tilt_2", "chi_eff"],
        effective_spin=["chi_eff", "mass_ratio"],
    )
    for name, result in review_results.items():
        for param_set in parameter_sets:
            param = parameter_sets[param_set]
            fname = f"{plot_dir}/{name}_{param_set}.png"
            plot_overlaid_corner(result, lvk_results[name], param, fname)


def _check_list_is_subset(l1, l2):
    s1 = set(l1)
    s2 = set(l2)
    if not s1.issubset(s2):
        raise ValueError(f"Missing parameters: {s1.difference(s2)}")


def plot_overlaid_corner(r1: CBCResult, r2: CBCResult, parameters: List[str], fname: str):
    # ensure parameters are in both results posteriors
    _check_list_is_subset(parameters, list(r1.posterior.columns))
    _check_list_is_subset(parameters, list(r2.posterior.columns))

    # plot corner
    fig = r1.plot_corner(parameters=parameters, color="C0", save=False, titles=True)
    fig = r2.plot_corner(parameters=parameters, color="C1", fig=fig, save=False, quantiles=[], titles=False)

    # add legend to figure to right using the following colors
    labels, colors = ["Review", "GWTC"], ["C0", "C1"]
    legend_elements = [Line2D([0], [0], color=c, lw=4, label=l) for c, l in zip(colors, labels)]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.95, 0.95), bbox_transform=fig.transFigure, frameon=False, fontsize=16)

    fig.savefig(fname)
