import pandas as pd
import os
import tqdm
import h5py
import glob
import re
import json
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import scipy
import logging
from collections import namedtuple

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

PP_TEST_DATAFILE = "pp_test_posteriors.h5"


def get_idx_from_filename(filename):
    """Eg filename: bbh_1_0_result.hdf5"""
    # get first number in filename
    return int(re.findall(r"\d+", filename)[0])



def get_credible_intervals(posterior, injection: dict) -> dict:
    credible_intervals = {}
    # intersection of params in injections and posterior
    pp_params = set(injection.keys()).intersection(set(posterior.columns))
    for param in pp_params:
        inj_val = injection[param]
        post1d = posterior[param]
        if len(set(post1d)) == 1:
            continue
        credible_intervals[param] = sum(np.array(post1d < inj_val)) / len(post1d)
    return credible_intervals


def cache_credible_intervals_for_all_events(pp_test_datafile=PP_TEST_DATAFILE):
    with h5py.File(pp_test_datafile, "r") as h5file:
        injection_df = pd.DataFrame(h5file['injection_parameters'][:])
        posteriors = h5file['posteriors']
        credible_intervals = {}
        for event in tqdm.tqdm(posteriors, desc='Computing credible intervals'):
            event_num = int(event.split("_")[1])
            injection = injection_df.loc[event_num]
            event_posterior = pd.DataFrame(posteriors[event][:])
            credible_intervals[event] = get_credible_intervals(event_posterior, injection)
    credible_intervals = pd.DataFrame(credible_intervals).T
    with h5py.File(pp_test_datafile, "a") as h5file:
        h5file.create_dataset("credible_intervals", data=credible_intervals.to_records(index=False))
    return credible_intervals


def combine_posteriors_in_h5(posterior_regex, injection_file, h5filename=PP_TEST_DATAFILE):
    """Combine posteriors in h5 files in a directory into a single h5 file
    h5file
    |--- posteriors
    |--- posteriors/injection_0 (dataframe)
    |--- posteriors/injection_1 (dataframe)
    ...
    |--- injection_parameters (dataframe)
    """
    with h5py.File(h5filename, "w") as h5file:
        with open(injection_file, 'r') as file:
            injections = json.load(file)
        injection_df = pd.DataFrame(injections['injections']['content'])
        print(f"Injection df: {injection_df}")
        h5file.create_dataset("injection_parameters", data=injection_df.to_records(index=False))
        posteriors = h5file.create_group("posteriors")
        posterior_files = glob.glob(posterior_regex)
        for fn in tqdm.tqdm(posterior_files, desc="Saving posteriors to h5 file"):
            posterior_df = pd.read_csv(fn, delimiter=" ")
            idx_num = get_idx_from_filename(os.path.basename(fn))
            # create dataset using posterior_df including column names
            posteriors.create_dataset(
                f"injection_{idx_num}",
                data=posterior_df.to_records(index=False),
            )
        # save the h5 file
        h5file.close()

        # credible_intervals = get_credible_intervals_for_all_events(h5filename)


def plot_credible_intervals(credible_levels:pd.DataFrame, confidence_interval=[0.68, 0.95, 0.997]):
    """Plot credible intervals for a set of parameters"""
    fig, ax = plt.subplots()
    colors = ["C{}".format(i) for i in range(8)]
    linestyles = ["-", "--", ":"]
    lines = ["{}{}".format(a, b) for a, b in product(linestyles, colors)]
    x_values = np.linspace(0, 1, 1001)
    N = len(credible_levels)
    confidence_interval_alpha = [0.1] * len(confidence_interval)
    for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
        edge_of_bound = (1. - ci) / 2.
        lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
        upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
        # The binomial point percent function doesn't always return 0 @ 0,
        # so set those bounds explicitly to be sure
        lower[0] = 0
        upper[0] = 0
        ax.fill_between(x_values, lower, upper, alpha=alpha, color='k')
    pvalues = []
    logger.info("Key: KS-test p-value")
    for ii, key in enumerate(credible_levels):
        pp = np.array([sum(credible_levels[key].values < xx) /
                       len(credible_levels) for xx in x_values])
        pvalue = scipy.stats.kstest(credible_levels[key], 'uniform').pvalue
        pvalues.append(pvalue)
        logger.info("{}: {}".format(key, pvalue))
        label = "{} ({:2.3f})".format(key, pvalue)
        plt.plot(x_values, pp, lines[ii], label=label)
    Pvals = namedtuple('pvals', ['combined_pvalue', 'pvalues', 'names'])
    pvals = Pvals(combined_pvalue=scipy.stats.combine_pvalues(pvalues)[1],
                  pvalues=pvalues,
                  names=list(credible_levels.keys()))
    logger.info(
        "Combined p-value: {}".format(pvals.combined_pvalue))
    ax.set_title("N={}, p-value={:2.4f}".format(len(credible_levels), pvals.combined_pvalue))
    ax.set_xlabel("C.I.")
    ax.set_ylabel("Fraction of events in C.I.")
    # ax legend to the right of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # keep the x and y axis the same size (square)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig("pp.png", dpi=500)
    return fig, pvals



combine_posteriors_in_h5("out_bbh*/res*/*.dat", "datafiles/bbh_injections.json")
cred_int = cache_credible_intervals_for_all_events(PP_TEST_DATAFILE)
plot_credible_intervals(cred_int)
