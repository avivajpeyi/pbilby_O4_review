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





x= [14.94,
17.16,
12.69,
12.28,
8.27,
6.80,
10.30,
9.64,
14.01,
12.08,
14.95,
14.96,
5.96,
6.78,
7.22,
7.33,
13.50,
3.18,
14.19,
16.94,
10.03,
10.75,
8.60,
10.77,
2.21,
7.69,
75.79,
137.32,
10.46,
21.87,
58.32,
49.92,
3.42,
3.19,
21.10,
13.81,
5.48,
7.10,
7.33,
5.69,
19.77,
20.05,
4.58,
3.94,
17.35,
16.50,
5.86,
5.12,
14.52,
12.37,
5.86,
7.47,
1.97,
1.71,
1.54,
2.96,
6.29,
5.27,
10.66,
0.86,
27.45,
14.06,
5.46,
10.61,
6.99,
8.22,
16.60,
15.83,
13.39,
11.98,
4.97,
3.02,
24.82,
24.14,
2.28,
0.90,
15.44,
19.43,
6.34,
7.21,
2.24,
3.42,
4.02,
1.99,
4.45,
5.73,
15.72,
14.57,
16.13,
16.34,
8.87,
3.11,
7.68,
5.88,
68.55,
65.08,
4.71,
0.50,
2.86,
2.44,
6.15,
2.44,
12.86,
6.44,
8.52,
6.34,
14.85,
9.45,
2.96,
3.24,
14.35,
10.72,
4.82,
5.24,
4.96,
5.59,
6.01,
5.87,
10.82,
13.30,
16.66,
18.54,
10.63,
14.00,
4.51,
5.62,
8.25,
8.46,
5.41,
3.28,
8.45,
9.20,
9.45,
9.00,
6.82,
7.19,
0.87,
2.04,
3.31,
4.32,
1.17,
5.22,
27.55,
22.23,
3.63,
3.49,
4.58,
2.90,
4.00,
5.43,
8.54,
12.85,
4.39,
5.54,
18.61,
23.45,
2.19,
1.25,
17.47,
7.43,
5.46,
7.68,
3.04,
2.55,
18.81,
13.81,
5.13,
9.40,
9.83,
10.01,
4.22,
2.87,
6.53,
11.22,
5.96,
7.16,
4.49,
2.13,
13.13,
10.26,
7.72,
11.20,
4.17,
5.06,
12.80,
11.60,
8.25,
10.45,
3.48,
4.45,
5.06,
5.57,
5.35,
4.76,
12.19,
15.54,
3.62,
4.47,
1.11,
0.56,]

import matplotlib.pyplot as plt
import numpy as np

plt.hist(x, bins=np.geomspace(0.5, 150, 20))
plt.xscale('log')
plt.xlabel('SNR')
plt.ylabel('Number of events')
plt.savefig('snr.png', dpi=500)