import numpy as np
import matplotlib.pyplot as plt


def load_psd(fn):
    print(f"Loading {fn}")
    try:
        return np.loadtxt(fn, delimiter="\t")
    except ValueError:
        return np.loadtxt(fn, delimiter=" ")


def plot_psds(psds, colors, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    for (label, psd), color in zip(psds.items(), colors):
        ax.loglog(psd[:, 0], psd[:, 1], label=label, color=color, alpha=0.5, **kwargs)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [Hz$^{-1}$]")
    ax.set_xlim(left=10, right=2048)
    ax.set_ylim(top=1e-37)
    ax.legend()
    return ax


def plot_psds_from_files(fns, colors, ax=None, **kwargs):
    psds = {fn: load_psd(fn) for fn in fns}
    return plot_psds(psds, colors, ax=ax, **kwargs)


if __name__ == '__main__':
    fns = [
        "GW150914_gwtc2.1_H1_psd.txt",
        "review_GW150914_h1_psd.dat",
    ]
    ax = plot_psds_from_files(fns, ["C0", "C1"])
    ax.set_title("GW150914 H1")
    plt.savefig("GW150914-H1-psds.png", dpi=300)

    fns = [
        "GW150914_gwtc2.1_L1_psd.txt",
        "review_GW150914_l1_psd.dat",
    ]
    ax = plot_psds_from_files(fns, ["C0", "C1"])
    ax.set_title("GW150914 L1")
    plt.savefig("GW150914-L1-psds.png", dpi=300)
