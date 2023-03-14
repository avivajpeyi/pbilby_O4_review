from src.plotter import plot_overlaid_corner
import bilby
import os
import pandas as pd
import numpy as np

np.random.seed(0)


def generate_fake_res():
    pri = bilby.core.prior.PriorDict(
        dict(
            f1=bilby.core.prior.Uniform(name="f1", minimum=0, maximum=1, latex_label="f_1"),
            f2=bilby.core.prior.Uniform(name="f2", minimum=0, maximum=1, latex_label="f_2"),
        ))
    true = pri.sample(1)
    post = bilby.core.prior.PriorDict(
        dict(
            f1=bilby.core.prior.TruncatedGaussian(name="f1", minimum=0, maximum=1, mu=true["f1"], sigma=0.3),
            f2=bilby.core.prior.TruncatedGaussian(name="f2", minimum=0, maximum=1, mu=true["f2"], sigma=0.3),
        )).sample(10000)
    res = bilby.core.result.Result(
        label="test",
        outdir=".",
        sampler="test",
        priors=pri,
        posterior=pd.DataFrame(post),
        search_parameter_keys=list(pri.keys()),
    )
    return res


def test_plot_overlaid_corner():
    r1, r2 = generate_fake_res(), generate_fake_res()
    fname = f"./test.png"
    plot_overlaid_corner(r1, r2, ["f1", "f2"], fname)
    assert os.path.exists(fname)
    print(fname)

