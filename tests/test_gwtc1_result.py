from src.reader import GWTC1Result
import os

ROOT = os.path.dirname(__file__)
LVK_GW150914 = os.path.join("../gwtc1_samples/GW150914_GWTC-1.hdf5")


def test_corner_from_gwtc1():
    r = GWTC1Result.from_hdf5(LVK_GW150914).CBCResult
    fname = "./test.png"
    r.plot_corner(parameters=["mass_1", "mass_2", "luminosity_distance", "a_1", "a_2", "tilt_1", "tilt_2"], color="C0",
                  save=True, titles=True, filename=fname)
