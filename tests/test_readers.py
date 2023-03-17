from src.reader import GWTC1Result, GWTC3Result, BilbyResult
import os

ROOT = os.path.dirname(__file__)
GWTC1_GW150914 = os.path.join("../gwtc1_samples/GW150914_GWTC-1.hdf5")
GWTC3_GW150914 = os.path.join("../gwtc3_samples/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5")
REVIEW_GW150914 = os.path.join("../review_results/GW150914_0_result.hdf5")


def test_corner_from_gwtc1():
    r = GWTC1Result.from_hdf5(GWTC1_GW150914).CBCResult
    fname = "./test.png"
    r.plot_corner(parameters=["mass_1", "mass_2", "luminosity_distance", "a_1", "a_2", "tilt_1", "tilt_2"], color="C0",
                  save=True, titles=True, filename=fname)


def test_corner_from_gwtc3():
    r = GWTC3Result.from_hdf5(GWTC3_GW150914)
    cbc = r.CBCResult
    fname = "./test.png"
    cbc.plot_corner(parameters=["mass_1", "mass_2", "luminosity_distance", "a_1", "a_2", "tilt_1", "tilt_2"], color="C0",
                  save=True, titles=True, filename=fname)
    r.write_psds(".")


def test_corner_from_review():
    r = BilbyResult.from_hdf5(REVIEW_GW150914).CBCResult
    fname = "./test.png"
    r.plot_corner(parameters=["mass_1", "mass_2", "luminosity_distance", "a_1", "a_2", "tilt_1", "tilt_2"], color="C0",
                  save=True, titles=True, filename=fname)