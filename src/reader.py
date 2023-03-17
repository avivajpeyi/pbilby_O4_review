"""Module to load a GWTC1 result from a hdf5 file"""
import h5py
import pandas as pd
from bilby.gw.result import CBCResult
import bilby
from bilby.gw.conversion import component_masses_to_chirp_mass, chirp_mass_and_mass_ratio_to_component_masses
import numpy as np
from .utils import get_event_name
import os
from abc import abstractmethod

from typing import List, Dict, Optional


class Res:
    def __init__(self, posterior: pd.DataFrame, event_name: str, psds: Optional[Dict[str, np.ndarray]] = None):
        self.event_name = event_name
        self.posterior = posterior
        self.psds = psds

    @classmethod
    @abstractmethod
    def from_hdf5(cls, fname: str):
        return cls(posterior=pd.DataFrame(), event_name="")

    @property
    def CBCResult(self) -> CBCResult:
        priors = bilby.gw.prior.BBHPriorDict()
        result = CBCResult(
            label=self.event_name,
            outdir=f'out_{self.event_name}',
            sampler="dynesty",
            search_parameter_keys=list(priors.keys()),
            fixed_parameter_keys=list(),
            priors=priors,
            sampler_kwargs=dict(test="test", func=lambda x: x),
            meta_data=dict(
                likelihood=dict(
                    phase_marginalization=False,
                    distance_marginalization=False,
                    time_marginalization=False,
                    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                    waveform_arguments=dict(
                        reference_frequency=20.0, waveform_approximant="NRSur7dq4"
                    ),
                    interferometers=dict(
                        H1=dict(optimal_SNR=1),
                        L1=dict(optimal_SNR=1),
                    ),
                    sampling_frequency=4096,
                    duration=4,
                    start_time=0,
                    waveform_generator_class=bilby.gw.waveform_generator.WaveformGenerator,
                    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                )
            ),
            posterior=self.posterior,
        )
        return result


class GWTC1Result(Res):

    @property
    def posterior(self) -> pd.DataFrame:
        return self._posterior

    @posterior.setter
    def posterior(self, posterior: pd.DataFrame):
        """Ensure that the parameters are the ones we expect"
        GWTC params:
            luminosity_distance_Mpc
            m1_detector_frame_Msun
            m2_detector_frame_Msun
            right_ascension
            declination
            costheta_jn
            spin1
            costilt1
            spin2
            costilt2
        """
        posterior['luminosity_distance'] = posterior['luminosity_distance_Mpc']
        posterior['mass_1'] = posterior['m1_detector_frame_Msun']
        posterior['mass_2'] = posterior['m2_detector_frame_Msun']
        posterior['ra'] = posterior['right_ascension']
        posterior['dec'] = posterior['declination']
        posterior['theta_jn'] = np.arccos(posterior['costheta_jn'])
        posterior['a_1'] = posterior['spin1']
        posterior['cos_tilt_1'] = posterior['costilt1']
        posterior['tilt_1'] = np.arccos(posterior['costilt1'])
        posterior['a_2'] = posterior['spin2']
        posterior['cos_tilt_2'] = posterior['costilt2']
        posterior['tilt_2'] = np.arccos(posterior['costilt2'])

        # conversions
        m1, m2 = posterior['mass_1'], posterior['mass_2']
        q = m2 / m1
        c1, c2 = posterior['cos_tilt_1'], posterior['cos_tilt_2']
        a1, a2 = posterior['a_1'], posterior['a_2']
        posterior['mass_ratio'] = q
        posterior['chirp_mass'] = component_masses_to_chirp_mass(m1, m2)
        posterior['total_mass'] = m1 + m2
        posterior['chi_eff'] = (c1 * a1 + c2 * a2 * q) / (q + 1)

        self._posterior = posterior

    @classmethod
    def from_hdf5(cls, path: str) -> 'GWTC1Result':
        event_name = get_event_name(path)
        with h5py.File(path, 'r') as f:
            if 'Overall_posterior' in f:
                posterior = pd.DataFrame(f['Overall_posterior'][:])
            elif 'IMRPhenomPv2NRT_lowSpin_posterior' in f:
                posterior = pd.DataFrame(f['IMRPhenomPv2NRT_lowSpin_posterior'][:])
        return cls(posterior, event_name)

class GWTC3Result(Res):

    @classmethod
    def from_hdf5(cls, path: str) -> 'GWTC3Result':
        event_name = get_event_name(path)
        with h5py.File(path, 'r') as f:
            if 'C01:IMRPhenomXPHM' in f:
                d = f['C01:IMRPhenomXPHM']
                posterior = pd.DataFrame(d['posterior_samples'][:])
                psds = dict(H1=d['psds/H1'][:], L1=d['psds/L1'][:])
            else:
                raise ValueError(f"Could not find posterior in {path}")
        return cls(posterior, event_name, psds)

    def write_psds(self, outdir:str='.'):
        for det, psd in self.psds.items():
            fname = f"{self.event_name}_gwtc3_{det}_psd.txt"
            fpath = os.path.join(outdir, fname)
            np.savetxt(fpath, psd, delimiter=" ")



class BilbyResult(Res):

    @classmethod
    def from_hdf5(self, fname)->'BilbyResult':
        event_name = get_event_name(fname)
        try:
            r = CBCResult.from_hdf5(fname)
            return BilbyResult(r.posterior, event_name)
        except Exception as e:
            # this is a ridiculous hack to get around the fact that bilby
            # sometimes fails reading its own file...
            print(f"Error loading {fname}: {e}")

        with h5py.File(fname, "r") as f:
            data = bilby.result.recursively_load_dict_contents_from_group(f, '/')
            posterior = pd.DataFrame(data['posterior'])

        return BilbyResult(posterior, event_name)


    @property
    def posterior(self) -> pd.DataFrame:
        return self._posterior

    @posterior.setter
    def posterior(self, posterior: pd.DataFrame):
        posterior['cos_tilt_1'] = np.cos(posterior['tilt_1'])
        posterior['cos_tilt_2'] = np.cos(posterior['tilt_2'])
        if 'mass_1' not in posterior:
            m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(
                posterior['chirp_mass'], posterior['mass_ratio']
            )
            posterior['mass_1'] = m1
            posterior['mass_2'] = m2

        # conversions
        q = posterior['mass_ratio']
        c1, c2 = posterior['cos_tilt_1'], posterior['cos_tilt_2']
        a1, a2 = posterior['a_1'], posterior['a_2']
        posterior['chi_eff'] = (c1 * a1 + c2 * a2 * q) / (q + 1)

        self._posterior = posterior










