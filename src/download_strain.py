"""downloads strain data for the interferometers specified in the ini file."""
import sys
from .utils import parse_ini


def download_strain(detector, start_time, end_time, duration, sampling_frequency):
    pass


if __name__ == '__main__':
    ini_fname = sys.argv[1:]
    args = parse_ini(ini_fname)
    download_strain(args.detector, args.start_time, args.end_time, args.duration, args.sampling_frequency)
