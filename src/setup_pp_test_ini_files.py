"""Generate pp test files"""
import numpy as np
import sys
from .utils import parse_ini

np.random.seed(0)

N_INJ = 100


def create_injection_file(prior_fn, n=N_INJ):
    """Generate a json file with injection parameters"""
    pass


def create_ini_files_from_template(template_ini, prior_fn, n=N_INJ):
    """Generate N ini files from a template ini file"""
    pass


if __name__ == '__main__':
    ini_fname = sys.argv[1:]
    args = parse_ini(ini_fname)
    create_injection_file(args.prior_fn, N_INJ)
    create_ini_files_from_template(ini_fname, args.prior_fn, N_INJ)
