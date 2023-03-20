import re
from argparse import Namespace


def parse_ini(cli_args):
    """Returns the biby/pbily args """
    return Namespace()

def get_event_name(s) -> str:
    """Regex extraction of the name"""
    return re.findall(r"(GW\d{6})", s)[0]
