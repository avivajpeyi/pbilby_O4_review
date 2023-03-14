import re


def get_event_name(s) -> str:
    """Regex extraction of the name"""
    return re.findall(r"(GW\d{6}\_\d{6}|GW\d{6})", s)[0]
