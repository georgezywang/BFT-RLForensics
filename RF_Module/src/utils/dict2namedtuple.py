"""
Code adapted from https://github.com/TonghanWang/ROMA
"""

from collections import namedtuple


def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)
