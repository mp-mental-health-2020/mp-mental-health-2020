import glob
import os
import warnings
from pathlib import Path

"""
Utility functions for file handling.
"""

MATCH_ANY_INCLUDE_SUB_DIR = "**"


def get_current_directory():
    return os.path.dirname(os.path.abspath("__file__"))


def get_parent_directory():
    return get_parent_directory_for(get_current_directory())


def get_parent_directory_for(directory):
    return os.path.abspath(os.path.join(directory, os.pardir))


def get_project_directory():
    return Path(__file__).parent.parent


def get_file_names_in_directory_for_pattern(directory, pattern, print_file_names=False):
    try:
        requested_file_pattern = os.path.join(directory, MATCH_ANY_INCLUDE_SUB_DIR, pattern)
    except TypeError:
        raise Exception("NoneType Error for directory {dir} or pattern {pattern}"
                        .format(dir=directory, pattern=pattern))

    filtered_file_names = glob.glob(requested_file_pattern, recursive=True)
    if len(filtered_file_names) == 0:
        warnings.warn("No files matched the given pattern: {}".format(requested_file_pattern), stacklevel=2)

    if print_file_names:
        for file_name in filtered_file_names:
            short_file_name = file_name.split(os.path.sep)[-1]
            print(short_file_name)
    return filtered_file_names
