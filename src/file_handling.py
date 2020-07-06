import glob
import json
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


def get_sub_directories(experiments_folder_path):
    return [os.path.join(experiments_folder_path, o) for o in os.listdir(experiments_folder_path) if os.path.isdir(os.path.join(experiments_folder_path, o))]


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


def read_json_file(file_name):
    try:
        with open(file_name, "r") as file_:
            data_text = file_.read()
        try:
            json_string = json.loads(data_text)
            return json_string
        except json.JSONDecodeError as e:
            raise Exception("Decoding json failed for {file_name}: {error}".format(file_name=file_name, error=e))
    except FileNotFoundError as e:
        raise Exception("File {file_name} not found: {error}".format(file_name=file_name, error=e))
