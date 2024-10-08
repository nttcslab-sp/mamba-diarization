# Copyright (c) 2024 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Alexis Plaquet, 2024

from pathlib import Path


def find_correct_relative_parent_path(relativepath: Path | str, parent: Path | str) -> Path | None:
    """Returns the first possible path that exists.
    To do so it tries the whole hierarchy of `parent` as a root for `relativepath`.

    For example, `relativepath='x.zip'` and `parent='/a/b/c'` will try to find
    - `'./x.zip'`
    - `'/a/b/c/x.zip'`
    - `'/a/b/x.zip'`
    - `'/a/x.zip'`
    - `'/x.zip'`
    and return the first one that exists.

    Very useful when you extract relative paths from a file not in your cwd
    (then relativepath=the extracted path and parent=the file parent path).

    Parameters
    ----------
    relativepath : Path | str
        Input relative path
    parent : Path | str
        Parent hierarchy to try

    Returns
    -------
    Path | None
        A path if a valid one is found, None otherwise
    """

    relativepath = Path(relativepath)
    parent = Path(parent)

    if relativepath.exists():
        return relativepath

    for p in parent.parents:
        if (p / relativepath).exists():
            return p / relativepath
    # print(f'Couldnt find a matching parent dir for : {relativepath} with parent {parent}')
    return None
