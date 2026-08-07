"""
Fast file discovery utilities using glob-based search.

These are internal helpers used by summarise_dataset.
Use summarise_dataset() as the public entry point for file finding.
"""
import glob
import os
import re
from typing import List, Optional

import pandas as pd


def structure_to_glob(structure_pattern: str) -> str:
    """
    Convert a structure pattern to a glob pattern for fast first-pass matching.

    Replaces named groups like {pat}, {sdb}, {source_id:\\d+} with '*'
    to create a broad glob that approximately matches the target structure.

    Parameters
    ----------
    structure_pattern : str
        Pattern like "{pat}/{sdb}/{source_id}_{bscan_index:\\d+}.png"

    Returns
    -------
    str
        Glob pattern like "*/*/*.png" (or more specific if literals are present)
    """
    # Replace {name} and {name:regex} with *
    glob_pattern = re.sub(r"\{[^}]*\}", "*", structure_pattern)
    return glob_pattern


def structure_to_regex(structure_pattern: str) -> str:
    """
    Convert a structure pattern to a regex for precise matching and extraction.

    Parameters
    ----------
    structure_pattern : str
        Pattern like "{pat}/{sdb}/{source_id}_{bscan_index:\\d+}.png"

    Returns
    -------
    str
        Compiled regex pattern with named groups.
    """
    # Escape characters not between {}
    escaped_structure = re.sub(
        r"(?<!{)([^{}]*)*(?![^{}]*})",
        lambda match: re.escape(match.group(0)),
        structure_pattern,
    )

    # Replace items between brackets with named groups
    regex_pattern = re.sub(
        r"{([a-zA-Z0-9_]*?)(?::(.*?))?}",
        lambda match: f"(?P<{match.group(1)}>{match.group(2) or '[^/]*?'})",
        escaped_structure,
    )

    return regex_pattern + '$'


def find_files(root: str,
               structure: str,
               progress: bool = True) -> pd.DataFrame:
    """
    Find files matching a structure pattern using glob + regex.

    Internal helper — use summarise_dataset() as the public interface.

    Parameters
    ----------
    root : str
        Root directory to search from.
    structure : str
        Pattern describing the file structure, e.g.
        "{pat}/{sdb}/{source_id}_{bscan_index:\\d+}.png"
    progress : bool, default True
        Whether to show a progress bar.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: file_path, file_path_relative, plus any
        named groups from the structure pattern.
    """
    root = os.path.abspath(root)
    glob_pattern = structure_to_glob(structure)
    regex_pattern = structure_to_regex(structure)
    regex_compiled = re.compile(regex_pattern)

    # Use glob for fast first-pass
    full_glob = os.path.join(root, glob_pattern)
    matched_paths = glob.glob(full_glob, recursive=False)

    if progress:
        try:
            import tqdm
            matched_paths = tqdm.tqdm(matched_paths, desc="Filtering matches")
        except ImportError:
            pass

    records = []
    for file_path in matched_paths:
        rel_path = os.path.relpath(file_path, root)
        match = regex_compiled.match(rel_path)
        if match:
            record = {"file_path": file_path, "file_path_relative": rel_path}
            record.update(match.groupdict())
            records.append(record)

    return pd.DataFrame(records)
