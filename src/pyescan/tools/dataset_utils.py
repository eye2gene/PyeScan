import os


def _flatten_dict(dict_in, name=""):
    # Recursive function that "flattens" a nested dict by concatenating key names of sub-dicts
    # e.g. { "a": 1, "b": { "c": 2, "d": { "e": 3 } } } becomes { "a": 1, "b_c": 2, "b_d_e": 3}
    # Note: Does not handle lists.

    dict_out = dict()
    if type(dict_in) is dict:
        # Recursive case
        for k, v in dict_in.items():
            dict_out.update(_flatten_dict(v, name=f"{name}_{k}" if name else k))
    else:
        dict_out[name] = dict_in
    return dict_out


def _process_ce_metadata(metadata, identifier_dict, skip_image_level=False):
    # Takes metadata json format and processes into individual rows
    # identifier dict is mapping to uniquely identify scan (e.g. from file structure)

    UID_path = "/".join(identifier_dict.values())

    scan_records = list()

    scan_date = metadata["exam"]["scan_datetime"]
    series_info = _flatten_dict(metadata["series"])

    scans = metadata["images"]["images"]

    for i, scan in enumerate(scans):
        scan_data = dict()
        scan_data.update(identifier_dict)

        scan_data["scan_uid"] = UID_path + "/" + scan["source_id"]

        for attr in ["source_id", "group", "modality", "field_of_view"]:
            scan_data[attr] = scan[attr]
        scan_data["scan_number"] = i
        scan_data["date"] = scan_date
        scan_data.update(series_info)

        if scan.get("crystal_eye_laterality"):
            scan_data["group_laterality"] = scan["crystal_eye_laterality"]

        # workaround for name collision
        scan_data["series_source_id"] = metadata["series"]["source_id"]
        scan_data["source_id"] = scan["source_id"]

        scan_data["number_of_images"] = len(scan["contents"])

        scan_data["scan_width_px"] = scan["size"]["width"]
        scan_data["scan_height_px"] = scan["size"]["height"]
        for attr in ["dimensions_mm", "resolutions_mm"]:
            scan_data.update(_flatten_dict(scan[attr], attr))

        if skip_image_level:
            scan_records.append(scan_data)
        else:
            images = scan["contents"]
            for j, image in enumerate(images):
                image_data = dict()
                image_data.update(scan_data)

                # image_data['image_number'] = j
                image_data["bscan_index"] = str(j)

                image_data["image_capture_datetime"] = image.get("capture_datetime")
                image_data["image_quality_heidelberg"] = image.get("quality")

                if image.get("photo_locations"):
                    locations = image["photo_locations"][0]
                    image_data.update(_flatten_dict(locations, "bscan_location"))
                scan_records.append(image_data)
    return scan_records


# TODO: Break into functions for parsing all the metadata, then can join with images table
def get_ce_export_summary(
    export_location, file_structure="pat/sdb", merged=True, skip_image_level=False
):
    import json

    import pandas as pd
    import tqdm

    scan_records = list()
    file_records = list()

    pbar = tqdm.tqdm(os.walk(export_location))
    for dirpath, _dirnames, filenames in pbar:
        # Only check for leaf directories
        # if not len(dirnames) == 0: continue

        if "metadata.json" not in filenames:
            continue

        file_structure_keys = file_structure.split("/")
        file_structure_elems = dirpath.split("/")[-len(file_structure_keys) :]
        file_structure_dict = dict(
            zip(file_structure_keys, file_structure_elems, strict=False)
        )

        UID_path = "/".join(file_structure_elems)

        with open(os.path.join(dirpath, "metadata.json")) as f:
            metadata = json.load(f)
            scan_records.extend(
                _process_ce_metadata(metadata, file_structure_dict, skip_image_level)
            )

        for filename in filenames:
            # Assume filename is of the form [SOURCE-ID]_[bscan_index].png
            if ".png" not in filename:
                continue

            source_id, bscan_index = filename.rsplit("_", 1)
            bscan_index = bscan_index.rsplit(".", 1)[0]  # remove file_extension
            filepath = os.path.join(dirpath, filename)

            file_records.append(
                {
                    "file_path": filepath,
                    "file_name": filename,
                    "scan_uid": UID_path + "/" + source_id,
                    "bscan_index": bscan_index,
                }
            )

        pbar.set_postfix({"scans_found": len(scan_records)})

    df_metadata = pd.DataFrame(scan_records)
    df_files = pd.DataFrame(file_records)

    if merged:
        merge_cols = ["scan_uid"]
        if not skip_image_level:
            merge_cols = [*merge_cols, "bscan_index"]
        df_files = df_files.merge(df_metadata, how="left", on=merge_cols)
        return df_files
    else:
        return df_files, df_metadata


def structure_to_regex(structure_pattern):
    # TODO: Add automatic conversion of python formatting strings to regex
    # TODO: Addoptional way to specify wildcards for path
    import re

    # Escape characters not between {}
    escaped_structure = re.sub(
        r"(?<!{)([^{}]*)*(?![^{}]*})",  # Need to add the [^{}]* to the lookahead
        lambda match: re.escape(match.group(0)),
        structure_pattern,
    )

    """
    This code uses the re.sub() method to replace each instance of a curly brace
    expression with a corresponding regular expression pattern. The pattern uses
    a named capture group to capture the value of the component, and it includes
    a non-capturing group that matches either the specified regular expression
    pattern (if provided) or any sequence of non-slash characters (if no pattern
    is specified). (Generated by Chat-GPT)
    """

    # Replace items between brackets with groups
    regex_pattern = re.sub(
        r"{([a-zA-Z0-9_]*?)(?::(.*?))?}",
        # r"{(.*?)(?::(.*?))?}",
        lambda match: f"(?P<{match.group(1)}>{match.group(2) or '[^/]*?'})",
        escaped_structure,
    )

    regex_pattern = regex_pattern + "$"  # Fix for End-Of-Strin

    return regex_pattern


def summarise_dataset(
    dataset_root,
    structure=r"{pat}/{sdb}/{source_id}_{bscan_index:\d+}.png",
    regex=None,
    progress=True,
):
    import os
    import re

    import pandas as pd
    import tqdm

    dataset_root = os.path.abspath(dataset_root)

    regex_pattern = regex if regex else structure_to_regex(structure)

    records = list()
    pbar = tqdm.tqdm(os.walk(dataset_root)) if progress else os.walk(dataset_root)
    for root, _dirs, filenames in pbar:
        for filename in filenames:
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, dataset_root)

            match = re.match(regex_pattern, rel_path)
            if match:
                record = {"file_path": file_path, "file_path_relative": rel_path}
                record.update(match.groupdict())
                records.append(record)

        if progress:
            pbar.set_postfix({"scans_found": len(records)})

    return pd.DataFrame(records)


def get_median_bscans(df, scan_id_key=None, index_key="bscan_index"):
    if scan_id_key is None:
        scan_id_key = ["sdb", "source_id"]
    median_row_ids = list()
    for _scan_id, df_scan in df.groupby(scan_id_key):
        scan_indices = sorted(df_scan[index_key].values)
        middle_slice = scan_indices[len(scan_indices) // 2]
        rows = df_scan[df_scan[index_key] == middle_slice].index
        median_row_ids.extend(rows)

    df_middle = df.loc[median_row_ids]
    return df_middle


def run_function_on_dataframe(df, fn, column, threaded=True):
    import tqdm

    if threaded:
        from pathos.multiprocessing import Pool

        with Pool(64) as p:
            df[column] = list(
                tqdm.tqdm(p.imap(fn, list(df.itertuples(index=False))), total=len(df))
            )
    else:
        outputs = list()
        for row in tqdm.tqdm(df.itertuples(index=False), total=len(df)):
            outputs.append(fn(row))
        df[column] = outputs
    return df


# TODO: Deal with situations where there is partially missing values
#  in a given column (e.g. single value + na)
def detect_pivot_cols(df, pivot_col, identifier_cols):
    # Find columns with identical values for each unique combination in the target columns (from ChatGPT)
    grouped_df = df.groupby(identifier_cols)
    index_cols = list()
    for col in df.columns:
        if col in identifier_cols:
            continue  # skip
        # Check if column only has 1 or 0 values - drop_duplicates might be faster
        if not ((grouped_df[col].nunique(dropna=False) > 1).any()):
            index_cols.append(col)
    index_cols = identifier_cols + index_cols

    # Get remaining cols
    value_cols = list(set(df.columns) - set([*index_cols, pivot_col]))
    return index_cols, value_cols


def narrow_to_wide(
    df, pivot_col, identifier_cols, verbose=True, flatten_column_names=True
):
    # Pivots dataframe intelligently on a set of identifier columns,
    # saving additional metadata columns which are consistent over the pivot,
    # and pivoting only on those that change accoring to the identifier cols and pivot col

    # Get pivot cols
    index_cols, value_cols = detect_pivot_cols(df, pivot_col, identifier_cols)

    # Remove pivot_col if it gets picked up in index_cols
    if pivot_col in index_cols:
        index_cols.remove(pivot_col)

    if verbose:
        print("Pivoting on ", value_cols)

    df_summary = df.pivot(
        index=index_cols, columns=pivot_col, values=value_cols
    ).reset_index()

    if flatten_column_names:
        # Rename columns, needs assigning to intermediate variable or weird stuff happens
        columns = ["_".join([str(c) for c in col if c]) for col in df_summary.columns]
        df_summary.columns = columns

    return df_summary
