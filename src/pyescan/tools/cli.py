"""PyeScan CLI tools."""

import os
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="pyescan",
    help="PyeScan CLI tools for working with retinal scan data.",
    no_args_is_help=True,
)


def _parse_key_value_pairs(value: str) -> dict[str, str]:
    """Parse a comma-separated list of key=value pairs into a dict."""
    config_dict = {}
    pairs = value.split(",") if "," in value else [value]
    for pair in pairs:
        key, val = pair.split("=", 1)
        config_dict[key.strip()] = val.strip()
    return config_dict


@app.command()
def summarise_export(
    export_location: Annotated[
        Path, typer.Argument(help="Path to the Crystal Eye export folder.", exists=True)
    ],
    output_csv: Annotated[Path | None, typer.Argument(help="Output CSV path.")] = None,
    file_structure: Annotated[
        str, typer.Option(help="File structure as names separated by /.")
    ] = "pat/sdb",
    skip_image_level: Annotated[
        bool,
        typer.Option("--skip-image-level", help="Skip image-level info for speed."),
    ] = False,
) -> None:
    """Summarise a Crystal Eye / Private Eye export into a CSV."""
    from .dataset_utils import get_ce_export_summary

    df = get_ce_export_summary(
        str(export_location),
        file_structure=file_structure,
        skip_image_level=skip_image_level,
    )

    if output_csv:
        df.to_csv(output_csv, index=False)
        typer.secho(f"Result saved to {output_csv}", fg=typer.colors.GREEN)
    else:
        default_output = os.path.join(str(export_location), "metadata_summary.csv")
        df.to_csv(default_output, index=False)
        typer.secho(f"Result saved to {default_output}", fg=typer.colors.GREEN)


@app.command()
def summarise_dataset(
    target_location: Annotated[
        Path, typer.Argument(help="Root directory to scan.", exists=True)
    ],
    output_csv: Annotated[Path, typer.Argument(help="Output CSV path.")],
    structure: Annotated[
        str, typer.Option(help="File structure pattern.")
    ] = r"{pat}/{sdb}/{source_id}_{bscan_index:\d+}.png",
) -> None:
    """Summarise a dataset directory into a CSV based on a file structure pattern."""
    from .dataset_utils import summarise_dataset as _summarise_dataset

    df = _summarise_dataset(str(target_location), structure=structure)
    df.to_csv(output_csv, index=False)
    typer.secho(f"Result saved to {output_csv}", fg=typer.colors.GREEN)


@app.command()
def run_function_on_csv(
    function: Annotated[
        str, typer.Argument(help="Function name to apply (DataFrame -> DataFrame).")
    ],
    csv_file: Annotated[Path, typer.Argument(help="Input CSV file.", exists=True)],
    output_csv: Annotated[Path | None, typer.Argument(help="Output CSV path.")] = None,
    skip_prompt: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip confirmation.")
    ] = False,
) -> None:
    """Run a function on a CSV (DataFrame -> DataFrame) and save the result."""
    if not output_csv:
        if not skip_prompt:
            typer.confirm(
                f"No output location specified. Overwrite {csv_file}?", abort=True
            )
        output_csv = csv_file

    try:
        function_to_apply = globals()[function]
    except KeyError as err:
        raise typer.BadParameter(
            f"Function '{function}' not found in the global scope."
        ) from err

    import pandas as pd

    df = pd.read_csv(csv_file)
    result_df = function_to_apply(df)
    result_df.to_csv(output_csv, index=False)
    typer.secho(f"Result saved to {output_csv}", fg=typer.colors.GREEN)


@app.command()
def run_function_over_csv(
    function: Annotated[
        str, typer.Argument(help="Function name or module.function to apply per-row.")
    ],
    csv_file: Annotated[Path, typer.Argument(help="Input CSV file.", exists=True)],
    output_csv: Annotated[Path | None, typer.Argument(help="Output CSV path.")] = None,
    column_headings: Annotated[
        str | None,
        typer.Option(
            "-c", "--column-headings", help="Column headings (comma-separated)."
        ),
    ] = None,
    threaded: Annotated[
        bool, typer.Option("--threaded", help="Use threading (requires pathos).")
    ] = False,
    skip_prompt: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip confirmation.")
    ] = True,
) -> None:
    """Run a function on each row of a CSV and append results as new columns."""
    if not output_csv:
        if not skip_prompt:
            typer.confirm(
                f"No output location specified. Overwrite {csv_file}?", abort=True
            )
        output_csv = csv_file

    # Resolve the function
    function_split = function.rsplit(".", 1)
    if len(function_split) == 1:
        if function in globals():
            function_to_apply = globals()[function]
        else:
            typer.secho(
                f"ERROR: Function '{function}' not found in the global scope.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
    else:
        module_name, function_name = function_split
        try:
            import importlib

            imported_module = importlib.import_module(module_name)
            function_to_apply = getattr(imported_module, function_name)
        except Exception as e:
            typer.secho(
                f"ERROR: Failed to import {function_name} from {module_name}:",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1) from e

    # Default to using the function name as column heading
    headings = column_headings or function
    headings_list = headings.split(",")
    final_headings: str | list[str] = (
        headings_list[0] if len(headings_list) == 1 else headings_list
    )

    import pandas as pd

    from .dataset_utils import run_function_on_dataframe

    df = pd.read_csv(csv_file)
    result_df = run_function_on_dataframe(
        df, function_to_apply, final_headings, threaded
    )
    result_df.to_csv(output_csv, index=False)
    typer.secho(f"Result saved to {output_csv}", fg=typer.colors.GREEN)


@app.command()
def run_metric(
    stats: Annotated[
        str, typer.Argument(help="Comma-separated metric names to compute.")
    ],
    csv_file: Annotated[Path, typer.Argument(help="Input CSV file.", exists=True)],
    output_csv: Annotated[Path | None, typer.Argument(help="Output CSV path.")] = None,
    mapping: Annotated[
        str | None, typer.Option(help="Column mapping as key=value,key=value pairs.")
    ] = None,
    suffix: Annotated[
        str | None,
        typer.Option("-s", "--suffix", help="Suffix for added column names."),
    ] = None,
    intermediates: Annotated[
        bool, typer.Option("--intermediates", help="Save intermediate metrics.")
    ] = False,
    threaded: Annotated[
        bool, typer.Option("--threaded", help="Use threading.")
    ] = False,
    skip_prompt: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip confirmation.")
    ] = False,
) -> None:
    """Compute metrics on a CSV and append results as new columns."""
    if not output_csv:
        if not skip_prompt:
            typer.confirm(
                f"No output location specified. Overwrite {csv_file}?", abort=True
            )
        output_csv = csv_file

    parsed_mapping = _parse_key_value_pairs(mapping) if mapping else None

    import pandas as pd

    from pyescan.metrics.helpers import run_on_dataframe

    df = pd.read_csv(csv_file)
    stat_list = [s.strip() for s in stats.split(",")]
    result_df = run_on_dataframe(
        df,
        stat_list,
        parsed_mapping,
        suffix=suffix,
        named_only=not intermediates,
        auto_merge=True,
        threaded=threaded,
    )

    result_df.to_csv(output_csv, index=False)
    typer.secho(f"Result saved to {output_csv}", fg=typer.colors.GREEN)


@app.command()
def narrow_to_wide(
    csv_file: Annotated[Path, typer.Argument(help="Input CSV file.", exists=True)],
    output_csv: Annotated[Path, typer.Argument(help="Output CSV path.")],
    pivot_col: Annotated[str, typer.Argument(help="Column to pivot on.")],
    identifier_cols: Annotated[
        str, typer.Argument(help="Comma-separated identifier columns.")
    ],
    flatten_column_names: Annotated[
        bool, typer.Option("--flatten/--no-flatten", help="Flatten column headings.")
    ] = True,
    skip_prompt: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip confirmation.")
    ] = False,
) -> None:
    """Pivot a narrow CSV to wide format around a given column."""
    import pandas as pd

    from .dataset_utils import detect_pivot_cols

    df = pd.read_csv(csv_file)
    id_cols = identifier_cols.split(",")
    index_cols, value_cols = detect_pivot_cols(df, pivot_col, id_cols)

    if not skip_prompt:
        typer.confirm(
            f"The following columns will be replicated across {pivot_col} values:\n{value_cols}\n\n Continue?",
            abort=True,
        )

    result_df = df.pivot(
        index=index_cols, columns=pivot_col, values=value_cols
    ).reset_index()

    if flatten_column_names:
        columns = ["_".join([str(c) for c in col if c]) for col in result_df.columns]
        result_df.columns = columns

    result_df.to_csv(output_csv)
    typer.secho(f"Result saved to {output_csv}", fg=typer.colors.GREEN)
