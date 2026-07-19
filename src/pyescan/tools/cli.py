import click
import os

class ClickKeyValueType(click.ParamType):
    name = "key=value"
    
    def convert(self, value, param, ctx):
        try:
            key, val = value.split('=', 1)
            return (key, val)
        except ValueError:
            self.fail(f'{value!r} is not a valid key=value pair', param, ctx)


class ClickDictParamType(click.ParamType):
    name = "dict"
    
    def convert(self, value, param, ctx):
        if isinstance(value, dict):
            return value
            
        config_dict = {}
        pairs = value.split(',') if ',' in value else [value]
        
        for pair in pairs:
            try:
                key, val = pair.split('=', 1)
                config_dict[key.strip()] = val.strip()
            except ValueError:
                self.fail(f'{pair!r} is not a valid key=value pair', param, ctx)
        
        return config_dict


@click.command()
@click.argument('export_location', type=click.Path(exists=True), required=True)
@click.argument('output_csv', type=click.Path(), default=None, required=False)
@click.option('--file_structure', type=str, default='pat/sdb', help='File structure gias list of names separated by / (default: pat/sdb).')
@click.option('--skip_image_level/--include-image-level', default=False, help='Skip image level information for faster parsing (default: False).')
def get_ce_export_summary_cli(export_location, output_csv, file_structure="pat/sdb", merged=True, skip_image_level=False):
    from .dataset_utils import get_ce_export_summary
    
    df = get_ce_export_summary(
        export_location,
        file_structure=file_structure,
        skip_image_level=skip_image_level
    )
    
    if output_csv:
        # If output_location is provided, save the DataFrame to the specified location
        df.to_csv(output_csv, index=False)
        click.echo(f"Result saved to {output_csv}")
    else:
        # If output_location is not provided, save to scans_folder with a default filename
        default_output = os.path.join(export_location, "metadata_summary.csv")
        df.to_csv(default_output, index=False)


@click.command()
@click.argument('target_location', type=click.Path(exists=True), required=True)
@click.argument('output_csv', type=click.Path(), required=True)
@click.option('--format', type=str, default='{pat}/{sdb}/{source_id}_{bscan_index:\d+}.png',
              help='File structure gias list of names separated by / (default: {pat}/{sdb}/{source_id}_{bscan_index:\d+}.png).')
def summarise_dataset_cli(target_location, output_csv, format):
    from .dataset_utils import summarise_dataset
    df = summarise_dataset(target_location, structure=format)
    df.to_csv(output_csv, index=False)
    click.echo(f"Result saved to {output_csv}")


@click.command()
@click.argument('function', type=str)
@click.argument('csv_file', type=click.Path(exists=True))
@click.argument('output_csv', type=click.Path(), default=None, required=False)
@click.option('--skip_prompt', '-y', is_flag=True, help='Skip confirmation.')
def run_function_on_csv_cli(csv_file, function, output_csv=None, skip_prompt=False):
    """
    Runs the specified function on the specified CSV and saves to the target location.
    If output_csv is not specified this will default to overwriting the input CSV.
    Function should map from Pandas DataFrame to Dataframe.
    """
    if not output_csv:
        confirmation = skip_prompt or click.confirm(f'No output location specified. Are you sure you want to overwrite {csv_file}?')
        if confirmation:
            output_csv = csv_file
        else:
            click.echo("Operation aborted.")
            return

    # Check if the specified function is available
    try:
        function_to_apply = globals()[function]
    except KeyError:
        raise ValueError(f"Function '{function}' not found in the global scope.")
        
    import pandas as pd
    
    # Load CSV into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Apply function
    result_df = function_to_apply(df)
    
    # Save the result to a CSV file
    result_df.to_csv(output_csv, index=False)
    click.echo(f"Result saved to {output_csv}")


@click.command()
@click.argument('function', type=str)
@click.argument('csv_file', type=click.Path(exists=True))
@click.argument('output_csv', type=click.Path(), default=None, required=False)
@click.option('--column_headings', '-c', type=click.STRING, help='Column headings separated by commas.')
@click.option('--threaded', is_flag=True, help='Use threading (recommended, requires pathos).')
@click.option('--skip_prompt', '-y', is_flag=True, help='Skip confirmation.')
def run_function_over_csv_cli(csv_file, function, output_csv=None, column_headings=None, threaded=None, skip_prompt=True):
    """
    Runs the specified function on each row of the specified CSV and appends the result as additional columns, with the names given in column_headings.
    If output_csv is not specified this will default to overwriting the input CSV.
    Function should accept a row parameter, where columns can be accessed by row.column_name, and output a single value / tuple of values.
    """
    if not output_csv:
        confirmation = skip_prompt or click.confirm(f'No output location specified. Are you sure you want to overwrite {csv_file}?')
        if confirmation:
            output_csv = csv_file
        else:
            click.echo("Operation aborted.")
            return

    # Check if the specified function is available
    function_split = function.rsplit(".",1)
    if len(function_split) == 1:
        function_name = function_split
        # Try get the function from globals()
        if function in globals():
            function_to_apply = globals()[function]
        else:
            click.echo(f"ERROR: Function '{function}' not found in the global scope.")
            return
    else:
        module_name, function_name = function_split
        # Try import function from module
        try:
            import importlib
            imported_module = importlib.import_module(module_name)
            function_to_apply = getattr(imported_module, function_name)
        except Exception as e:
            click.echo(f"ERROR: Failed to import {function_name} from {module_name}:")
            raise e
        
    # Deafult to using the function name
    if not column_headings:
        column_headings = function
        
    # Get headings
    column_headings = column_headings.split(",")
    if len(column_headings) == 1: column_headings = column_headings[0]
        
    import pandas as pd
    from .dataset_utils import run_function_on_dataframe
    
    # Load CSV into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Run function over dataframe
    result_df = run_function_on_dataframe(df, function_to_apply, column_headings, threaded)
    
    # Save the result to a CSV file
    result_df.to_csv(output_csv, index=False)
    click.echo(f"Result saved to {output_csv}")


@click.command()
@click.argument('stats', type=str)
@click.argument('csv_file', type=click.Path(exists=True))
@click.argument('output_csv', type=click.Path(), default=None, required=False)
@click.option('--mapping', type=ClickDictParamType(),
              help='Mapping from CSV column headings to internal ')
@click.option('--suffix', '-s', type=click.STRING, help='Suffix to append to added column names.')
@click.option('--intermediates', is_flag=True, help='Save intermediate metrics calculated in process.')
@click.option('--threaded', is_flag=True, help='Use threading (recommended, requires pathos).')
@click.option('--skip_prompt', '-y', is_flag=True, help='Skip confirmation.')
def run_metric_csv_cli(csv_file, stats, output_csv=None, mapping={}, suffix=None, intermediates=False, threaded=False, skip_prompt=False):
    """
    Runs the specified function on each row of the specified CSV and appends the result as additional columns.
    If output_csv is not specified this will default to overwriting the input CSV.
    Function should accept a row parameter, where columns can be accessed by row.column_name, and output a single value / tuple of values.
    """
    if not output_csv:
        confirmation = skip_prompt or click.confirm(f'No output location specified. Are you sure you want to overwrite {csv_file}?')
        if confirmation:
            output_csv = csv_file
        else:
            click.echo("Operation aborted.")
            return

    import pandas as pd
    from ..metrics.helpers import run_on_dataframe
    
    # Load CSV into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Run function over dataframe
    stats = [s.strip() for s in stats.split(',')]
    result_df = run_on_dataframe(df, stats, mapping, suffix=suffix, named_only=not intermediates, auto_merge=True, threaded=threaded)
    
    # Save the result to a CSV file
    result_df.to_csv(output_csv, index=False)
    click.echo(f"Result saved to {output_csv}")


@click.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.argument('output_csv', type=click.Path())
@click.argument('pivot_col', type=str)
@click.argument('identifier_cols', type=str)
@click.option('--flatten_column_names', is_flag=True, default=True, help='Flatten names of column headings.')
@click.option('--skip_prompt', '-y', is_flag=True, help='Skip confirmation.')
def narrow_to_wide_cli(csv_file, output_csv, pivot_col, identifier_cols, flatten_column_names=True, skip_prompt=False):
    """
    Pivots the specified CSV around the column heading given by pivot_col and ouputs new CSV to output_csv.
    Automatically detects which columns to pivot based on identifier_cols.
    identifier_cols should be supplied as a column separated list of a minimum number of required columns to identify each element to be pivoted.
    """
    
    import pandas as pd
    from .dataset_utils import detect_pivot_cols
    
    # Load CSV into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Get pivot cols
    identifier_cols = identifier_cols.split(",")
    index_cols, value_cols = detect_pivot_cols(df, pivot_col, identifier_cols)
    
    
    if not (skip_prompt or click.confirm(f'The following columns will be replicated across {pivot_col} values:\n{value_cols}\n\n Continue?')):
        click.echo("Operation aborted.")
        return

    # Pivot
    result_df = df.pivot(index=index_cols, columns=pivot_col, values=value_cols).reset_index()
    
    if flatten_column_names:
        # Rename columns, needs assigning to intermediate variable or weird stuff happens
        columns = [ "_".join([str(c) for c in col if c]) for col in result_df.columns]
        result_df.columns = columns
    
    # Save the result to a CSV file
    result_df.to_csv(output_csv)
    click.echo(f"Result saved to {output_csv}")

