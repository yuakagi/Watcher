"""Module for the postgresql database."""

import os
import glob
from datetime import datetime, date
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from sqlalchemy.exc import SQLAlchemyError
import psycopg
from psycopg import ServerCursor
import pandas as pd
from ...general_params import watcher_config as config
from ...utils import load_db_engine, load_psycopg_params


def _generate_create_table_query(
    table_name: str, table_params: dict, schema: str = "public"
) -> str:
    """
    Generate a CREATE TABLE SQL query from a dictionary definition.
    """
    columns_definitions = []

    # Generate IDs
    if table_params.get("id_prefix"):
        id_prefix = table_params["id_prefix"]
        columns_definitions.append("dummy_id SERIAL NOT NULL")
        columns_definitions.append(
            f"{config.COL_RECORD_ID} TEXT GENERATED ALWAYS AS ('{id_prefix}' || dummy_id) STORED UNIQUE"
        )

    # Generate column definitions
    for col_name, col_props in table_params["columns"].items():
        col_definition = f"{col_name} {col_props['sql_ops']}"
        columns_definitions.append(col_definition)

    # Add primary key constraint
    if "primary_key" in table_params and table_params["primary_key"]:
        columns_definitions.append(f"PRIMARY KEY ({table_params['primary_key']})")

    # Add foreign key constraint if exists
    if "foreign_key" in table_params and table_params["foreign_key"]:
        fk_col, fk_table = table_params["foreign_key"]  # Tuple
        fk_def = f"FOREIGN KEY ({fk_col}) REFERENCES {fk_table}({fk_col})"
        columns_definitions.append(fk_def)

    # Format the final SQL query
    joind_defs = ",\n".join(columns_definitions)
    query = f"""
    CREATE TABLE IF NOT EXISTS {schema}.{table_name} ({joind_defs});
    """

    return query.strip()


def create_schema(schema: str = "public"):
    """
    Creates the given schema if it doesn't already exist.
    """
    connection_params = load_psycopg_params()
    with psycopg.connect(connection_params) as conn:
        with conn.cursor() as cur:
            # Check if schema exists
            cur.execute(
                """
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name = %s;
            """,
                (schema,),
            )
            exists = cur.fetchone()
            if not exists:
                cur.execute(f"CREATE SCHEMA {schema};")
                print(f"Schema '{schema}' created.")
            else:
                print(f"Schema '{schema}' already exists.")


def create_indexes(cur, table: str, index_fields: list, schema: str = "public"):
    for col in index_fields:
        index_name = f"idx_{table}_{col}"
        query = f"""
        CREATE INDEX IF NOT EXISTS {index_name}
        ON {schema}.{table} ({col});
        """
        cur.execute(query)
        print(f"Index '{index_name}' created.")


def table_exists(cur: ServerCursor, table: str, schema: str = "public") -> bool:
    """
    Examine if a table already exists.
    """
    query = """
        SELECT table_name from information_schema.tables
        WHERE table_name = %s AND table_schema = %s;
    """
    cur.execute(query, (table, schema))
    result = cur.fetchone()
    return bool(result)


def create_empty_tables(schema: str = "public"):
    """Creates empty tables to initialize the database."""
    # Set up
    connection_params = load_psycopg_params()
    # Connect to db pylint: disable=not-context-manager
    with psycopg.connect(connection_params) as conn:
        with conn.cursor() as cur:
            for param_dict in [config.MAP_PARAMS, config.RECORD_PARAMS]:
                for table, table_params in param_dict.items():
                    table_existing = table_exists(cur, table=table, schema=schema)
                    if not table_existing:
                        # Create query
                        query = _generate_create_table_query(
                            table_name=table, table_params=table_params, schema=schema
                        )
                        cur.execute(query)
                        print(f"Table '{table}' was created.")
                    else:
                        print(f"table '{table}' already exists.")

                    # Create indexes if specified
                    if "index_fields" in table_params:
                        create_indexes(
                            cur, table, table_params["index_fields"], schema=schema
                        )

    print("Tables created")


def delete_all_tables(schema: str = "public"):
    """Deletes all tables in the database.
    CAUTION: The deletion is permanent. DO NOT USE outside experimental settings.
    """
    connection_params = load_psycopg_params()

    # Connect to db pylint: disable=not-context-manager
    with psycopg.connect(connection_params) as conn:
        with conn.cursor() as cur:
            for param_dict in [config.MAP_PARAMS, config.RECORD_PARAMS]:
                for table in param_dict:
                    table_existing = table_exists(cur, table=table, schema=schema)
                    if table_existing:
                        query = f"DROP TABLE {schema}.{table} CASCADE;"
                        cur.execute(query)
                        print(f"Table '{table}' was deleted.")

    print("Existing tables deleted.")


def _upload_single_csv(
    csv_path: str,
    table: str,
    check_list: dict,
    schema: str = "public",
) -> None:
    """Uploads a CSV."""
    # Read csv first
    df = pd.read_csv(
        csv_path,
        dtype=str,
        header=0,
    )
    # Replace empty strigns with null
    df = df.replace("", pd.NA)

    # Check
    for col, params in check_list.items():
        if col in df:
            # Missing values
            non_null = params["non_null"]
            if non_null and df[col].isnull().any():
                raise ValueError(f"Column '{col}' contains null values in {csv_path}")

            # Data type conversion
            dtype = params["dtype"]
            null_mask = df[col].isna()
            if dtype == datetime:
                df["new_col"] = pd.NaT
                df.loc[~null_mask, "new_col"] = pd.to_datetime(
                    df.loc[~null_mask, col],
                    format="%Y/%m/%d %H:%M",
                    errors="raise",
                )
            elif dtype == date:
                df["new_col"] = pd.NaT
                df.loc[~null_mask, "new_col"] = pd.to_datetime(
                    df.loc[~null_mask, col],
                    format="%Y/%m/%d",
                    errors="raise",
                )
            elif dtype == float:
                df["new_col"] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == int:
                df["new_col"] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            else:
                df["new_col"] = pd.NA
                df.loc[~null_mask, "new_col"] = df.loc[~null_mask, col].astype(dtype)
            # Assign
            df[col] = df["new_col"]
            df.drop("new_col", axis=1, inplace=True)

        # Missing col
        else:
            # Raise if the column is required
            if params["col_required"]:
                raise ValueError(f"Column '{col}' is missing in {csv_path}")
            # Otherwise, create an empty column
            else:
                # Initialize missing optional columns
                dtype = params["dtype"]
                if dtype == datetime or dtype == date:
                    df[col] = pd.Series([pd.NaT] * len(df), index=df.index)
                elif dtype in [int, "int", "int32", "int64"]:
                    df[col] = pd.Series(
                        [pd.NA] * len(df), index=df.index, dtype="Int64"
                    )
                elif dtype in [float, "float", "float32", "float64"]:
                    df[col] = pd.Series(
                        [pd.NA] * len(df), index=df.index, dtype="Float64"
                    )
                elif dtype in [str, "str", "string"]:
                    df[col] = pd.Series(
                        [pd.NA] * len(df), index=df.index, dtype="string"
                    )
                else:
                    # fallback
                    df[col] = pd.Series([pd.NA] * len(df), index=df.index)

    # Eliminate unnecessary columns
    valid_cols = list(check_list.keys())
    df = df[valid_cols]

    # Write
    try:
        engine = load_db_engine()
        df.to_sql(table, con=engine, schema=schema, if_exists="append", index=False)
    except SQLAlchemyError as e:
        # This will show you the underlying INSERT or COPY error
        print("to_sql failed:", e)
        raise


def upload_csv(
    data_source: str,
    schema: str = "public",
    max_workers: int = 1,
):
    """Upload CSV files to the database.

    Scans the given data source directory for matching CSV files
    (using wildcards if necessary), validates and casts columns
    based on provided configuration, and uploads them into the
    specified database schema and tables.

    Args:
        data_source (str): Path to the directory or base file pattern.
        schema (str, optional): Database schema. Defaults to "public".
        max_workers (int, optional): Number of parallel workers. Defaults to 1.

    Raises:
        ValueError: If required columns are missing or invalid values are found.
    """
    # Upload clinical records
    for d in [config.MAP_PARAMS, config.RECORD_PARAMS]:
        for table, params in d.items():
            # Find files
            source_csv = os.path.join(data_source, params["source_csv"])
            if "*" in source_csv:
                src_files = glob.glob(source_csv)
            else:
                src_files = [source_csv]
            # Data type checks
            check_list = {}
            for col, tp in params["columns"].items():
                check_list[col] = {
                    "dtype": tp["pandas_dtype"],
                    "col_required": tp["col_required"],
                    "non_null": "NON NULL" in tp["sql_ops"],
                }

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _upload_single_csv,
                        csv_path=f,
                        table=table,
                        schema=schema,
                        check_list=check_list,
                    )
                    for f in src_files
                ]
                for future in tqdm(
                    as_completed(futures), desc=f"Uploading files to '{table}'"
                ):
                    _ = future.result()


def init_db_with_csv(
    data_source: str, max_workers: int = 1, schema="public", delete_existing=False
):
    """
    Initialize a PostgreSQL database using CSV files.

    This function sets up a PostgreSQL schema and populates it with tables
    from the specified CSV directory. It can optionally drop all existing tables before import.

    Warnings:
        - If `delete_existing` is True, all tables in the schema will be dropped. This action is irreversible.

    Note:
        - Prepare source clinical records as separated CSV files in accordance with :ref:`clinical_records`.


    Example:
        .. code-block:: python

            from watcher.db import init_db_with_csv

            init_db_with_csv(
                data_source="path/to/csv_files",
                max_workers=4,
                schema="public,
                delete_existing=True,
            )

    Args:
        data_source (str): Absolute path to the directory containing source CSV files.
        max_workers (int): Number of parallel workers for uploading. Defaults to 1.
        schema (str): Target PostgreSQL schema. Defaults to "public".
        delete_existing (bool): Whether to drop all existing tables before creation.

    Returns:
        None
    """

    # Delete old data
    if delete_existing:
        delete_all_tables(schema=schema)

    # Initialize tables (if not exists)
    create_schema(schema=schema)
    delete_all_tables(schema=schema)
    create_empty_tables(schema=schema)

    # Upload
    upload_csv(
        data_source=data_source,
        max_workers=max_workers,
        schema=schema,
    )
