"""Module for the postgresql database."""

from datetime import datetime, date
from .general_config import *


# Table names.
TB_PATIENTS = "patients"
TB_ADMISSIONS = "admissions"
TB_DISCHARGES = "discharges"
TB_DIAGNOSES = "diagnoses"
TB_PRESC_ORD = "prescription_orders"
TB_INJEC_ORD = "injection_orders"
TB_LAB_RES = "laboratory_results"
TB_DX_CODES = "diagnosis_codes"
TB_MED_CODES = "medication_codes"
TB_LAB_CODES = "lab_test_codes"

# Table definitions for code maps
MAP_PARAMS = {
    TB_DX_CODES: {
        "source_csv": DX_CODE_TO_NAME,
        "columns": {
            COL_ITEM_CODE: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200) UNIQUE NOT NULL",
            },
            COL_ITEM_NAME: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200) NOT NULL",
            },
        },
        "index_fields": [],
        "primary_key": COL_ITEM_CODE,
        "foreign_key": None,
    },
    TB_MED_CODES: {
        "source_csv": MED_CODE_TO_NAME,
        "columns": {
            COL_ITEM_CODE: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200) UNIQUE NOT NULL",
            },
            COL_ITEM_NAME: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200) NOT NULL",
            },
        },
        "index_fields": [],
        "primary_key": COL_ITEM_CODE,
        "foreign_key": None,
    },
    TB_LAB_CODES: {
        "source_csv": LAB_CODE_TO_NAME,
        "columns": {
            COL_ITEM_CODE: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200) UNIQUE NOT NULL",
            },
            COL_ITEM_NAME: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200) NOT NULL",
            },
        },
        "index_fields": [],
        "primary_key": COL_ITEM_CODE,
        "foreign_key": None,
    },
}

# Table definitions for clinical records
RECORD_PARAMS = {
    TB_PATIENTS: {
        "source_csv": SRC_PATIENT_TABLE,
        "id_prefix": "PT",
        "columns": {
            # Others
            COL_PID: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200) UNIQUE NOT NULL",
            },
            COL_SEX: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": f"CHAR(1) NOT NULL CHECK ({COL_SEX} IN ('M', 'F', 'O', 'U', 'A', 'N'))",
            },
            "first_name": {
                "col_required": False,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(50)",
            },
            "last_name": {
                "col_required": False,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(50)",
            },
            COL_DOB: {
                "col_required": True,
                "pandas_dtype": date,
                "sql_ops": "DATE NOT NULL",
            },
            "index_fields": ["patient_id", COL_TIMESTAMP],
        },
        # TODO: make 'COL_PID' primary key.
        "primary_key": COL_RECORD_ID,
        "foreign_key": None,
    },
    TB_LAB_RES: {
        "source_csv": SRC_LAB_RESULT_TABLE_PATTERN,
        "id_prefix": "LABRSLT",
        "columns": {
            COL_PID: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200) NOT NULL",
            },
            COL_TIMESTAMP: {
                "col_required": True,
                "pandas_dtype": datetime,
                "sql_ops": "TIMESTAMP NOT NULL",
            },
            COL_ITEM_CODE: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200) NOT NULL",
            },
            COL_NUMERIC: {
                "col_required": True,
                "pandas_dtype": float,
                "sql_ops": "numeric",
            },
            "unit": {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "varchar(20)",
            },
            COL_NONNUMERIC: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200)",
            },
            COL_TIME_AVAILABLE: {
                "col_required": False,
                "pandas_dtype": datetime,
                "sql_ops": "TIMESTAMP",
            },
        },
        "index_fields": ["patient_id", COL_TIMESTAMP],
        "primary_key": COL_RECORD_ID,
        "foreign_key": (COL_PID, TB_PATIENTS),
    },
    TB_ADMISSIONS: {
        "source_csv": SRC_ADMISSION_TABLE_PATTERN,
        "id_prefix": "ADM",
        "columns": {
            COL_PID: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200) NOT NULL",
            },
            COL_TIMESTAMP: {
                "col_required": True,
                "pandas_dtype": datetime,
                "sql_ops": "TIMESTAMP NOT NULL",
            },
            COL_DEPT: {
                "col_required": False,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200)",
            },
        },
        "index_fields": ["patient_id", COL_TIMESTAMP],
        "primary_key": COL_RECORD_ID,
        "foreign_key": (COL_PID, TB_PATIENTS),
    },
    TB_DISCHARGES: {
        "source_csv": SRC_DISCHARGE_TABLE_PATTERN,
        "id_prefix": "DSC",
        "columns": {
            # Others
            COL_PID: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200) NOT NULL",
            },
            COL_TIMESTAMP: {
                "col_required": True,
                "pandas_dtype": datetime,
                "sql_ops": "TIMESTAMP NOT NULL",
            },
            "disposition": {
                "col_required": True,
                "pandas_dtype": int,
                "sql_ops": "INTEGER CHECK (disposition IN (0, 1)) NOT NULL",
            },
        },
        "index_fields": ["patient_id", COL_TIMESTAMP],
        "primary_key": COL_RECORD_ID,
        "foreign_key": (COL_PID, TB_PATIENTS),
    },
    TB_DIAGNOSES: {
        "source_csv": SRC_DIAGNOSIS_TABLE_PATTERN,
        "id_prefix": "DX",
        "columns": {
            COL_PID: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200) NOT NULL",
            },
            COL_TIMESTAMP: {
                "col_required": True,
                "pandas_dtype": datetime,
                "sql_ops": "TIMESTAMP NOT NULL",
            },
            COL_ITEM_CODE: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200) NOT NULL",
            },
            COL_PROVISIONAL_FLAG: {
                "col_required": True,
                "pandas_dtype": int,
                "sql_ops": f"INTEGER CHECK ({COL_PROVISIONAL_FLAG} IN (0, 1)) NOT NULL",
            },
        },
        "index_fields": ["patient_id", COL_TIMESTAMP],
        "primary_key": COL_RECORD_ID,
        "foreign_key": (COL_PID, TB_PATIENTS),
    },
    TB_PRESC_ORD: {
        "source_csv": SRC_PRESCRIPTION_ORDER_TABLE_PATTERN,
        "id_prefix": "PRSCORD",
        "columns": {
            COL_PID: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200) NOT NULL",
            },
            COL_TIMESTAMP: {
                "col_required": True,
                "pandas_dtype": datetime,
                "sql_ops": "TIMESTAMP NOT NULL",
            },
            COL_ITEM_CODE: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200) NOT NULL",
            },
        },
        "index_fields": ["patient_id", COL_TIMESTAMP],
        "primary_key": COL_RECORD_ID,
        "foreign_key": (COL_PID, TB_PATIENTS),
    },
    TB_INJEC_ORD: {
        "source_csv": SRC_INJECTION_ORDER_TABLE_PATTERN,
        "id_prefix": "INJCORD",
        "columns": {
            COL_PID: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200) NOT NULL",
            },
            COL_TIMESTAMP: {
                "col_required": True,
                "pandas_dtype": datetime,
                "sql_ops": "TIMESTAMP NOT NULL",
            },
            COL_ITEM_CODE: {
                "col_required": True,
                "pandas_dtype": str,
                "sql_ops": "VARCHAR(200) NOT NULL",
            },
        },
        "index_fields": ["patient_id", COL_TIMESTAMP],
        "primary_key": COL_RECORD_ID,
        "foreign_key": (COL_PID, TB_PATIENTS),
    },
}
