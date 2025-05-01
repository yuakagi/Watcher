"""Configs for preprocessing"""

from datetime import datetime, timedelta
from .general_config import *


# ***** Table definitions *****
# NOTE: Optional columns are created during cleaning.
# TODO: In the first cleaning step, validate input
# TODO: 'time_available' is currently not intended for end-user use. Therefore, create the col if not existing.

CODE_MAP_COLS = {COL_ITEM_CODE: str, COL_ITEM_NAME: str}


# Minimum column requirements for Postgres DB
RAW_PATIENT_COLS = {
    # This must be a single table.
    COL_PID: str,  # Non-null, unique
    COL_SEX: str,  # Non-null
    COL_DOB: datetime,  # Non-null, YYYY/MM/DD
    COL_RECORD_ID: str,  # Non-null, PK
}
RAW_OPV_COLS = {
    COL_PID: str,  # Non-null
    COL_DEPT: str,
    COL_TIMESTAMP: datetime,  # YYYY/MM/DD HH:MM, non-null ("visiting_date")
    COL_RECORD_ID: str,  # Non-null, PK
}
RAW_ADM_COLS = {
    COL_PID: str,  # Non-null
    COL_TIMESTAMP: datetime,  # YYYY/MM/DD HH:MM, non-null ("admission_date")
    COL_RECORD_ID: str,  # Non-null, PK
    # Optional: COL_DEPT
}
RAW_DSC_COLS = {
    COL_PID: str,  # Non-null
    COL_TIMESTAMP: datetime,  # YYYY/MM/DD HH:MM, non-null ("discharge_date")
    "disposition": int,  # binary (0 for exp 1 for alive), Non-null
    COL_RECORD_ID: str,  # Non-null, PK
}
RAW_DX_COLS = {
    COL_PID: str,  # Non-null
    COL_ITEM_CODE: str,  # Non-null
    COL_PROVISIONAL_FLAG: int,  # binary (1 for provisional), Non-null
    COL_TIMESTAMP: datetime,  # "time_of_update", Non-null
    COL_RECORD_ID: str,  # Non-null, PK
    # * 'item_name' for Postgres
}
RAW_PSC_O_COLS = {
    COL_PID: str,  # Non-null
    COL_ITEM_CODE: str,  # Non-null
    COL_TIMESTAMP: datetime,  # "start_of_order"
    COL_RECORD_ID: str,  # Non-null, PK
    # * 'item_name' for Postgres
}
RAW_INJ_O_COLS = {
    COL_PID: str,  # Non-null
    COL_ITEM_CODE: str,  # Non-null
    COL_TIMESTAMP: datetime,  # "start_of_order"
    COL_RECORD_ID: str,
    # * 'item_name' for Postgres
}
RAW_LAB_R_COLS = {
    COL_PID: str,
    COL_ITEM_CODE: str,
    COL_NUMERIC: float,  # "value"
    "unit": str,
    COL_NONNUMERIC: str,  # "value"
    COL_TIMESTAMP: datetime,  # "sampled_time"
    COL_RECORD_ID: str,
    # Optional: COL_TIME_AVAILABLE: timedelta ("reported_time")
    # * 'item_name' for Postgres
}

CLEANED_DMG_COLS = {
    COL_PID: str,
    COL_SEX: str,
    COL_DOB: datetime,
    COL_RECORD_ID: str,
}
CLEANED_OPV_COLS = {
    COL_PID: str,
    COL_DEPT: str,
    COL_TIMEDELTA: timedelta,  # "visiting_date"
    COL_TIMESTAMP: datetime,  # "visiting_date"
    COL_TIME_AVAILABLE: timedelta,
    COL_RECORD_ID: str,
}
CLEANED_ADM_COLS = {
    COL_PID: str,
    COL_TIMEDELTA: timedelta,  # "admission_date"
    COL_TIMESTAMP: datetime,  # "admission_date"
    COL_TIME_AVAILABLE: timedelta,
    COL_RECORD_ID: str,
    # Optional:
    COL_DEPT: str,
}
CLEANED_DSC_COLS = {
    COL_PID: str,
    COL_TIMEDELTA: timedelta,  # "discharge_date"
    COL_TIMESTAMP: datetime,  # "discharge_date"
    "disposition": int,  # 0 or 1, binary indicator
    COL_TIME_AVAILABLE: timedelta,
    COL_RECORD_ID: str,
}
CLEANED_DX_COLS = {
    COL_PID: str,
    COL_ITEM_CODE: str,
    COL_PROVISIONAL_FLAG: int,
    COL_TIMEDELTA: timedelta,  # "time_of_update"
    COL_TIMESTAMP: datetime,  # "time_of_update"
    COL_TIME_AVAILABLE: timedelta,
    COL_RECORD_ID: str,
}

CLEANED_PSC_O_COLS = {
    COL_PID: str,
    COL_ITEM_CODE: str,
    COL_TIMEDELTA: timedelta,  # "start_of_order"
    COL_TIMESTAMP: datetime,  # "start_of_order"
    COL_TIME_AVAILABLE: timedelta,
    COL_RECORD_ID: str,
}

CLEANED_INJ_O_COLS = {
    COL_PID: str,
    COL_ITEM_CODE: str,
    COL_TIMEDELTA: timedelta,  # "start_of_order"
    COL_TIMESTAMP: datetime,  # "start_of_order"
    COL_TIME_AVAILABLE: timedelta,
    COL_RECORD_ID: str,
}

CLEANED_LAB_R_COLS = {
    COL_PID: str,
    COL_ITEM_CODE: str,
    COL_NUMERIC: float,  # "value"
    "unit": str,
    COL_NONNUMERIC: str,  # "value"
    COL_TIMEDELTA: timedelta,  # "sampled_time"
    COL_TIMESTAMP: datetime,  # "sampled_time"
    COL_TIME_AVAILABLE: timedelta,  # "reported_time"
    COL_RECORD_ID: str,
}

VISITING_DATE_TABLE_COLS = {
    COL_PID: str,
    COL_FIRST_VISIT_DATE: datetime,
    COL_LAST_VISIT_DATE: datetime,
}

DEFAULT_CATEGORICAL_COLS = {f"c{i}": int for i in range(MAX_CATEGORICAL_DIM)}

SEQUENCED_TABLE_COLS = {
    COL_PID: str,
    COL_TIMESTAMP: datetime,
    COL_TIMEDELTA: timedelta,
    COL_TIME_AVAILABLE: timedelta,
    COL_ORIGINAL_VALUE: str,
    COL_TYPE: int,
    COL_TASK_NO: int,
    COL_ITEM_NO: int,
    COL_PRIORITY: int,
    COL_RECORD_ID: str,
    COL_YEARS: float,
    COL_MONTHS: float,
    COL_DAYS: float,
    COL_HOURS: float,
    COL_MINUTES: float,
    COL_NUMERIC: float,
    **DEFAULT_CATEGORICAL_COLS,
}

AGG_TABLE_COLS = {**SEQUENCED_TABLE_COLS}

FINAL_AGG_TABLE_COLS = {
    COL_PID: str,
    COL_RECORD_ID: str,
    COL_ROW_NO: int,
    COL_TYPE: int,
    COL_TIMESTAMP: datetime,
    COL_TIME_AVAILABLE: timedelta,
    COL_TIMEDELTA: timedelta,
    COL_ORIGINAL_VALUE: str,
    COL_YEARS: float,
    COL_MONTHS: float,
    COL_DAYS: float,
    COL_HOURS: float,
    COL_MINUTES: float,
    COL_NUMERIC: float,
    **DEFAULT_CATEGORICAL_COLS,
    COL_ADM: int,
}

LABELLED_TABLE_COLS = {
    COL_PID: str,
    COL_RECORD_ID: str,
    COL_ROW_NO: int,
    COL_TYPE: int,
    COL_TIMESTAMP: datetime,
    COL_TIME_AVAILABLE: timedelta,
    COL_TIMEDELTA: timedelta,
    COL_ORIGINAL_VALUE: str,
    COL_TRAIN_PERIOD: int,
    COL_TEST_PERIOD: int,
    COL_YEARS: float,
    COL_MONTHS: float,
    COL_DAYS: float,
    COL_HOURS: float,
    COL_MINUTES: float,
    COL_NUMERIC: float,
    **DEFAULT_CATEGORICAL_COLS,
    COL_ADM: int,
    COL_LABEL: int,
    COL_INCLUDED: int,
}

EVAL_TABLE_COLS = {
    COL_PID: str,
    COL_TYPE: int,
    COL_AGE: timedelta,
    COL_CODE: str,
    COL_TEXT: str,
    COL_RESULT: str,
    COL_TRAIN_PERIOD: int,
    COL_TEST_PERIOD: int,
}

# ***** Preprocess params *****
CLEANING_PARAMS = {
    TB_PATIENTS: {
        "record_type": DMG,
        "dst_table_pattern": DEMOGRAPHIC_TABLE_PATTERN,
        # Records are dropped if 'any' of these are missing
        "dropna_any": [
            COL_PID,
            COL_SEX,
            COL_RECORD_ID,
        ],
        # Records are dropped if all of the list are missing
        "dropna_all": None,
        # List of final columns and their dtypes
        "final_cols": CLEANED_DMG_COLS,
    },
    TB_ADMISSIONS: {
        "record_type": ADM,
        "dst_table_pattern": ADMISSION_TABLE_PATTERN,
        # Records are dropped if 'any' of these are missing
        "dropna_any": [
            COL_PID,
            COL_TIMESTAMP,
            COL_RECORD_ID,
        ],
        # Records are dropped if all of the list are missing
        "dropna_all": None,
        # List of final columns and their dtypes
        "final_cols": CLEANED_ADM_COLS,
    },
    TB_DISCHARGES: {
        "record_type": DSC,
        "dst_table_pattern": DISCHARGE_TABLE_PATTERN,
        # Records are dropped if 'any' of these are missing
        "dropna_any": [
            COL_PID,
            COL_TIMESTAMP,
            "disposition",
            COL_RECORD_ID,
        ],
        # Records are dropped if all of the list are missing
        "dropna_all": None,
        # List of final columns and their dtypes
        "final_cols": CLEANED_DSC_COLS,
    },
    TB_DIAGNOSES: {
        "record_type": DX,
        "dst_table_pattern": DIAGNOSIS_TABLE_PATTERN,
        # Records are dropped if 'any' of these are missing
        "dropna_any": [
            COL_PID,
            COL_ITEM_CODE,
            COL_PROVISIONAL_FLAG,
            COL_TIMESTAMP,
            COL_RECORD_ID,
        ],
        # Records are dropped if all of the list are missing
        "dropna_all": None,
        # List of final columns and their dtypes
        "final_cols": CLEANED_DX_COLS,
    },
    TB_PRESC_ORD: {
        "record_type": PSC_O,
        "dst_table_pattern": PRESCRIPTION_ORDER_TABLE_PATTERN,
        # Records are dropped if 'any' of these are missing
        "dropna_any": [
            COL_PID,
            COL_ITEM_CODE,
            COL_TIMESTAMP,
            COL_RECORD_ID,
        ],
        # Records are dropped if all of the list are missing
        "dropna_all": None,
        # List of final columns and their dtypes
        "final_cols": CLEANED_PSC_O_COLS,
    },
    TB_INJEC_ORD: {
        "record_type": INJ_O,
        "dst_table_pattern": INJECTION_ORDER_TABLE_PATTERN,
        # Records are dropped if 'any' of these are missing
        "dropna_any": [
            COL_PID,
            COL_ITEM_CODE,
            COL_TIMESTAMP,
            COL_RECORD_ID,
        ],
        # Records are dropped if all of the list are missing
        "dropna_all": None,
        # List of final columns and their dtypes
        "final_cols": CLEANED_INJ_O_COLS,
    },
    TB_LAB_RES: {
        "record_type": LAB_R,
        "dst_table_pattern": LAB_RESULT_TABLE_PATTERN,
        # Records are dropped if 'any' of these are missing
        "dropna_any": [
            COL_PID,
            COL_ITEM_CODE,
            COL_TIMESTAMP,
        ],
        # Records are dropped if all of the list are missing
        "dropna_all": [COL_NUMERIC, COL_NONNUMERIC],
        # List of final columns and their dtypes
        "final_cols": CLEANED_LAB_R_COLS,
    },
}

# Future implementation of OPV
# NOTE: Currently, outpatient visit records are not a part of model inputs, only used for downstream tasks.
#       You must be careful handling the visiting dates for model input because they usually only contain date-level timestamps.
"""{
    OPV: {
        "record_type": "outpatient_visits",
        "dst_table_pattern": OUTPATIENT_VISIT_TABLE_PATTERN,
        # Records are dropped if 'any' of these are missing
        "dropna_any": [
            COL_PID,
            COL_TIMESTAMP,
            COL_RECORD_ID,
        ],
        # Records are dropped if both of the columns in this list are missing together
        "dropna_all": None,
        # List of final columns and their dtypes
        "final_cols": CLEANED_OPV_COLS,
    },
}"""


# Lab cleaning
POS_NEG_DICT = {
    "(+)": "[pos]",
    "(-)": "[neg]",
    "(+-)": "[n/p]",
}

# ***** Laboratory stats configs *****
# NOTE: These limits are only applied for preprocessing.
OUTLIER_LIMIT = 0.005  # <- in percentile, like 0.01
UPPER_OUTLIER_LIMIT = 0.9999  # Deprecated
LOWER_OUTLIER_LIMIT = 0.0001  # Deprecated
PATIENT_BATCH_SIZE_FOR_LAB = 50

# ***** Sequencing-related configs *****
# Type col
# NOTE: Record type for [EOT] is supposed to be the largest number
# TODO (Yu Akagi): Consider fusion of PSC_O and INJ_O for easy understanding.
RECORD_TYPE_NUMBERS = {
    DMG: 0,
    ADM: 1,
    DSC: 2,
    DX: 3,
    PSC_O: 4,
    INJ_O: 5,
    LAB_R: 6,
    EOT: 7,
}
ISOLATE_PROVISIONAL_DIAGNOSES = True

# ***** Aggregation-related params *****
PATIENTS_PER_FILE = 10000

# ***** Labelling-related configs *****
TD_SECTIONS = [[1, 30, 1], [31, 180, 10], [181, 360, 30], [361, 1800, 180]]
