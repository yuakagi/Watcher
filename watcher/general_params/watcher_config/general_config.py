"""General configurations

NOTE: Use snake case without upper-case letters for naming throughout.
"""

# Formats
DATETIME_FORMAT = "%Y/%m/%d %H:%M"

# Postgres table names.
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
ALL_TB = [
    TB_PATIENTS,
    TB_ADMISSIONS,
    TB_DISCHARGES,
    TB_DIAGNOSES,
    TB_PRESC_ORD,
    TB_INJEC_ORD,
    TB_LAB_RES,
]

# Source table name patterns
SRC_PATIENT_TABLE = "patients.csv"  # <- single file (Other tables can be chunked.)
SRC_OUTPATIENT_VISIT_TABLE_PATTERN = "outpatient_visits*.csv"
SRC_ADMISSION_TABLE_PATTERN = "admission_records*.csv"
SRC_DISCHARGE_TABLE_PATTERN = "discharge_records*.csv"
SRC_DIAGNOSIS_TABLE_PATTERN = "diagnosis_records*.csv"
SRC_PRESCRIPTION_ORDER_TABLE_PATTERN = "prescription_order_records*.csv"
SRC_INJECTION_ORDER_TABLE_PATTERN = "injection_order_records*.csv"
SRC_LAB_RESULT_TABLE_PATTERN = "laboratory_test_results*.csv"

# Processed table name patterns (intermediate tables are serialized.)
VISITING_DATE_TABLE = "visiting_dates.pkl"
DEMOGRAPHIC_TABLE_PATTERN = "demographics_*.pkl"
OUTPATIENT_VISIT_TABLE_PATTERN = "outpatient_visits_*.pkl"
ADMISSION_TABLE_PATTERN = "admission_records_*.pkl"
DISCHARGE_TABLE_PATTERN = "discharge_records_*.pkl"
DIAGNOSIS_TABLE_PATTERN = "diagnosis_records_*.pkl"
PRESCRIPTION_ORDER_TABLE_PATTERN = "prescription_order_records_*.pkl"
INJECTION_ORDER_TABLE_PATTERN = "injection_order_records_*.pkl"
LAB_RESULT_TABLE_PATTERN = "laboratory_test_results_*.pkl"
LABELLED_FILE_PATTERN = "labelled_*.pkl"
EVAL_TABLE_FILE_PATTERN = "evaluation_table_*.pkl"

# Timeline file patterns
TRAJECTORY_BUNDLE_FILE_PATTERN = "timeline_bundle_*.pkl"
TRAJECTORY_STATS_PATTERN = "timeline_stats_*.json"
TRAJECTORY_METADATA = "timeline_metadata.pkl"

# Other directories
DIR_CHECKPOINTS = "checkpoints"
DIR_TENSORBOARD_LOGS = "tensorboard_logs"
DIR_TENSORBOARD_ACTIVE = "tensorboard_active"
DIR_CATALOGS = "catalogs"
DIR_LAB_STATS = "laboratory_stats"
DIR_SNAPSHOTS = "snapshots"
DIR_BLUEPRINT = "watcher_blueprint"
DIR_TRAJECTORY_BUNDLES = "timeline_bundles"

# Other paths, file names.
DX_CODE_TO_NAME = "dx_codes.csv"
MED_CODE_TO_NAME = "med_codes.csv"
LAB_CODE_TO_NAME = "lab_codes.csv"
TOTAL_PATIENT_ID_LIST = "total_patient_id_list.json"
INCLUDED_PATIENT_ID_LIST = "included_patient_id_list.json"
LAB_NUM_STATS_PATTERN = "numeric_stats_*.csv"
LAB_NONNUM_STATS_PATTERN = "nonnumeric_stats_*.csv"
LAB_PERCENTILES_PATTERN = "percentiles_*.csv"
MODEL_STATE = "model_state.pt"
TRAINING_STATE = "training_state.pt"
TRAINING_REPORT = "training_report.json"
MAIN_TRAINING_REPORT = "main_training_report.json"
CATALOG_FILE = "watcher_catalog.csv"
CATALOG_INFO_FILE = "catalog_info.json"

# Common names
TRAIN = "train"
VAL = "validation"
TEST = "test"
DX_CODE = "diagnosis_code"
MED_CODE = "medication_code"
LAB_CODE = "lab_test_code"
DMG = "demographics"
OPV = "outpatient_visits"
ADM = "admissions"
DSC = "discharges"
DX = "diagnosis"
PSC_O = "prescription_orders"
INJ_O = "injection_orders"
LAB_R = "lab_test_results"
EOT = "end_of_timeline"
PROV_SUFFIX = "(prov.)"


# Common column names
COL_ITEM_CODE = "item_code"
COL_ITEM_NAME = "item_name"
COL_RECORD_ID = "unique_record_id"
COL_PID = "patient_id"
COL_DOB = "date_of_birth"
COL_SEX = "sex"
COL_FIRST_VISIT_DATE = "first_visit_date"
COL_LAST_VISIT_DATE = "last_visit_date"
COL_DEPT = "department"
COL_TOKEN = "token"
COL_ORIGINAL_VALUE = "original_value"  # For code mapping.
COL_TOKENIZED_VALUE = "tokenized_value"
COL_TIMESTAMP = "timestamp"  # For timestamp col in cleaned table
COL_TIME_AVAILABLE = "time_available"
COL_TIMEDELTA = "timedelta"  # For timedelta col in processed table
COL_YEARS = "years"
COL_MONTHS = "months"
COL_DAYS = "days"
COL_HOURS = "hours"
COL_MINUTES = "minutes"
COL_NUMERIC = "numeric"
COL_NONNUMERIC = "nonnumeric"
COL_ORIGINAL_NUMERIC = "original_numeric"
COL_ORIGINAL_NONNUMERIC = "original_nonnumeric"
COL_TYPE = "type"  # For columns to identify record type.
COL_PRIORITY = "priority"
COL_TASK_NO = "task_number"
COL_ITEM_NO = "item_number"
COL_PROVISIONAL_FLAG = "provisional"
COL_TRAIN_PERIOD = "train_period"
COL_TEST_PERIOD = "test_period"
COL_UPDATE_PERIOD = "update_period"
COL_ROW_NO = "row_number"
COL_LABEL = "label"
COL_ADM = "admitted"
COL_INCLUDED = "included"
COL_TEXT = "text"
COL_AGE = "age"
COL_CODE = "code"
COL_RESULT = "result"

# Column groups
SORT_COLS = [
    COL_PID,  # Patient ID
    COL_TIMEDELTA,
    COL_TYPE,  # Record type
    COL_TASK_NO,  # Number of task processed by child processes
    COL_ITEM_NO,
    COL_PRIORITY,  # Priority among input types
]
PREPROCESS_META_COLS = [
    COL_PID,
    COL_TIMESTAMP,
    COL_TIMEDELTA,
    COL_TIME_AVAILABLE,
    COL_ORIGINAL_VALUE,
    COL_TYPE,
    COL_TASK_NO,
    COL_ITEM_NO,
    COL_PRIORITY,
    COL_RECORD_ID,
]
FINAL_META_COLS = [
    COL_PID,
    COL_RECORD_ID,
    COL_ROW_NO,
    COL_TYPE,
    COL_TIMESTAMP,
    COL_TIME_AVAILABLE,
    COL_TIMEDELTA,
    COL_ORIGINAL_VALUE,
    COL_TRAIN_PERIOD,
    COL_TEST_PERIOD,
]
TIMEDELTA_COMPONENT_COLS = [
    COL_YEARS,
    COL_MONTHS,
    COL_DAYS,
    COL_HOURS,
    COL_MINUTES,
]

# Strings recognized as missing values (passed to pd.read_csv as 'na_values' )
NA_VALUES = ["", "NA", "N/A", "na", "nan", "NaN", "NaT", "None", "none"]

# Random seed
SEED = 0

# Multiprocessing related configs
LOGICAL_CPU = False
CPU_MARGIN = 1

# Max number of code segments
MAX_CATEGORICAL_DIM = 10
# Dimensions
NUMERIC_DIM = 1
# Number of the demographic rows per timeline
DEMOGRAPHIC_ROWS = 2
# Training configs (Unexpected abrupt surge in loss over this threshold forces training to restart.)
LOSS_SURGE_THRESHOLD = 2.0

# Training related
SHUFFLE_INPUT = True
MAX_RESAMPLING = 10
