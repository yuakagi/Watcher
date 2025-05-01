"""Configs for tokenization"""

from .general_config import *

# Index for OOV
OOV_INDEX = -2
LOGITS_IGNORE_INDEX = -1

# Special tokens
SPECIAL_TOKENS = [
    "[PAD]",  # <- [PAD] must always be at the top
    "[DX]",  # Diagnosis code
    "[MED]",  # Medication code
    "[LAB]",  # Laboratory test code
    "[ADM]",  # Admission
    "[DSC]",  # Discharge
    "[PRV]",  # Provisional diagnosis flag
    "[EOT]",  # End of timeline
    "[OOV]",  # Out of vocabulary
]

# Tokens for sex (from HL7 v2.8 user-defined table 0001)
# NOTE: Depending on EHR systems, some of these are not used at all. But still all of them are included in the model's vocabulary.
SEX_TOKENS = {
    "F": "[F]",  # Female
    "M": "[M]",  # Male
    "O": "[O]",  # Other
    "U": "[U]",  # Unknown
    "A": "[A]",  # Ambiguous
    "N": "[N]",  # Not applicable
}

# Tokens for discharge dispositions
DISCHARGE_STATUS_TOKENS = {0: "[DSC_EXP]", 1: "[DSC_ALV]"}

# Tokens and descriptions
TOKEN_DESCRIPTIONS = {
    "[PAD]": "padding",
    "[DX]": "diagnosis code",
    "[MED]": "medication code",
    "[LAB]": "laboratory test code",
    "[ADM]": "admitted",
    "[DSC]": "discharged",
    "[PRV]": "provisional",
    "[DSC_ALV]": "survived",
    "[DSC_EXP]": "expired",
    "[M]": "male patient",
    "[F]": "female patient",
    "[O]": "other sex",
    "[U]": "unknown sex",
    "[A]": "ambiguous sex",
    "[N]": "sex not applicable",
    "[EOT]": "end of timeline",
    "[OOV]": "out of vocabulary",
}
