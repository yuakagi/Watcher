"""Function wrapper for general purposes"""

import os
from datetime import datetime
import json
import gc
from typing import Any


# Write performance history
def write_history(key: str, json_path: str, value: Any = "now"):
    """Writes analytic data to a json file.
    Args:
        key (str): Dictionary key
        json_path (str): Path to a json file. Values will be written to this file.
        value (Any): Value to be written to the json file with the dictionary key.
            If "now" is passed to this argument, current time is recorded.
    """
    if value == "now":
        value = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            dct = json.load(f)
            dct[key] = value
    else:
        dct = {key: value}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dct, f, indent=2)


def function_wrapper(
    function,
    section_description: str,
    step_description: str,
    json_path: str,
    **kwargs,
):
    """Wrapper function for preprocess functions.
    This function performs:
        - 1. Visualization of details of each step for CLI.
        - 2. Execute a custom function.
        - 3. Record step details in a json file.
    Args:
        function (function): Function to be executed.
        section_description (str): Short description of the current step.
        json_path (str): See 'write_history'.
    """
    displayed_text = f"|| {section_description}: {step_description} ||"
    text_length = len(displayed_text)
    print("*" * (text_length))
    print("||" + "*" * (text_length - 4) + "||")
    print(displayed_text)
    print("||" + "*" * (text_length - 4) + "||")
    print("*" * (text_length))

    print(f"{step_description} initiated...")
    log_key = f"{step_description} initiated"

    # Wirte the time of initiation
    start_dt = datetime.now()
    start = start_dt.strftime("%Y/%m/%d %H:%M:%S")
    write_history(log_key, json_path=json_path, value=start)

    # Execute function
    analytics = function(**kwargs)
    gc.collect()

    # Write the time of finishing
    print(f"{step_description} finished.")
    log_key = f"{step_description} finished"
    end_dt = datetime.now()
    end = end_dt.strftime("%Y/%m/%d %H:%M:%S")
    write_history(log_key, json_path=json_path, value=end)

    log_key = f"{step_description} duration"
    duration = (end_dt - start_dt).total_seconds()
    duration_hr = int(duration // 60**2)
    duration_min = int((duration % 60**2) // 60)
    duration_str = f"{duration_hr} hours {duration_min} min"
    write_history(log_key, json_path=json_path, value=duration_str)

    # Write the process analytics
    if analytics:
        write_history(
            key=f"process_details: {step_description})",
            value=analytics,
            json_path=json_path,
        )
