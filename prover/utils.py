import json
import os
from loguru import logger

def log_failed_tactic(state: str, tactic: str, error: str, theorem_name: str):
    log_file = os.environ.get("FAILED_TACTIC_LOG")
    if not log_file:
        return

    entry = {
        "state": state,
        "tactic": tactic,
        "error": error,
        "theorem": theorem_name,
    }
    
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write to failed tactic log: {e}")
