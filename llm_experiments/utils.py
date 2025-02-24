import os
from pathlib import Path

from dotenv import load_dotenv


def get_app_root():
    load_dotenv()
    return Path(os.getenv("LLM_EXPERIMENTS_PATH"))
