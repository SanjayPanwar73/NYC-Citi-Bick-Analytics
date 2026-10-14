"""Import FIRST in every test module: isolates config to a temp root so tests never touch real data."""
import os
import sys
import tempfile
from pathlib import Path

os.environ["CITIBIKE_ROOT"] = os.environ.get("CITIBIKE_ROOT") or tempfile.mkdtemp(prefix="citibike_test_")
os.environ["DATA_MODE"] = "real"
os.environ["START_MONTH"] = "2024-01"
os.environ["END_MONTH"] = "2024-03"
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
