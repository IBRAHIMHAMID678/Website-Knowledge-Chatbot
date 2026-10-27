import sys
from pathlib import Path

# Add project root directory to sys.path so app imports work seamlessly in Vercel Lambda
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import app
