import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Retrieval in tests is TF-IDF only: deterministic, and no 26 s embedding of
# the library (forge/kit/library.py), which on a busy Mac hung the suite.
import os
os.environ.setdefault("FORGE_RETRIEVAL", "tfidf")
# The server defaults the library to "release" (held-out topics kept) when it
# is imported; tests always see the evaluation library.
os.environ["FORGE_LIBRARY"] = "eval"
# No adapter download from the Hub in tests.
os.environ["FORGE_ADAPTER"] = "none"
