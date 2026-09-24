"""3Blue1Brown-style building blocks; see kit.py. KIT_SOURCE is the file's
text, which the assembler prepends to a scene so it renders anywhere."""
from pathlib import Path

from .kit import *  # noqa: F401,F403
from .kit import KIT_API  # noqa: F401

KIT_SOURCE = (Path(__file__).parent / "kit.py").read_text()
