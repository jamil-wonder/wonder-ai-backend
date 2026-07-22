"""Wonder AI Backend application loader.

The implementation is split into ordered chunks under ``backend/main_parts``.
Each chunk is executed in this module namespace so existing globals, route
decorators, startup handlers, and public API behavior stay unchanged while the
source files remain small enough to maintain.
"""
from pathlib import Path

_CHUNK_DIR = Path(__file__).with_name("main_parts")
_CHUNK_FILES = [
    "part_01.py",
    "part_02.py",
    "part_03.py",
    "part_04.py",
    "part_05.py",
    "part_06.py",
    "part_07.py",
    "part_08.py",
    "part_09.py",
    "part_10.py",
    "part_11.py",
    "part_12.py",
]

for _chunk_file in _CHUNK_FILES:
    _chunk_path = _CHUNK_DIR / _chunk_file
    exec(compile(_chunk_path.read_text(encoding="utf-8"), str(_chunk_path), "exec"), globals())
