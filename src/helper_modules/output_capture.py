# helper module to save output of the program to a .md file

import sys
from io import StringIO
from rich.text import Text


class OutputCapture:
    """Captures all stdout output and saves a clean version to a file."""

    def __init__(self, path="output.md"):
        self.path = path
        self._original_stdout = None
        self._buffer = None

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._buffer = StringIO()
        sys.stdout = _Tee(self._original_stdout, self._buffer)
        return self

    def __exit__(self, *exc):
        sys.stdout = self._original_stdout
        raw_output = self._buffer.getvalue()
        clean_text = Text.from_ansi(raw_output).plain
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(clean_text)
        self._buffer.close()


class _Tee:
    """Writes to two streams simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    @property
    def encoding(self):
        return self.streams[0].encoding

    def isatty(self):
        return self.streams[0].isatty()