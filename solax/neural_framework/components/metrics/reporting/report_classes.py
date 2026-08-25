"""
Low-level pieces used by the reporting context managers to format and
write one line of text per reported metrics entry: string-formatting
helpers, an in-memory string buffer (Report), and the Reporter that
ties a buffer and a prefix together.
"""
from dataclasses import dataclass
import io
import typing


class Report(io.StringIO):
    """
    An in-memory text buffer (subclass of io.StringIO) used as the
    non-stdout report destination by reporting()/reporting_updates().
    repr() returns the buffer's full contents so far (instead of the
    default io.StringIO repr), for easy inspection.
    """
    def __repr__(self):
        """Returns the buffer's full contents so far (see class docstring)."""
        return super().getvalue()


def form_entry_str(entry, *, sep="\t"):
    """Formats a metrics entry dict as "name=value" pairs (values in
    scientific notation), joined by "sep"."""
    return sep.join([f"{name}={value:e}" for name, value in entry.items()])


def form_prefix_str(prefix, label, *, sep=" "):
    """Joins "prefix" and "label" with "sep", dropping either if it is
    falsy (empty)."""
    return sep.join(s for s in [f"{prefix}", f"{label}"] if s)


def form_printed_str(prefix_str, entry_str, *, sep=":\t"):
    """Joins a prefix string and an entry string with "sep", dropping
    either if it is falsy (empty)."""
    return sep.join(s for s in [prefix_str, entry_str] if s)


@dataclass
class Reporter:
    """
    Writes one formatted line per reported metrics entry to "buffer".
    Each line has the form "<prefix> <label>:\t<name>=<value>\t...",
    with the prefix/label part or the entry part dropped if empty.
    """
    buffer: typing.IO
    prefix: str = ""

    def make_report(self, entry: dict, label):
        """Formats "entry" (a metric name -> value dict) and "label"
        into one line (see form_entry_str/form_prefix_str/
        form_printed_str) and prints it to "self.buffer"."""
        entry_str = form_entry_str(entry)
        prefix_str = form_prefix_str(self.prefix, label)
        printed_str = form_printed_str(prefix_str, entry_str)
        print(printed_str, file=self.buffer)