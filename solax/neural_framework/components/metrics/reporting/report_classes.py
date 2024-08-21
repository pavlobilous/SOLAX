from dataclasses import dataclass
import io, typing


class Report(io.StringIO):
    def __repr__(self):
        return super().getvalue()
    
    
def form_entry_str(entry, *, sep="\t"):
    return sep.join([f"{name}={value:e}" for name, value in entry.items()])


def form_prefix_str(prefix, label, *, sep=" "):
    return sep.join(s for s in [f"{prefix}", f"{label}"] if s)


def form_printed_str(prefix_str, entry_str, *, sep=":\t"):
    return sep.join(s for s in [prefix_str, entry_str] if s)
    
    
@dataclass
class Reporter:
    buffer: typing.IO
    prefix: str = ""
    
    def make_report(self, entry: dict, label):
        entry_str = form_entry_str(entry)
        prefix_str = form_prefix_str(self.prefix, label)
        printed_str = form_printed_str(prefix_str, entry_str)
        print(printed_str, file=self.buffer)