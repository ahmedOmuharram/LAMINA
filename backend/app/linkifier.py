import re
from .utils import linkify_mp_numbers

class MPLinkifyBuffer:
    """
    Buffers text so we don't emit partial matches of (?<!\\w)mp-\\d+ across chunks.
    Only the 'safe head' (which cannot be extended into a match) is emitted,
    and the tail is kept until the boundary is certain.
    """
    def __init__(self):
        self.buf = ""

    def _split_safe_head(self, s: str) -> int:
        """
        Return index 'safe_idx' such that s[:safe_idx] can be finalized now.
        Keep any trailing potential start of an mp-<digits> token in the buffer.
        """
        n = len(s)
        safe_idx = n

        # If tail looks like start of a candidate, keep it.
        # 1) mp-<digits>* at the very end (could extend with more digits)
        m = re.search(r'(?<!\w)mp-\d*$', s)
        if m:
            safe_idx = min(safe_idx, m.start())

        # 2) Bare 'mp-' at end
        if s.endswith("mp-"):
            safe_idx = min(safe_idx, n - 3)
        # 3) Bare 'mp' at end
        elif s.endswith("mp"):
            safe_idx = min(safe_idx, n - 2)
        # 4) Bare 'm' at end
        elif s.endswith("m"):
            safe_idx = min(safe_idx, n - 1)

        return max(0, safe_idx)

    def process(self, chunk: str) -> str:
        if not chunk:
            return ""
        self.buf += chunk

        # Decide how much is safe to emit
        safe_idx = self._split_safe_head(self.buf)
        head = self.buf[:safe_idx]
        self.buf = self.buf[safe_idx:]

        if not head:
            return ""

        # Linkify only the finalized head
        return linkify_mp_numbers(head)

    def flush(self) -> str:
        """Flush whatever remains (end of stream): now the buffer is final."""
        if not self.buf:
            return ""
        out = linkify_mp_numbers(self.buf)
        self.buf = ""
        return out
