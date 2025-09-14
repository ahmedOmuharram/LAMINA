from __future__ import annotations
from typing import Any, Dict, List, Iterable, Optional
import time
from .linkifier import MPLinkifyBuffer
from .utils import delta_chunk, delta_chunk_raw, tool_panel_done, pretty_print_tool_output
from ..kani_client import MPKani

class StreamState:
    """Holds state for tool tracking and output buffering during streaming."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tool_fifo: List[str] = []
        self.tool_name_by_id: Dict[str, str] = {}
        self.tool_started_at: Dict[str, float] = {}
        self.tools_open: int = 0

        # Answer buffering (only while tools are running)
        self.buffer_chunks: List[str] = []
        self.seen_chunks: set[str] = set()

        # Streaming mp-linkifier buffer (no rewrite)
        self.linkbuf = MPLinkifyBuffer()

    # ------------------------------
    # Tool tracking
    # ------------------------------
    def register_tool_calls(self, msg: Any) -> None:
        """Register incoming tool calls as in-progress."""
        if not getattr(msg, "tool_calls", None):
            return
        now = time.time()
        for tc in msg.tool_calls:
            tc_id = getattr(tc, "id", f"tc-{int(now*1e6)}")
            tool_name = getattr(getattr(tc, "function", None), "name", "tool")
            self.tool_fifo.append(tc_id)
            self.tool_name_by_id[tc_id] = tool_name
            self.tool_started_at[tc_id] = now
            self.tools_open += 1

    def tools_in_progress(self) -> bool:
        return self.tools_open > 0 or len(self.tool_fifo) > 0

    # ------------------------------
    # Emission helpers
    # ------------------------------
    def emit_stream_text(self, text: str) -> Optional[str]:
        """Buffer-aware linkify; only emits finalized, safe content."""
        processed = self.linkbuf.process(text)
        if not processed:
            return None
        return delta_chunk_raw(processed, self.model_name)

    def emit_tool_done_panel(self, tool_name: str, duration: float, logs_md: str) -> str:
        panel_md = tool_panel_done(tool_name, duration, logs_md)
        # panels are full strings; stateless linkify is fine
        return delta_chunk(panel_md, self.model_name)

    def complete_next_tool(self, tool_msg: Any, kani_instance: MPKani) -> Iterable[str]:
        """Close the next queued tool, emit its panel, and flush buffer if needed."""
        chunks: List[str] = []

        if self.tool_fifo:
            tc_id = self.tool_fifo.pop(0)
            tool_name = self.tool_name_by_id.pop(tc_id, getattr(tool_msg, "name", "tool") or "tool")
            started = self.tool_started_at.pop(tc_id, None)
            duration = time.time() - started if started else 0.0

            logs_md = ""
            # Prioritize recent_tool_outputs which contains actual Python objects
            tool_output = None
            latest_out = getattr(kani_instance, "recent_tool_outputs", None)
            if latest_out and len(latest_out) > 0:
                try:
                    tool_output = latest_out[-1]
                except Exception:
                    pass
            
            # Fallback to tool message content if recent_tool_outputs is not available
            if not tool_output and hasattr(tool_msg, 'content') and tool_msg.content:
                try:
                    # Try to parse the content as JSON first
                    import json
                    tool_output = json.loads(tool_msg.content)
                except (json.JSONDecodeError, TypeError):
                    try:
                        # If JSON parsing fails, try to evaluate as Python literal
                        import ast
                        tool_output = ast.literal_eval(tool_msg.content)
                    except (ValueError, SyntaxError):
                        # If all else fails, use the string as is
                        tool_output = tool_msg.content
                except Exception:
                    pass
            
            if tool_output:
                try:
                    logs_md = f"**Tool**: `{tool_name}`\n\n**Output**:\n\n{pretty_print_tool_output(tool_output)}"
                except Exception:
                    pass

            chunks.append(self.emit_tool_done_panel(tool_name, duration, logs_md))
            self.tools_open = max(0, self.tools_open - 1)
        else:
            # No queued tool but got a FUNCTION end — show zero-duration panel anyway
            tool_name = getattr(tool_msg, "name", None) or "tool"
            chunks.append(self.emit_tool_done_panel(tool_name, 0.0, ""))

        # If this was the LAST tool, flush the buffered answer exactly once
        if not self.tools_in_progress() and self.buffer_chunks:
            chunks.append(delta_chunk("".join(self.buffer_chunks), self.model_name))
            self.buffer_chunks.clear()
            self.seen_chunks.clear()

        return chunks

    def flush_buffer_if_any(self) -> Iterable[str]:
        if self.buffer_chunks:
            out = [delta_chunk("".join(self.buffer_chunks), self.model_name)]
            self.buffer_chunks.clear()
            self.seen_chunks.clear()
            return out
        return []

    def close_orphan_tools(self) -> Iterable[str]:
        chunks: List[str] = []
        while self.tool_fifo:
            tc_id = self.tool_fifo.pop(0)
            tool_name = self.tool_name_by_id.pop(tc_id, "tool")
            self.tool_started_at.pop(tc_id, None)
            chunks.append(self.emit_tool_done_panel(tool_name, 0.0, ""))
        return chunks

    def flush_linkbuf(self) -> Optional[str]:
        """Flush any remaining buffered text at end-of-stream."""
        tail = self.linkbuf.flush()
        if not tail:
            return None
        return delta_chunk_raw(tail, self.model_name)
