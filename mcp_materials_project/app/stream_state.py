from __future__ import annotations
from typing import Any, Dict, List, Iterable, Optional
import time
import json
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
        
        # Interactive plot link capture
        self.pending_plot_link: Optional[tuple[str, str]] = None  # (link_text, url)

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

        # Capture (and remember) any interactive plot link; do NOT emit it yet
        processed = self._maybe_store_interactive_plot_link(processed)

        return delta_chunk_raw(processed, self.model_name)
    
    def _maybe_store_interactive_plot_link(self, text: str) -> str:
        """Detect and store interactive plot link without modifying the text."""
        import re
        pattern = r'\[([^\]]+)\]\((http://localhost:8000/static/plots/[^)]+\.html)\)'
        m = re.search(pattern, text)
        if m:
            self.pending_plot_link = (m.group(1), m.group(2))
        return text
    
    def emit_pending_plot_link(self) -> Optional[str]:
        """Emit any pending interactive plot link at the end of the stream."""
        if self.pending_plot_link:
            link_text, url = self.pending_plot_link
            self.pending_plot_link = None
            md = f"\n\n**Interactive Plot:** [{link_text}]({url})\n"
            return delta_chunk(md, self.model_name)
        return None

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
                    # Use the tool name from recent_tool_outputs if available
                    if isinstance(tool_output, dict) and "tool_name" in tool_output:
                        tool_name = tool_output["tool_name"]
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

            # Collect any plot link from the tool's pretty-printed output
            if logs_md:
                self._maybe_store_interactive_plot_link(logs_md)

            chunks.append(self.emit_tool_done_panel(tool_name, duration, logs_md))
            self.tools_open = max(0, self.tools_open - 1)
            
            # Check if this tool generated an image and emit it as a separate chunk
            if tool_name in ["plot_binary_phase_diagram", "plot_composition_temperature"]:
                # Emit analysis FIRST before the image chunk clears the metadata
                analysis_chunk = self._emit_analysis_panel()
                print(f"Analysis chunk generated: {analysis_chunk is not None}", flush=True)
                if analysis_chunk:
                    print(f"Adding analysis chunk of length: {len(analysis_chunk)}", flush=True)
                    chunks.append(analysis_chunk)
                
                # Then emit the image chunk (which will clear the metadata)
                image_chunk = self._emit_image_chunk()
                if image_chunk:
                    chunks.append(image_chunk)
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
    
    def _emit_image_chunk(self) -> Optional[str]:
        """Emit an image as a separate chunk outside the tool output."""
        # Try to get the image data from the kani instance
        kani_instance = getattr(self, '_kani_instance', None)
        if not kani_instance:
            return None
            
        image_data = getattr(kani_instance, '_last_image_data', None)
        metadata = getattr(kani_instance, '_last_image_metadata', {})
        
        if not image_data:
            return None
            
        # Get metadata about the image
        system = metadata.get("system", "Unknown")
        description = metadata.get("description", "Generated phase diagram")
        phases = metadata.get("phases", [])
        temp_range = metadata.get("temperature_range_K", "Unknown")
        composition_info = metadata.get("composition_info")
        analysis = metadata.get("analysis", "")
        
        # Create a markdown display with embedded image
        phases_str = ", ".join(phases) if phases else "Unknown"
        
        # Add composition details if available
        comp_details = ""
        if composition_info:
            zn_pct = composition_info.get("zn_percentage", 0)
            al_pct = composition_info.get("al_percentage", 0)
            mole_frac = composition_info.get("target_composition", 0)
            comp_details = f"""- **Composition**: Al{al_pct:.0f}Zn{zn_pct:.0f} ({mole_frac:.3f} mole fraction Zn)
"""
        
        # Build the markdown with image first, then analysis
        markdown_parts = []
        markdown_parts.append(f"**{description}**")
        markdown_parts.append("")
        markdown_parts.append(f"- **System**: {system}")
        markdown_parts.append(f"- **Phases**: {phases_str}")
        markdown_parts.append(f"- **Temperature Range**: {temp_range} K")
        if comp_details:
            markdown_parts.append(comp_details.strip())
        markdown_parts.append("")
        markdown_parts.append(f"![{system} Phase Diagram](data:image/png;base64,{image_data})")
        
        markdown = "\n".join(markdown_parts)
        
        # Clear only the large base64 image data to save memory, but keep metadata for analysis
        if hasattr(kani_instance, '_last_image_data'):
            delattr(kani_instance, '_last_image_data')
        # Keep _last_image_metadata for potential analysis by analyze_last_generated_plot()
        
        # Return as a delta chunk
        from .utils import delta_chunk
        return delta_chunk(markdown, self.model_name)
    
    def _emit_analysis_panel(self) -> Optional[str]:
        """Emit a separate analysis panel as a tool-style panel."""
        # Try to get the analysis from the kani instance
        kani_instance = getattr(self, '_kani_instance', None)
        print(f"Analysis panel: kani_instance exists: {kani_instance is not None}", flush=True)
        if not kani_instance:
            return None
            
        metadata = getattr(kani_instance, '_last_image_metadata', {})
        analysis = metadata.get("analysis", "")
        print(f"Analysis panel: analysis length: {len(analysis) if analysis else 0}", flush=True)
        print(f"Analysis panel: metadata keys: {list(metadata.keys())}", flush=True)
        
        # Debug: Check if kani_instance has the attributes directly
        has_image_data = hasattr(kani_instance, '_last_image_data')
        has_metadata = hasattr(kani_instance, '_last_image_metadata')
        print(f"Analysis panel: kani_instance has _last_image_data: {has_image_data}", flush=True)
        print(f"Analysis panel: kani_instance has _last_image_metadata: {has_metadata}", flush=True)
        
        if has_metadata:
            raw_metadata = getattr(kani_instance, '_last_image_metadata', {})
            print(f"Analysis panel: raw metadata keys: {list(raw_metadata.keys())}", flush=True)
            if "analysis" in raw_metadata:
                print(f"Analysis panel: raw analysis length: {len(raw_metadata['analysis'])}", flush=True)
                print(f"Analysis panel: raw analysis preview: {raw_metadata['analysis'][:200]}...", flush=True)
        
        if not analysis:
            return None
        
        # Create a tool panel for the analysis with magnifying glass icon
        from .utils import tool_panel_general
        analysis_panel = tool_panel_general("🔍 Analysis", analysis)
        print(f"Analysis panel: panel created with length: {len(analysis_panel)}", flush=True)
        
        # Return as a delta chunk
        from .utils import delta_chunk
        return delta_chunk(analysis_panel, self.model_name)