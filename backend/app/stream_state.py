from __future__ import annotations
from typing import Any, Dict, List, Iterable, Optional
import time
from .utils import (
    delta_chunk_raw, 
    tool_start_event, 
    tool_end_event, 
    image_event, 
    analysis_event,
    linkify_mp_numbers
)
from ..kani_client import MPKani

class StreamState:
    """Holds state for tool tracking and output buffering during streaming."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tool_fifo: List[str] = []
        self.tool_name_by_id: Dict[str, str] = {}
        self.tool_input_by_id: Dict[str, Any] = {}
        self.tool_started_at: Dict[str, float] = {}
        self.tools_open: int = 0
        
        # Track which tools have already been completed to prevent duplicates
        self.completed_tools: set[str] = set()
        
        # Track which tool outputs have been consumed
        self.tool_output_index: int = 0

        # Answer buffering (only while tools are running)
        self.buffer_chunks: List[str] = []
        self.seen_chunks: set[str] = set()
        
        # Track partial text buffer for linkification
        self.text_buffer: str = ""
        
        # Track all emitted text for token counting
        self.all_emitted_text: str = ""

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
            tool_func = getattr(tc, "function", None)
            tool_name = getattr(tool_func, "name", "tool")
            
            # Extract tool input arguments
            tool_input = None
            if tool_func and hasattr(tool_func, "arguments"):
                try:
                    import json
                    tool_input = json.loads(tool_func.arguments)
                except (json.JSONDecodeError, TypeError):
                    tool_input = tool_func.arguments
            
            # Prevent duplicate registration
            if tc_id in self.tool_name_by_id or tc_id in self.tool_fifo:
                print(f"DEBUG: Skipping duplicate tool registration for tc_id={tc_id}, tool={tool_name}", flush=True)
                continue
            
            print(f"DEBUG: Registering tool tc_id={tc_id}, name={tool_name}, input={tool_input}", flush=True)
            self.tool_fifo.append(tc_id)
            self.tool_name_by_id[tc_id] = tool_name
            self.tool_input_by_id[tc_id] = tool_input
            self.tool_started_at[tc_id] = now
            self.tools_open += 1

    def tools_in_progress(self) -> bool:
        return self.tools_open > 0 or len(self.tool_fifo) > 0

    # ------------------------------
    # Emission helpers
    # ------------------------------
    def emit_stream_text(self, text: str) -> Optional[str]:
        """Emit text with MP number linkification."""
        # Hide tool-call tags if they appear in raw text
        if "<|functions." in text or "<|function." in text:
            # swallow the markup; do not emit
            return None

        # Track the original text for token counting (before linkification)
        self.all_emitted_text += text
        
        # Apply linkification and emit
        processed = linkify_mp_numbers(text)
        return delta_chunk_raw(processed, self.model_name)
    
    def emit_tool_start(self, tool_name: str, tool_id: str, tool_input: Any = None) -> str:
        """Emit a tool start event."""
        return tool_start_event(tool_name, tool_id, self.model_name, tool_input)
    
    def emit_tool_end(self, tool_name: str, tool_id: str, duration: float, output: Any, tool_input: Any = None) -> str:
        """Emit a tool end event."""
        return tool_end_event(tool_name, tool_id, duration, output, self.model_name, tool_input)

    def complete_next_tool(self, tool_msg: Any, kani_instance: MPKani) -> Iterable[str]:
        """Close the next queued tool, emit its completion event, and flush buffer if needed."""
        chunks: List[str] = []

        if self.tool_fifo:
            tc_id = self.tool_fifo.pop(0)
            
            print(f"DEBUG: Processing tool completion for tc_id={tc_id}, fifo length={len(self.tool_fifo)}", flush=True)
            
            # Check if we've already completed this tool (prevents duplicates)
            if tc_id in self.completed_tools:
                print(f"DEBUG: Skipping duplicate tool completion for tc_id={tc_id}", flush=True)
                return chunks
            
            # Mark this tool as completed
            self.completed_tools.add(tc_id)
            
            tool_name = self.tool_name_by_id.pop(tc_id, getattr(tool_msg, "name", "tool") or "tool")
            tool_input = self.tool_input_by_id.pop(tc_id, None)
            started = self.tool_started_at.pop(tc_id, None)
            duration = time.time() - started if started else 0.0
            
            print(f"DEBUG: Tool name={tool_name}, duration={duration:.2f}s", flush=True)

            # Prioritize recent_tool_outputs which contains actual Python objects
            tool_output = None
            latest_out = getattr(kani_instance, "recent_tool_outputs", None)
            if latest_out and self.tool_output_index < len(latest_out):
                try:
                    # Use the next unprocessed tool output
                    tool_output = latest_out[self.tool_output_index]
                    self.tool_output_index += 1
                    print(f"DEBUG: Consumed tool output #{self.tool_output_index-1}, {len(latest_out) - self.tool_output_index} remaining", flush=True)
                    # Use the tool name from recent_tool_outputs if available
                    if isinstance(tool_output, dict) and "tool_name" in tool_output:
                        tool_name = tool_output["tool_name"]
                        # Also check if this exact tool output was already processed
                        tool_key = f"{tool_name}_{duration:.2f}"
                        if tool_key in self.completed_tools:
                            print(f"DEBUG: Skipping duplicate tool output for {tool_key}", flush=True)
                            return chunks
                        self.completed_tools.add(tool_key)
                except Exception as e:
                    print(f"DEBUG: Error processing recent_tool_outputs: {e}", flush=True)
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

            # Emit tool completion event with output and input
            chunks.append(self.emit_tool_end(tool_name, tc_id, duration, tool_output, tool_input))
            self.tools_open = max(0, self.tools_open - 1)
            
            # Check if this tool generated an image and emit it as a separate event
            if tool_name in ["plot_binary_phase_diagram", "plot_composition_temperature", "plot_ternary_phase_diagram"]:
                # Emit analysis FIRST before the image chunk clears the metadata
                analysis_chunk = self._emit_analysis_event()
                print(f"Analysis event generated: {analysis_chunk is not None}", flush=True)
                if analysis_chunk:
                    print(f"Adding analysis event of length: {len(analysis_chunk)}", flush=True)
                    chunks.append(analysis_chunk)
                
                # Then emit the image event (which will clear the metadata)
                image_chunk = self._emit_image_event()
                if image_chunk:
                    chunks.append(image_chunk)
        else:
            # No queued tool but got a FUNCTION end — show zero-duration event anyway
            tool_name = getattr(tool_msg, "name", None) or "tool"
            tc_id = f"tc-orphan-{int(time.time()*1e6)}"
            chunks.append(self.emit_tool_end(tool_name, tc_id, 0.0, None))

        # If this was the LAST tool, flush the buffered answer exactly once
        if not self.tools_in_progress() and self.buffer_chunks:
            buffered_text = "".join(self.buffer_chunks)
            chunks.append(delta_chunk_raw(linkify_mp_numbers(buffered_text), self.model_name))
            self.buffer_chunks.clear()
            self.seen_chunks.clear()
            # Clear completed tools tracking for next request
            self.completed_tools.clear()
            self.tool_output_index = 0

        return chunks

    def flush_buffer_if_any(self) -> Iterable[str]:
        if self.buffer_chunks:
            buffered_text = "".join(self.buffer_chunks)
            # Track buffered text for token counting
            self.all_emitted_text += buffered_text
            out = [delta_chunk_raw(linkify_mp_numbers(buffered_text), self.model_name)]
            self.buffer_chunks.clear()
            self.seen_chunks.clear()
            return out
        return []

    def close_orphan_tools(self) -> Iterable[str]:
        chunks: List[str] = []
        while self.tool_fifo:
            tc_id = self.tool_fifo.pop(0)
            # Skip if already completed (prevents duplicates)
            if tc_id not in self.completed_tools:
                self.completed_tools.add(tc_id)
                tool_name = self.tool_name_by_id.pop(tc_id, "tool")
                self.tool_started_at.pop(tc_id, None)
                chunks.append(self.emit_tool_end(tool_name, tc_id, 0.0, None))
        # Clear completed tools tracking after orphans are closed
        self.completed_tools.clear()
        self.tool_output_index = 0
        return chunks
    
    def _emit_image_event(self) -> Optional[str]:
        """Emit an image as a structured event."""
        # Try to get the image URL from the kani instance
        kani_instance = getattr(self, '_kani_instance', None)
        if not kani_instance:
            return None
            
        # Look for image URL on the kani instance (which includes CalPhadHandler)
        image_url = getattr(kani_instance, '_last_image_url', None)
        metadata = getattr(kani_instance, '_last_image_metadata', {})
        
        # Debug: print what we found
        print(f"Image event: Found image_url: {bool(image_url)}", flush=True)
        print(f"Image event: Found metadata: {bool(metadata)}", flush=True)
        if image_url:
            print(f"Image event: Image URL: {image_url}", flush=True)
        if metadata:
            print(f"Image event: Metadata keys: {list(metadata.keys())}", flush=True)
        
        if not image_url:
            return None
        
        # Clear the image URL to save memory, but keep metadata for analysis
        if hasattr(kani_instance, '_last_image_url'):
            delattr(kani_instance, '_last_image_url')
        # Keep _last_image_metadata for potential analysis by analyze_last_generated_plot()
        
        # Return as a structured image event
        result = image_event(image_url, metadata, self.model_name)
        print(f"Image event: Event created, length: {len(result)}", flush=True)
        return result
    
    def _emit_analysis_event(self) -> Optional[str]:
        """Emit a separate analysis event."""
        # Try to get the analysis from the kani instance
        kani_instance = getattr(self, '_kani_instance', None)
        print(f"Analysis event: kani_instance exists: {kani_instance is not None}", flush=True)
        if not kani_instance:
            return None
            
        metadata = getattr(kani_instance, '_last_image_metadata', {})
        analysis_content = metadata.get("analysis", "")
        print(f"Analysis event: analysis length: {len(analysis_content) if analysis_content else 0}", flush=True)
        print(f"Analysis event: metadata keys: {list(metadata.keys())}", flush=True)
        
        # Debug: Check if kani_instance has the attributes directly
        has_image_url = hasattr(kani_instance, '_last_image_url')
        has_metadata = hasattr(kani_instance, '_last_image_metadata')
        print(f"Analysis event: kani_instance has _last_image_url: {has_image_url}", flush=True)
        print(f"Analysis event: kani_instance has _last_image_metadata: {has_metadata}", flush=True)
        
        if has_metadata:
            raw_metadata = getattr(kani_instance, '_last_image_metadata', {})
            print(f"Analysis event: raw metadata keys: {list(raw_metadata.keys())}", flush=True)
            if "analysis" in raw_metadata:
                print(f"Analysis event: raw analysis length: {len(raw_metadata['analysis'])}", flush=True)
                print(f"Analysis event: raw analysis preview: {raw_metadata['analysis'][:200]}...", flush=True)
        
        if not analysis_content:
            return None
        
        # Return as a structured analysis event
        result = analysis_event(analysis_content, self.model_name)
        print(f"Analysis event: Event created with length: {len(result)}", flush=True)
        return result