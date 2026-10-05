"""
Runner utilities for LAMINA Evaluation Framework.

Contains dataclasses for structured records and the main runner interface.
"""

import asyncio
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Add backend to path for imports
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation."""
    tool_name: str
    family: str  # e.g. "thermodynamics", "electrochemistry"
    library: Optional[str]  # e.g. "COST507", "MC_AL", "CHGNet", "MP"
    arguments: Dict[str, Any]
    result: Any  # parsed result (dict/float/etc.)
    raw_result: Any  # raw JSON/text from the tool
    error: Optional[str]
    latency_ms: float


@dataclass
class PredictionRecord:
    """Record of a model prediction for a single claim."""
    claim_id: str
    claim_text: str
    gold_label: int  # -2..2
    predicted_label: int  # -2..2
    reasoning: str  # model's explanation
    raw_response: str  # full raw LLM response
    mapping_id: Optional[str]  # for H3 ambiguous mappings
    mapping_description: Optional[str]
    tool_calls: List[ToolCallRecord]
    router_type: str  # "no-policy" / "policy" / "oracle"
    tools_enabled: bool
    multi_source: Optional[bool]  # for H4
    model_name: str
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


def _extract_tool_family(tool_name: str) -> str:
    """Extract tool family from tool name."""
    family_map = {
        # CALPHAD
        "plot_composition_temperature": "CALPHAD",
        "plot_binary_phase_diagram": "CALPHAD",
        "plot_ternary_phase_diagram": "CALPHAD",
        "calculate_phase_fractions_vs_temperature": "CALPHAD",
        "calculate_equilibrium_at_point": "CALPHAD",
        "verify_phase_formation_across_composition": "CALPHAD",
        "sweep_microstructure_claim_over_region": "CALPHAD",
        "fact_check_microstructure_claim": "CALPHAD",
        "equilibrium_phase_fractions": "CALPHAD",
        "binary_phase_diagram": "CALPHAD",
        "ternary_phase_diagram": "CALPHAD",
        "liquidus_temperature": "CALPHAD",
        "solidus_temperature": "CALPHAD",
        "scheil_solidification": "CALPHAD",
        # Electrochemistry
        "calculate_voltage_from_formation_energy": "Electrochemistry",
        "compare_electrode_materials": "Electrochemistry",
        "analyze_anode_viability": "Electrochemistry",
        "analyze_lithiation_mechanism": "Electrochemistry",
        "estimate_ion_hopping_barrier": "Electrochemistry",
        "open_circuit_potential": "Electrochemistry",
        "lithiation_voltage": "Electrochemistry",
        "battery_capacity": "Electrochemistry",
        # Materials Project
        "get_elastic_properties": "Materials Project",
        "analyze_doping_effect": "Materials Project",
        "bulk_modulus": "Materials Project",
        "elastic_modulus": "Materials Project",
        "shear_modulus": "Materials Project",
        # Semiconductors
        "analyze_phase_transition_structures": "Semiconductors",
        "predict_defect_site_preference": "Semiconductors",
        "analyze_doping_site_preference": "Semiconductors",
        # Alloys
        "estimate_surface_diffusion_barrier": "Alloys",
        "assess_phase_strength_and_stiffness_claims": "Alloys",
        "diffusion_barrier": "Alloys",
        # Superconductors
        "analyze_cuprate_octahedral_stability": "Superconductors",
        # Magnets
        "assess_magnet_strength_with_doping": "Magnets",
        "assess_doping_effect_on_saturation_magnetization": "Magnets",
        "compare_dopants_for_saturation_magnetization": "Magnets",
        "saturation_magnetization": "Magnets",
        "magnetic_moment": "Magnets",
        "coercivity": "Magnets",
        # Solutes
        "analyze_solute_lattice_effect": "Solutes",
        "compare_solute_lattice_effects": "Solutes",
    }
    
    # Check direct mapping
    if tool_name in family_map:
        return family_map[tool_name]
    
    # Check partial matches
    for key, family in family_map.items():
        if key in tool_name.lower():
            return family
    
    return "unknown"


def _extract_tool_library(tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
    """Extract library/database used by tool."""
    # Check for explicit library in arguments
    if "database" in arguments:
        return arguments["database"]
    if "tdb" in arguments or "tdb_file" in arguments:
        tdb = arguments.get("tdb") or arguments.get("tdb_file", "")
        if "COST507" in tdb:
            return "COST507"
        if "mc_al" in tdb.lower():
            return "MC_AL"
        return tdb
    if "source" in arguments:
        return arguments["source"]
    
    # Infer from tool name
    if "mp_" in tool_name.lower() or "materials_project" in tool_name.lower():
        return "MP"
    if "chgnet" in tool_name.lower():
        return "CHGNet"
    
    return None


async def run_kani_on_claim(
    claim: Dict[str, Any],
    model_name: str,
    router_type: str = "policy",
    tools_enabled: bool = True,
    multi_source: bool = True,
    mapping_override: Optional[Dict[str, Any]] = None,
    oracle_tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.0,
    seed: int = 42,
    max_tool_calls: Optional[int] = None,  # H4: limit tool call iterations (None = unlimited)
) -> PredictionRecord:
    """
    Run LAMINA/Kani on a single claim and return structured prediction.
    
    Args:
        claim: Claim dictionary with id, text, gold_label
        model_name: LLM model to use (e.g., "gpt-4o-mini")
        router_type: Router configuration ("no-policy", "policy", "oracle")
        tools_enabled: Whether computational tools are available
        multi_source: Whether to allow multiple data sources
        mapping_override: For ambiguous claims, override numeric parameters
        oracle_tools: Pre-specified tool calls for oracle router (list of {tool, family, arguments})
        temperature: LLM temperature (default 0 for reproducibility)
        seed: Random seed
        max_tool_calls: H4 - limit the number of tool call iterations (None = unlimited)
        
    Returns:
        PredictionRecord with full prediction details
    """
    import json as json_module
    from kani import ChatRole
    from backend.kani_client import MPKani
    from backend.prompts import KANI_SYSTEM_PROMPT
    
    claim_id = claim["id"]
    # Use demystified_text if provided in mapping_override (for H3 ambiguity testing)
    if mapping_override and mapping_override.get("demystified_text"):
        claim_text = mapping_override["demystified_text"]
    else:
        claim_text = claim["text"]
    gold_label = claim["gold_label"]
    
    tool_calls: List[ToolCallRecord] = []
    response_parts: List[str] = []
    oracle_results: List[str] = []
    
    start_time = time.time()
    
    try:
        # Determine enabled functions based on tools_enabled flag
        enabled_functions = None if tools_enabled else set()  # Empty set = no tools
        
        # Choose system prompt based on router_type **and** whether tools are enabled.
        # - policy + tools_enabled=True  → full KANI_SYSTEM_PROMPT (production routing behavior)
        # - policy + tools_enabled=False → lightweight evaluation prompt (no routing pressure)
        # - any other router_type        → lightweight evaluation prompt
        if router_type == "policy" and tools_enabled:
            system_prompt = KANI_SYSTEM_PROMPT
        else:
            # Barebones evaluation prompt with no strong tool-calling requirements.
            system_prompt = (
                "You are an expert materials science assistant. "
                "Your goal is to carefully evaluate the feasibility of materials science claims. "
                "If tools are available you may use them, but if the user asks you not to use tools "
                "you must obey and answer directly based on your prior knowledge."
            )
        
        # Create Kani instance with temperature=0, top_p=1 for reproducibility
        kani_instance = MPKani(
            model=model_name,
            system_prompt=system_prompt,
            enabled_functions=enabled_functions,
            temperature=temperature,
            top_p=1.0,  # Always use top_p=1 for reproducibility
            seed=seed,
        )
        
        # If oracle_tools provided, execute them first and collect results
        if oracle_tools and router_type == "oracle":
            for oracle_tool in oracle_tools:
                tool_name = oracle_tool.get("tool")
                tool_args = oracle_tool.get("arguments", {})
                tool_family = oracle_tool.get("family", "unknown")
                
                tool_start = time.time()
                tool_result = None
                tool_error = None
                raw_result = None
                
                try:
                    # Get the tool function from the Kani instance
                    tool_func = getattr(kani_instance, tool_name, None)
                    if tool_func is not None:
                        # Call the tool with the specified arguments
                        result = await tool_func(**tool_args) if asyncio.iscoroutinefunction(tool_func) else tool_func(**tool_args)
                        raw_result = str(result)
                        
                        # Try to parse as JSON if it's a string
                        if isinstance(result, str):
                            try:
                                tool_result = json_module.loads(result)
                            except (json_module.JSONDecodeError, TypeError):
                                tool_result = result
                        else:
                            tool_result = result
                        
                        # Format result for including in the query
                        if isinstance(tool_result, dict):
                            oracle_results.append(f"Tool '{tool_name}' with args {tool_args}:\n{json_module.dumps(tool_result, indent=2, default=str)}")
                        else:
                            oracle_results.append(f"Tool '{tool_name}' with args {tool_args}:\n{tool_result}")
                    else:
                        tool_error = f"Tool '{tool_name}' not found on Kani instance"
                        oracle_results.append(f"Tool '{tool_name}': ERROR - {tool_error}")
                except Exception as e:
                    tool_error = str(e)
                    oracle_results.append(f"Tool '{tool_name}': ERROR - {tool_error}")
                
                tool_latency = (time.time() - tool_start) * 1000
                
                tool_calls.append(ToolCallRecord(
                    tool_name=tool_name,
                    family=tool_family,
                    library=_extract_tool_library(tool_name, tool_args),
                    arguments=tool_args,
                    result=tool_result,
                    raw_result=raw_result,
                    error=tool_error,
                    latency_ms=tool_latency,
                ))
        
        # Build the query with context - use a specific output format for reliable parsing
        # Different prompts for tools-enabled vs no-tools
        if tools_enabled:
            query = f"""Evaluate the following materials science claim and provide a feasibility rating.

                        Claim: {claim_text}

                        Rate the claim's feasibility on a scale from -2 to +2:
                        - -2: Extremely infeasible (clearly contradicted by evidence)
                        - -1: Likely infeasible (evidence suggests this is unlikely)
                        - 0: Undecidable (insufficient evidence or ambiguous)
                        - +1: Likely feasible (evidence suggests this is plausible)
                        - +2: Extremely feasible (strongly supported by evidence)

                        Use computational tools to verify the claim if appropriate. Provide your reasoning, then end your response with exactly this format on its own line:
                        VERDICT: <integer from -2 to 2>"""
            
            # H4: Add single-call instructions if limited to 1 tool call
            if max_tool_calls == 1:
                query += """

                        IMPORTANT: You are limited to exactly ONE successful tool call for this evaluation.
                        Choose your tool wisely - pick the single most informative tool that will give you the best evidence to evaluate this claim.
                        If your tool call fails or returns an error, you will get one retry, but a successful call ends your tool usage.
                        After your tool call succeeds, you must provide your verdict based on that result."""
        else:
            # No tools - rely on training knowledge only (no external calls)
            query = f"""Evaluate the following materials science claim and provide a feasibility rating **without calling or assuming access to any external tools, APIs, databases, or new simulations**. You must rely only on your prior knowledge of materials science, thermodynamics, and physics.

                        Claim: {claim_text}

                        Rate the claim's feasibility on a scale from -2 to +2:
                        - -2: Extremely infeasible (clearly contradicted by known science)
                        - -1: Likely infeasible (scientific evidence suggests this is unlikely)
                        - 0: Undecidable (insufficient information or genuinely ambiguous)
                        - +1: Likely feasible (scientific principles suggest this is plausible)
                        - +2: Extremely feasible (strongly supported by scientific knowledge)

                        Provide your reasoning qualitatively based on materials science principles, phase diagrams you have learned about, thermodynamics, or other relevant prior knowledge. **Do not say you will “run a calculation”, “use CALPHAD tools”, or call any external software.** Instead, imagine the analysis and directly explain your conclusion. Then end your response with exactly this format on its own line:
                        VERDICT: <integer from -2 to 2>"""

        # Note: demystified_text is already substituted into claim_text above,
        # so no need to append mapping_override params to the query
        
        # For H3 (ambiguity testing), add explicit feasibility instructions
        if mapping_override:
            query += """

                IMPORTANT: Be decisive about feasibility based on the specific values in this claim.
                - If the claim specifies a range (e.g., "between X and Y"), check if the phenomenon occurs ANYWHERE within that range. If yes → feasible (+1 or +2). If no → infeasible (-1 or -2).
                - If the claim specifies a threshold (e.g., "more than X%"), check if the computed value meets that threshold. If yes → feasible. If no → infeasible.
                - The claim contains explicit numeric criteria. Your job is to verify whether those criteria are met, not to hedge."""
                        
        # If we have oracle results, include them in the query
        if oracle_results:
            query += "\n\n--- Pre-computed Tool Results ---\n"
            query += "\n\n".join(oracle_results)
            query += "\n\nBased on the above tool results, provide your assessment and end with VERDICT: <rating>."
        
        # Run the full round with streaming, passing hyperparams for reproducibility
        hyperparams = kani_instance.get_hyperparams()
        # For GPT-5.1, request high reasoning effort and let the API use its default temperature.
        # Passing temperature=0.0 causes a 400 for this model, so we drop it entirely.
        if model_name == "gpt-5.1":
            hyperparams["reasoning_effort"] = "high"
            hyperparams.pop("temperature", None)
        stream_iterator = kani_instance.full_round_stream(query, **hyperparams)
        
        # Track tool calls for H4 max_tool_calls limit
        completed_tool_calls = 0
        hit_tool_limit = False
        
        async for stream in stream_iterator:
            role = getattr(stream, "role", None)
            
            if role == ChatRole.FUNCTION:
                # Tool call result - get the message and update existing record
                tool_start = time.time()
                tool_msg = await stream.message()
                tool_latency = (time.time() - tool_start) * 1000
                
                # Extract tool call info from the message
                tool_name = getattr(tool_msg, "name", None) or "unknown"
                tool_content = getattr(tool_msg, "content", None)
                
                # Try to parse the tool result
                tool_result = tool_content
                tool_error = None
                if isinstance(tool_content, str):
                    try:
                        import json
                        tool_result = json.loads(tool_content)
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                # Check if it's an error
                if isinstance(tool_result, dict) and "error" in tool_result:
                    tool_error = str(tool_result.get("error"))
                
                # Find and update the existing tool call record (from assistant message)
                existing_record = None
                for tc in tool_calls:
                    if tc.tool_name == tool_name and tc.result is None:
                        existing_record = tc
                        break
                
                if existing_record:
                    # Update the existing record with the result
                    existing_record.result = tool_result
                    existing_record.raw_result = tool_content
                    existing_record.error = tool_error
                    existing_record.latency_ms = tool_latency
                else:
                    # No existing record - create new one (shouldn't happen normally)
                    tool_calls.append(ToolCallRecord(
                        tool_name=tool_name,
                        family=_extract_tool_family(tool_name),
                        library=_extract_tool_library(tool_name, {}),
                        arguments={},
                        result=tool_result,
                        raw_result=tool_content,
                        error=tool_error,
                        latency_ms=tool_latency,
                    ))
                
                # H4: Track completed tool calls and check limit
                # Only count SUCCESSFUL tool calls against the limit (errors get a retry)
                if tool_error is None:
                    completed_tool_calls += 1
                    if max_tool_calls is not None and completed_tool_calls >= max_tool_calls:
                        hit_tool_limit = True
                        # Disable tools so model can't make more calls on next iteration
                        kani_instance.functions = {}
                
                continue
            
            # For assistant messages, collect the response text
            async for token in stream:
                if token is not None:
                    response_parts.append(token)
            
            # Get the final message
            msg = await stream.message()
            
            # Check for tool calls in the message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    # Record tool call metadata
                    tc_name = getattr(tc.function, "name", "unknown") if hasattr(tc, "function") else "unknown"
                    tc_args = {}
                    if hasattr(tc, "function") and hasattr(tc.function, "arguments"):
                        try:
                            import json
                            tc_args = json.loads(tc.function.arguments)
                        except:
                            tc_args = {"raw": tc.function.arguments}
                    
                    # Check if we already recorded this tool call
                    already_recorded = any(t.tool_name == tc_name for t in tool_calls)
                    if not already_recorded:
                        tool_calls.append(ToolCallRecord(
                            tool_name=tc_name,
                            family=_extract_tool_family(tc_name),
                            library=_extract_tool_library(tc_name, tc_args),
                            arguments=tc_args,
                            result=None,  # Result comes in FUNCTION message
                            raw_result=None,
                            error=None,
                            latency_ms=0,
                        ))
        
        raw_response = "".join(response_parts)
        
        # Extract predicted label from response
        predicted_label, reasoning = _parse_response(raw_response)
        
    except Exception as e:
        import traceback
        raw_response = f"Error: {str(e)}\n{traceback.format_exc()}"
        reasoning = f"Evaluation failed: {str(e)}"
        predicted_label = 0
        hit_tool_limit = False  # Ensure defined for error case
        completed_tool_calls = 0  # Ensure defined for error case
    
    total_time = (time.time() - start_time) * 1000
    
    return PredictionRecord(
        claim_id=claim_id,
        claim_text=claim_text,
        gold_label=gold_label,
        predicted_label=predicted_label,
        reasoning=reasoning,
        raw_response=raw_response,
        mapping_id=mapping_override.get("mapping_id") if mapping_override else None,
        mapping_description=mapping_override.get("description") if mapping_override else None,
        tool_calls=tool_calls,
        router_type=router_type,
        tools_enabled=tools_enabled,
        multi_source=multi_source,
        model_name=model_name,
        extra_metadata={
            "temperature": temperature,
            "seed": seed,
            "total_latency_ms": total_time,
            "mapping_override": mapping_override,
            "original_claim_text": claim["text"] if mapping_override else None,
            "max_tool_calls": max_tool_calls,
            "successful_tool_calls": completed_tool_calls if tools_enabled else 0,
            "hit_tool_limit": hit_tool_limit if tools_enabled else None,
        },
    )


def _parse_response(response: str) -> tuple[int, str]:
    """
    Parse LLM response to extract predicted label and reasoning.
    
    Looks for "VERDICT: <number>" format first (reliable), then falls back to heuristics.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Tuple of (predicted_label, reasoning)
    """
    import re
    
    predicted_label = None  # Use None to track if we found a valid rating
    
    # First, look for our specific format: "VERDICT: <number>" (with optional markdown bold)
    # Handles: VERDICT: -2, **VERDICT**: -2, **VERDICT: -2**, VERDICT: **-2**, etc.
    verdict_match = re.search(r"\*{0,2}VERDICT\*{0,2}:\s*\*{0,2}([+-]?\d)\*{0,2}", response, re.IGNORECASE)
    if verdict_match:
        try:
            label = int(verdict_match.group(1))
            if -2 <= label <= 2:
                predicted_label = label
        except ValueError:
            pass
    
    # Fallback patterns if VERDICT: format not found
    if predicted_label is None:
        patterns = [
            # "rating is -2" or "rating: -2" or "rating of -2"
            r"(?:rating|score|feasibility)[\s:]+(?:is\s+|of\s+|=\s*)?([+-]?\d)",
            # "is -2" at end of sentence (common pattern)
            r"\bis\s+([+-]?\d)(?:\s*[,.]|\s+indicating|\s*$)",
            # "**-2**" or "**+2**" (markdown bold)
            r"\*\*([+-]?\d)\*\*",
            # "-2 (extremely" or "+2 (extremely" 
            r"([+-]?\d)\s*\((?:extremely|likely|undecidable)",
            # "conclude with -2" or "final rating -2"
            r"(?:conclude|final)\s+(?:with\s+)?(?:a\s+)?(?:rating\s+(?:of\s+)?)?([+-]?\d)",
            # "I rate this -2" or "I would rate this -2"
            r"(?:I\s+)?(?:would\s+)?rate\s+(?:this\s+)?(?:claim\s+)?(?:as\s+|a\s+)?([+-]?\d)",
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, response, re.IGNORECASE))
            if matches:
                # Take the last match (usually the final verdict)
                match = matches[-1]
                try:
                    label = int(match.group(1))
                    if -2 <= label <= 2:
                        predicted_label = label
                        break
                except ValueError:
                    continue
    
    # Default to 0 if nothing found
    if predicted_label is None:
        predicted_label = 0
    
    # Extract reasoning (everything before VERDICT line, or full response)
    reasoning = response.strip()
    # Remove the VERDICT line from reasoning if present (handles markdown variants)
    reasoning = re.sub(r"\n*\*{0,2}VERDICT\*{0,2}:\s*\*{0,2}[+-]?\d\*{0,2}\s*$", "", reasoning, flags=re.IGNORECASE).strip()
    if len(reasoning) > 1000:
        reasoning = reasoning[:1000] + "..."
    
    return predicted_label, reasoning


def run_kani_sync(
    claim: Dict[str, Any],
    model_name: str,
    router_type: str = "policy",
    tools_enabled: bool = True,
    multi_source: bool = True,
    mapping_override: Optional[Dict[str, Any]] = None,
    oracle_tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.0,
    seed: int = 42,
    max_tool_calls: Optional[int] = None,
) -> PredictionRecord:
    """
    Synchronous wrapper for run_kani_on_claim.
    """
    return asyncio.run(run_kani_on_claim(
        claim=claim,
        model_name=model_name,
        router_type=router_type,
        tools_enabled=tools_enabled,
        multi_source=multi_source,
        mapping_override=mapping_override,
        oracle_tools=oracle_tools,
        temperature=temperature,
        seed=seed,
        max_tool_calls=max_tool_calls,
    ))


# Simpler mock runner for testing without LAMINA backend
def mock_run_claim(
    claim: Dict[str, Any],
    model_name: str,
    router_type: str = "policy",
    tools_enabled: bool = True,
    multi_source: bool = True,
    mapping_override: Optional[Dict[str, Any]] = None,
    temperature: float = 0.0,
    seed: int = 42,
    max_tool_calls: Optional[int] = None,
) -> PredictionRecord:
    """
    Mock runner for testing the evaluation framework.
    Returns random predictions for testing purposes.
    """
    import random
    
    random.seed(seed + hash(claim["id"]))
    
    # Use demystified_text if provided in mapping_override
    if mapping_override and mapping_override.get("demystified_text"):
        claim_text = mapping_override["demystified_text"]
    else:
        claim_text = claim["text"]
    
    # Generate mock prediction
    predicted_label = random.randint(-2, 2)
    
    # Generate mock tool calls based on tools_enabled and max_tool_calls
    tool_calls = []
    if tools_enabled:
        mock_tools = [
            ("plot_binary_phase_diagram", "CALPHAD", "COST507"),
            ("calculate_equilibrium_at_point", "CALPHAD", "MC_AL"),
            ("get_elastic_properties", "Materials Project", "MP"),
            ("analyze_anode_viability", "Electrochemistry", "MP"),
            ("assess_magnet_strength_with_doping", "Magnets", "CHGNet"),
            ("analyze_solute_lattice_effect", "Solutes", "CHGNet"),
            ("estimate_surface_diffusion_barrier", "Alloys", "CHGNet"),
        ]
        num_calls = random.randint(1, 3)
        if max_tool_calls is not None:
            num_calls = min(num_calls, max_tool_calls)
        for i in range(num_calls):
            tool_name, family, library = mock_tools[i % len(mock_tools)]
            tool_calls.append(ToolCallRecord(
                tool_name=tool_name,
                family=family,
                library=library,
                arguments={"mock": True},
                result={"mock_result": random.random()},
                raw_result="mock raw result",
                error=None,
                latency_ms=random.uniform(100, 500),
            ))
    
    return PredictionRecord(
        claim_id=claim["id"],
        claim_text=claim_text,
        gold_label=claim["gold_label"],
        predicted_label=predicted_label,
        reasoning=f"Mock reasoning for claim {claim['id']}",
        raw_response=f"Mock response with rating: {predicted_label}",
        mapping_id=mapping_override.get("mapping_id") if mapping_override else None,
        mapping_description=mapping_override.get("description") if mapping_override else None,
        tool_calls=tool_calls,
        router_type=router_type,
        tools_enabled=tools_enabled,
        multi_source=multi_source,
        model_name=model_name,
        extra_metadata={
            "temperature": temperature,
            "seed": seed,
            "mock": True,
            "original_claim_text": claim["text"],
            "max_tool_calls": max_tool_calls,
        },
    )

