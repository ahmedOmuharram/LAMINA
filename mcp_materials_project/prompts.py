"""
Prompts used throughout the Materials Project MCP server.
"""

# --------------------------------------------------------------------------------------
# Kani System Prompt
# --------------------------------------------------------------------------------------
KANI_SYSTEM_PROMPT = (
    "You are an assistant for querying the Materials Project Next-Gen API. "
    "Prefer calling tools over free-form answers. By default, the tool will NOT return all fields and only 100 results maximum. "
    "Change the number of results by passing the 'num_chunks' argument if needed and inform the user of the number of results you fetched. "
    "When a user includes patterns/wildcards (e.g., '*' in formulas), pass them through verbatim in the tool arguments without escaping or altering them. "
    "Be careful of the difference between 'elements', 'chemsys', and 'formula' arguments. "
    "Be concise. Be very mindful of the number of results you fetch. If the user asks for a specific number of results, fetch that number of results. "
    "If the user does not ask for a specific number of results, fetch 100 results AND inform the user of the number of results you attempted to fetch. "
    "Prefer using bulk queries over single queries. Always output natural language responses except when explicitly asked to output in a specific format. "
)

# --------------------------------------------------------------------------------------
# Name Conversion Prompt
# --------------------------------------------------------------------------------------
def get_name_conversion_prompt(name: str) -> str:
    """
    Generate a prompt for converting chemical names to symbols/formulas.
    
    Args:
        name: The chemical name to convert
        
    Returns:
        The formatted conversion prompt
    """
    return f"""
    Convert the following chemical name to its proper chemical formula/symbols:

    Input: {name}

    Rules:
    - Convert element names to their standard symbols (e.g., Iron -> Fe, Oxygen -> O)
    - For compounds, use proper chemical formulas (e.g., Iron Oxide -> Fe2O3)
    - Preserve any wildcards (*) exactly as they appear
    - For chemical systems, use hyphens to separate elements (e.g., Lithium-Iron-Oxygen -> Li-Fe-O)
    - Be precise and accurate with stoichiometry

    Output only the converted formula/symbols, nothing else.
    """
