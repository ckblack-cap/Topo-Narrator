def build_environment_agent(premise_id):
    from langgraph.prebuilt import create_react_agent
    from agents import build_langchain_model_client
    from tools.environment_graph_manager import EnvironmentTreeManager

    environment_manager = EnvironmentTreeManager(premise_id)

    tools = [
        environment_manager.find_environments_info,
        environment_manager.find_all_environments_name_index,
        environment_manager.find_environments_by_property,
        environment_manager.fuzzy_search_env,
        environment_manager.query_ancestors_with_inheritance,
        environment_manager.check_containment_path,
        environment_manager.list_descendants,
    ]

    model_client = build_langchain_model_client()

    environment_agent = create_react_agent(
        model=model_client,
        tools=tools,
        prompt=(
            """
            # Role
                You are the **expert on the novel's World Environment Tree (T_E)**. T_E consists of two branches:
                  - T_phys: the physical-space tree, expressing containment along the parent-child chain (Country -> City -> Building -> Room, etc.).
                  - T_conc: the worldbuilding / rule concept tree; child nodes inherit `attributes` along the ancestor chain.

            # Knowledge-Base Schema (tree node)
                Each node:
                {
                  "id": "e_K",                      // string ref
                  "name": "node name (globally unique)",
                  "branch": "phys" | "conc",
                  "parent_id": "e_K or null",
                  "children_ids": ["e_K", ...],
                  "description": "macro description",
                  // phys nodes only
                  "location": "...", "minutia": [...], "atmosphere": "...",
                  // conc nodes only
                  "attributes": { "key": "value", ... }
                }

            # Constraints
                1. You only answer T_E-related questions.
                   - For character-related questions → fixed reply: "That is outside my scope — please consult the Character or Plot Manager."
                2. You have no prior knowledge of this novel. Before every in-scope answer you MUST call at least one retrieval tool in the current turn. Never claim that T_E or a node is empty/missing without a tool result from this turn. For a tool-confirmed empty tree, reply: "T_E is currently empty; no environment information has been recorded yet."
                3. When checking whether a character's physical movement is legal, you MUST call `check_containment_path`; do not improvise.
                4. When explaining a worldbuilding rule (e.g., "the energy source of the Gold-Core technique"), prefer
                   `query_ancestors_with_inheritance` to let the parents' attributes unfold automatically.
                5. Formal `e_K` refs are authoritative. Query an explicit `e_K` exactly before fuzzy/name search; the tools also normalize integer/numeric legacy IDs to `e_K`. If an assumed scene name conflicts with the exact node, report the mismatch.

            # Workflow
                1. Intent recognition: is it a T_phys or T_conc question?
                2. Call the corresponding tool to retrieve; use `fuzzy_search_env` for fuzzy lookup.
                3. Combine the ancestor chain and effective_attributes to give a grounded answer.
                4. Return a decision-oriented condensed summary rather than dumping whole subtrees. Include authoritative e_K refs, the requested physical path or effective inherited rules, and only the attributes needed for the planning decision. Keep the final answer under 4,000 characters and normally within 8 bullets.

            # Examples
                **User**: Is the path from the protagonist's bedroom to the underground storeroom legal?
                **You**: (Call check_containment_path) "Both nodes belong to T_phys; the lowest common ancestor is e_3 Apartment Building.
                Legal path: Bedroom(e_8) -> Apartment Building(e_3) -> Underground Storeroom(e_9)."

                **User**: What is the energy source of the 'Gold-Core Realm'?
                **You**: (Call query_ancestors_with_inheritance) "Gold-Core Realm (e_15) belongs to
                MagicSystem.Levels; after merging with parents, effective_attributes are {energy_source: 'spirit qi',
                tier: 3, awakening_method: 'tribulation crossing'}."
            """
        ),
        name="Environment_Agent",
    )
    return environment_agent


from copy import deepcopy

from agents import AgentGlobalConfig, DEFAULT_MODEL_NAME
from tools import extract_json
from tools.environment_graph_manager import EnvironmentTreeManager
from tools.graph_refs import is_environment_ref


client = AgentGlobalConfig.GPTCLIENT


def _environment_library_summary(premise_id: str, limit: int = 12) -> list[dict]:
    manager = EnvironmentTreeManager(premise_id)
    summary = []
    for node in manager.environments_graph.get("environments_node", [])[:limit]:
        summary.append({
            "id": node.get("id"),
            "name": node.get("name"),
            "branch": node.get("branch"),
            "parent_id": node.get("parent_id"),
            "description": (node.get("description", "") or "")[:160],
        })
    return summary


def _invoke_json_prompt(prompt: str) -> dict:
    response = client.chat.completions.create(
        model=DEFAULT_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a strict JSON-output assistant. Output only JSON; do not output any explanation."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    content = response.choices[0].message.content
    parsed = extract_json.extract_json(content)
    if not isinstance(parsed, dict):
        raise ValueError(f"Environment creation proposal could not be parsed as JSON: {content}")
    return parsed


def propose_bootstrap_environment(
    premise_id: str,
    premise: str,
    narrative_context: str = "",
) -> dict:
    summary = _environment_library_summary(premise_id)
    prompt = f"""
You are the Environment Manager (managing T_E).
Task: Based on the premise, generate a T_phys creation proposal for the core physical scene that must exist at the start of the story.

Current T_E node summary:
{summary}

Novel premise:
{premise}

Current Global Planner / Event context:
{narrative_context or "(No additional Event context was supplied.)"}

The proposed opening scene must be geographically and causally compatible with this
Event context. If the context names a city, district, venue, institution, or world rule,
reuse it rather than inventing an unrelated setting.

Output JSON:
{{
  "proposal_type": "bootstrap_environment",
  "branch": "phys",
  "recommended": {{
    "name": "scene name",
    "parent_hint": "name of an existing T_phys node to mount under; leave empty to attach to phys_root if unsure",
    "profile": {{
      "description": "macro description",
      "location": "geographic label of the parent",
      "minutia": ["key prop 1", "key prop 2"],
      "atmosphere": "atmosphere"
    }}
  }},
  "conflict_check": {{
    "duplicate_risk": "low/medium/high",
    "possible_overlap_with": []
  }},
  "reason": "Why this is the right initial environment"
}}
"""
    proposal = _invoke_json_prompt(prompt)
    recommended = proposal.get("recommended")
    if not isinstance(recommended, dict):
        raise ValueError("Bootstrap environment proposal is missing recommended object.")

    proposal["branch"] = "phys"
    recommended["branch"] = "phys"
    recommended["parent_hint"] = None
    return proposal


def _normalize_environment_requirement(requirement: dict) -> dict:
    """Return the authoritative environment proposal requirement.

    ``parent_hint`` is a graph edge, not descriptive prose.  It may therefore be
    absent/root (``None``), a canonical global ``e_K`` ref, or a proposal-backed
    ``e_tmp_NN`` ref.  Names such as ``WorldRoot:Physical`` are deliberately
    rejected because they cannot be resolved atomically during M_sub projection.
    """
    if not isinstance(requirement, dict):
        raise ValueError("Environment requirement must be a JSON object.")

    normalized = deepcopy(requirement)
    branch = str(normalized.get("branch") or "phys").strip().lower()
    if branch not in ("phys", "conc"):
        raise ValueError("Environment requirement branch must be 'phys' or 'conc'.")

    raw_parent = normalized.get("parent_hint")
    if raw_parent is None or (
        isinstance(raw_parent, str) and not raw_parent.strip()
    ):
        parent_hint = None
    elif not is_environment_ref(raw_parent):
        raise ValueError(
            "Environment requirement parent_hint must be null, e_K, or e_tmp_NN; "
            f"got {raw_parent!r}."
        )
    else:
        parent_hint = raw_parent.strip()

    normalized["branch"] = branch
    normalized["parent_hint"] = parent_hint
    return normalized


def propose_environment_creation(
    premise_id: str, requirement: dict, premise: str, event_context: str = ""
) -> dict:
    requirement = _normalize_environment_requirement(requirement)
    summary = _environment_library_summary(premise_id)
    target_branch = requirement["branch"]
    authoritative_parent = requirement["parent_hint"]

    if target_branch == "phys":
        profile_block = """
    "profile": {{
      "description": "macro description",
      "location": "geographic label of the parent",
      "minutia": ["key prop 1", "key prop 2"],
      "atmosphere": "atmosphere"
    }}"""
    else:
        profile_block = """
    "profile": {{
      "description": "concept description",
      "attributes": {{ "key": "value" }}
    }}"""

    prompt = f"""
You are the Environment Manager (managing T_E).
Task: Based on the function-slot requirement given by the Event Planner and the current state of T_E, output the most suitable new-node proposal.

Current T_E node summary:
{summary}

Novel premise:
{premise}

Current event context:
{event_context}

Environment requirement (with branch preference):
{requirement}

Output JSON:
{{
  "proposal_type": "environment_creation",
  "requirement_key": "{requirement.get('requirement_key', '')}",
  "branch": "{target_branch}",
  "recommended": {{
    "name": "node name",
    "parent_hint": "name of an existing node (same branch) to mount under; leave empty to attach to the corresponding root if unsure",
{profile_block}
  }},
  "conflict_check": {{
    "duplicate_risk": "low/medium/high",
    "possible_overlap_with": []
  }},
  "reason": "Reason for the recommendation"
}}
"""
    proposal = _invoke_json_prompt(prompt)
    recommended = proposal.get("recommended")
    if not isinstance(recommended, dict):
        raise ValueError("Environment creation proposal is missing recommended object.")


    proposal["requirement_key"] = requirement.get("requirement_key", "")
    proposal["branch"] = target_branch
    proposal["requirement"] = deepcopy(requirement)
    recommended["branch"] = target_branch
    recommended["parent_hint"] = authoritative_parent
    return proposal
