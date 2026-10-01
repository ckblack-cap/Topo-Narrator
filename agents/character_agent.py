import re

from langgraph.prebuilt import create_react_agent
# from langchain.agents import create_agent
from agents import build_langchain_model_client, DEFAULT_MODEL_NAME
from agents import AgentGlobalConfig
from tools.character_graph_manager import CharacterGraphManager
from tools import extract_json


client = AgentGlobalConfig.GPTCLIENT

def build_character_agent(premise_id):
    character_manager = CharacterGraphManager(premise_id)

    tools = [
        character_manager.fuzzy_search_characters,
        character_manager.find_characters_info_by_id,
        character_manager.find_relationship_by_id,
        character_manager.find_characters_info_by_name,
        character_manager.search_character_by_attribute
    ]

    model_client = build_langchain_model_client()

    character_agent = create_react_agent(
        model=model_client,
        tools=tools,
        prompt=("""
                # Role
                You are **the Character Information Management expert for a novel**. You manage all character records of this novel and the complex web of relationships among them. Your core duty is to use the database query tools and to verify accuracy in your answers — especially the fuzzy_search_characters results (which may return the wrong character) — to answer any "character"-related question accurately.
                # Core Principles (Critical)
                1. **You have no prior knowledge about this novel. Speculation is forbidden.** Before answering any question, you **MUST** call a tool to fetch data. You may never say "there is no data" or "the database is empty" without first having called a tool.
                2. **Formal character refs are authoritative.** A public ref such as `c_1` maps to internal integer id `1`. Whenever a question contains `c_K`, perform an exact ID lookup (the query tools accept `c_K`) before any fuzzy/name lookup. If the returned node's name conflicts with the name assumed by the question, explicitly report that conflict; never silently equate them.
                3. **Even when the question feels vague, try the most relevant tool first to probe.**
                # Character Information Schema
                    The graph consists of **Nodes** and **Edges**. The data structure is defined as follows:
                    1. **Nodes (characters_node)**: a single character.
                    2. **Edges (characters_relationship)**: relationships between characters.
                        {
                            "characters_node":[{
                              "id": 1 , // internal integer; public ref is c_1
                              "name": "Character name",
                              "aliases": ["alias 1", "alias 2", "nickname", "title / honorific"],
                              "description": "[Global setting] Inherent character attributes: background, personality traits, core abilities, appearance. Note: update this field only when the character undergoes a major transformation (e.g., reconstructive surgery, moral darkening / corruption, identity reveal).",
                              "short_term_goal": "[Current intent] Specific goal the character wants to achieve in the current chapter or near future.",
                              "long_term_goal": "[Ultimate vision] Long-term pursuit running through the whole book; updated very rarely.",
                              "importance": "Protagonist | Supporting | Background",
                              "current_plot_participation": "key character | participant | non-participant",
                              "status": "[Real-time state] Current physical / mental state (e.g., severely wounded, unconscious, fearful, excited, captured)."
                            }],
                            "characters_relationship": [
                            {
                              "source": 1 , // source character id (integer), NOT a name string
                              "target": 3 , // target character id (integer), NOT a name string
                              "current_type": "current relationship type (e.g., ally / enemy / lover / master-disciple)",
                              "change_history": [
                                {
                                  "from": "old relationship",
                                  "to": "new relationship",
                                  "reason": "summary of the specific event that caused the change"
                                }
                              ]
                            }]
                        }'''
                # Constraints
                    1. **Domain constraint**: You **only handle character-related** questions (attributes, relationships, motivations, etc.).
                        **Strictly forbidden**: Do not answer questions purely about locations (maps, environments) or purely about plot (story direction, historical background) that are unrelated to character information. **Fixed reply**: When you receive such a question, respond exactly: "That is outside my scope — please consult the Environment or Plot Management Agent."

                    2. **Empty / missing handling**:
                        * If a tool call shows the database is empty (no nodes), reply: "The character graph is currently empty; no information has been recorded yet."
                        * If the graph is not empty but the specific character cannot be found, reply: "The information for character XX is not currently recorded in the graph."

                # Workflow (think-and-act steps)
                    After receiving a user question, follow these steps:
                    1. **Intent recognition**: decide whether the question belongs to the "character" category.
                        * If it asks about locations / pure plot -> apply the refusal policy.
                        * If it asks about characters -> continue.
                    2. **Tool retrieval**:
                        * Based on the question, query `characters_node` to get character attributes.
                        * If necessary, query `characters_relationship` to get social connections or relationship-change reasons.
                    3. **Analysis**:
                        If a single tool query does not yield a result, you must reason and call tools multiple times until you get the answer (if it exists in the database).
                    4. **Final output for Event Planner retrieval**: return a decision-oriented condensed summary, not a graph dump. Include the authoritative c_K ref/name and only the fields directly requested (state, goal, relationship/current_type, or relevant H_ij reasons). Keep the final answer under 4,000 characters and normally within 8 bullets. Never reproduce a full description/profile unless the caller explicitly needs that exact field.

                # Examples (Few-Shot)
                    **User**: "What does the character Alice want? What is her relationship with Bob?"
                    **You**: (Call tools to query Alice's attributes and her relationship with Bob...)
                    "Alice (id:1) is the protagonist of the book. Her long-term goal is ... [long_term_goal]. She and Bob (id:3) are currently in a [current_type] relationship. This relationship has changed before: [change_history details]..."

                    **User**: "Who is the protagonist? Who are her adversaries?"
                    **You**: (Call tools to query the protagonist and her adversaries and return the complete JSON-formatted information...)
                    "characters_node":[{
                          "id": 1 , // integer
                          "name": "Alice",
                          "aliases": [],
                          "description": "Born in a coastal town where convergence of magical fortune marks the starting point of the story. At age 5, her bonded artifact was shattered, her parents perished from the backlash, and she became an orphan, surviving by hauling porcelain for others, enduring bullying yet keeping her conscience...",
                          "short_term_goal": "Escape from the pursuers",
                          "long_term_goal": "Avenge her father",
                          "importance": "Protagonist",
                          "current_plot_participation": "key character",
                          "status": "captured"
                        }
                        ...adversary records...
                    ],
                    **User**: "Where is Mount Falling located in this novel?"
                    **You**: "That is outside my scope — please consult the Environment or Plot Management Agent."

                    **User**: (when the character graph is empty) "Tell me about the protagonist."
                    **You**: "The character graph is currently empty; no information has been recorded yet."
                """),

        name="Character_Agent",
    )
    return character_agent


def _character_library_summary(premise_id: str, limit: int = 12) -> list[dict]:
    manager = CharacterGraphManager(premise_id)
    summary = []
    for node in manager.characters_graph.get("characters_node", [])[:limit]:
        summary.append({
            "id": node.get("id"),
            "name": node.get("name"),
            "aliases": node.get("aliases", []),
            "importance": node.get("importance", ""),
            "description": node.get("description", "")[:180],
            "status": node.get("status", ""),
            "short_term_goal": node.get("short_term_goal", ""),
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
        raise ValueError(f"Character creation proposal could not be parsed as JSON: {content}")
    return parsed


_NAMED_PARTICIPANT_RE = re.compile(
    r"\b([A-Z][\w''.-]*(?:\s+[A-Z][\w''.-]*)+),\s+(?:an?|the)\s+",
    re.UNICODE,
)


def _validate_bootstrap_character_context(proposal: dict, narrative_context: str) -> None:
    recommended = proposal.get("recommended") if isinstance(proposal, dict) else None
    name = recommended.get("name") if isinstance(recommended, dict) else None
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Bootstrap character proposal is missing recommended.name.")

    context = narrative_context.strip() if isinstance(narrative_context, str) else ""
    if not context:
        return
    folded_context = context.casefold()
    if name.strip().casefold() in folded_context:
        return

    named_participants = [
        match.group(1).strip()
        for match in _NAMED_PARTICIPANT_RE.finditer(context)
        if match.group(1).strip()
    ]
    if named_participants:
        raise ValueError(
            "Bootstrap protagonist is inconsistent with the named Event participants: "
            f"recommended={name!r}, expected one of {named_participants!r}."
        )

    context_anchor = proposal.get("context_anchor")
    if not isinstance(context_anchor, str) or not context_anchor.strip():
        raise ValueError(
            "Bootstrap character proposal must include an exact context_anchor copied "
            "from the Event abstract."
        )
    if context_anchor.strip().casefold() not in folded_context:
        raise ValueError(
            f"Bootstrap context_anchor {context_anchor!r} does not occur in the Event abstract."
        )


def propose_bootstrap_character(
    premise_id: str,
    premise: str,
    narrative_context: str = "",
) -> dict:
    summary = _character_library_summary(premise_id)
    prompt = f"""
You are the Character CEP Agent for this novel.
Task: Based on the novel premise, generate a character-creation proposal for "the core protagonist who must stably exist before the story begins."

Current character library summary:
{summary}

Novel premise:
{premise}

Current Global Planner / Event context:
{narrative_context or "(No additional Event context was supplied.)"}

Identity consistency rules:
1. If the Event context explicitly names its protagonist or lead participant, reuse that
   exact name. Never invent a different lead in another city or setting.
2. Set context_anchor to the exact participant name or role phrase copied verbatim from
   the Event context (for example, "Theo Park" or "the protagonist").
3. The profile's background, profession, location, goals, and relationships must be
   compatible with both the premise and the Event context.

Output JSON:
{{
  "proposal_type": "bootstrap_character",
  "context_anchor": "exact participant name or role phrase copied from the Event context",
  "recommended": {{
    "name": "character name",
    "profile": {{
      "importance": "Protagonist / Supporting / Background",
      "gender": "",
      "age": 18,
      "appearance": "Specific physical traits: height, build, hair, eyes, distinguishing marks (scars, tattoos, posture quirks), and signature clothing/accessories.",
      "personality": "Core temperament + at least one internal contradiction or hidden trait that creates dramatic tension (e.g., publicly stoic but privately impulsive).",
      "identity_ability": "Social role / profession / class + key skills or supernatural abilities (if any) with concrete scope and limitations.",
      "description": "**Comprehensive character profile, MINIMUM 150 characters, structured as 4 paragraphs**: (1) Background — origin, upbringing, defining past event; (2) Personality — core traits + an inner contradiction; (3) Appearance — concrete physical and stylistic details; (4) Motivation — the wound or belief driving present behavior. AVOID generic clichés ('mysterious past', 'cold exterior warm heart'). GIVE concrete specifics: a particular scar, a verbal tic, a childhood loss, a moral line they refuse to cross.",
      "event_drive": "Specific in-world event or unresolved wound that explains why this character acts the way they do RIGHT NOW.",
      "short_term": "",
      "long_term": ""
    }}
  }},
  "conflict_check": {{
    "duplicate_risk": "low/medium/high",
    "possible_overlap_with": []
  }},
  "reason": "Why this is the best protagonist-initialization plan"
}}
"""
    proposal = _invoke_json_prompt(prompt)
    _validate_bootstrap_character_context(proposal, narrative_context)
    recommended = proposal.get("recommended")
    profile = recommended.get("profile") if isinstance(recommended, dict) else None
    if not isinstance(profile, dict):
        raise ValueError("Bootstrap character proposal is missing recommended.profile.")
    profile["importance"] = "Protagonist"
    return proposal


def propose_character_creation(premise_id: str, requirement: dict, premise: str, event_context: str = "") -> dict:
    summary = _character_library_summary(premise_id)
    prompt = f"""
You are the Character CEP Agent for this novel.
Task: Based on the "character function-slot requirement" given by the PlanAgent and the current character library, output the most suitable new-character proposal.

Current character library summary:
{summary}

Novel premise:
{premise}

Current event context:
{event_context}

Character requirement:
{requirement}

# Naming Convention (STRICT — prevents identity-split bugs in later chapters)
The `name` field MUST be the character's **formal proper noun** (e.g., a personal name, a fully
qualified title like "Captain Reyes"). This is the canonical identity used by the graph for the
entire story. It MUST NOT change later.

If this character is plotted to **first appear in the narrative under a relational, professional,
or anonymous descriptor** (and only later have the formal name revealed), still use the formal
proper noun in `name`, and put every descriptor the writer will use before the reveal into the
`aliases` list. Examples of strings that are NOT acceptable in `name` but DO belong in `aliases`:
  - Relational: "xx's sister", "the elder brother", "Lu Guanlan's older sister"
  - Professional / role: "the journalist", "the park ranger", "the photographer"
  - Anonymous descriptors: "the masked man", "a middle-aged woman", "a person dressed in black"

Rationale: if the graph is first registered with a descriptor as name, the later reveal triggers
a rename that downstream identity guards correctly refuse — which would either (a) freeze the
descriptor as the permanent name, or (b) cause a duplicate entity to be spawned for the revealed
name. Putting the formal name in `name` from the start avoids both.

Output JSON:
{{
  "proposal_type": "character_creation",
  "requirement_key": "{requirement.get('requirement_key', '')}",
  "recommended": {{
    "name": "Formal proper noun ONLY (see Naming Convention above)",
    "aliases": [
      "Descriptors the narrative will use before the formal name is revealed (e.g., an older sister of xx, the journalist, a middle-aged man). May be empty if the character is named openly from the first appearance."
    ],
    "profile": {{
      "importance": "Protagonist / Supporting / Background",
      "gender": "",
      "age": 18,
      "appearance": "Specific physical traits: height, build, hair, eyes, distinguishing marks (scars, tattoos, posture quirks), and signature clothing/accessories.",
      "personality": "Core temperament + at least one internal contradiction or hidden trait that creates dramatic tension (e.g., publicly stoic but privately impulsive).",
      "identity_ability": "Social role / profession / class + key skills or supernatural abilities (if any) with concrete scope and limitations.",
      "description": "**Comprehensive character profile, MINIMUM 150 characters, structured as 4 paragraphs**: (1) Background — origin, upbringing, defining past event; (2) Personality — core traits + an inner contradiction; (3) Appearance — concrete physical and stylistic details; (4) Motivation — the wound or belief driving present behavior. AVOID generic clichés ('mysterious past', 'cold exterior warm heart'). GIVE concrete specifics: a particular scar, a verbal tic, a childhood loss, a moral line they refuse to cross.",
      "event_drive": "Specific in-world event or unresolved wound that explains why this character acts the way they do RIGHT NOW in the current event context.",
      "short_term": "",
      "long_term": ""
    }}
  }},
  "conflict_check": {{
    "duplicate_risk": "low/medium/high",
    "possible_overlap_with": []
  }},
  "reason": "Reason for the recommendation"
}}
"""
    return _invoke_json_prompt(prompt)
