import json
import os
import re
from copy import deepcopy

from langchain_core.messages import AIMessage, HumanMessage

from agents import character_agent, environment_agent
from tools import extract_json, event_schema
from tools.agent_event_logger import get_trace_logger
from tools.graph_refs import (
    is_real_character_ref,
    is_real_environment_ref,
    is_real_plot_ref,
    is_temp_character_ref,
    is_temp_environment_ref,
    parse_character_ref,
    parse_plot_ref,
)


def build_writer_agent(first_writing=False):
    from langgraph.prebuilt import create_react_agent
    from agents import build_langchain_model_client

    first_writing_prompt = """
            # Role
            You are a senior ghostwriter and novelist. Your specialty is crafting **opening chapters** that hook readers instantly. You excel at "scene expansion" and "sensory grounding," and you can establish a strong narrative voice from nothing.

            # Task
            Your task is to rewrite a rough, sparse [Chapter Summary] into a polished, high-quality **first chapter** of a novel that includes the required plot points listed in [Must Plots].

            # Context Data and Variables
            Please read the following inputs carefully.
            (**Concrete content is provided below.**)

            1.  **[Character Profile]**:
                You must strictly follow these character settings. Introduce characters naturally through their actions and dialogue; avoid "info dumps" (long blocks of purely expository text).

            2.  **[Environment Information]**:
                Use these details to build the world. This is the reader's first entry into this setting, so the sensory details must be vivid and immersive.

            3.  **[Chapter Summary]**:
                Do not change the key plot beats. Your job is to expand this chapter summary into the full novel prose.

            4. **[Must Plots]**:
                These are required plot points that must appear in this chapter; you must include them.

            # Writing Guidelines (key requirements for the first chapter)

            ### 1. Establish tone and narrative voice (opening hook)
            * **Analyze the genre:** Based on the [Chapter Summary], immediately determine the appropriate tone (e.g., mysterious, romantic, action-driven, etc.).
            * **Strong opening:** The first paragraph matters most. Open with a strong image, an intriguing action, or a distinctive character voice. Do not open with banal weather descriptions or a "waking up" scene unless it is genuinely critical to the plot.
            * **Establish stakes:** Even subtly, hint early at the character's core desire or the impending conflict.

            ### 2. "Show, don't tell" (worldbuilding presentation)
            * **Immersive scene:** Do not just list environmental information; weave it into action (e.g., do not write "the room was messy" — write "she stepped over a pile of discarded clothes to reach the window").
            * **Sensory detail:** Engage all five senses (sight, sound, smell, touch, taste) so the world feels real and tangible immediately.

            ### 3. Character introduction
            * **Action first:** Introduce characters through what they are *doing* or *saying* rather than through a long paragraph telling the reader "who they are." Let the reader get to know them gradually.
            * **Internal state:** Express emotion through physiological reactions (e.g., shallow breathing, unconscious finger rubbing) rather than slapping an emotion label on the character directly.

            ### 4. Expansion and pacing
            * **Unpack the summary:** The [Chapter Summary] is usually a compressed summary of events. You must expand it into full scenes with real dialogue and interaction.
            * **Pacing:** As the first chapter, balance worldbuilding introduction and plot progression. Do not let description slow the story to a crawl.

            # Negative Constraints (strictly forbidden)
            * **Entity scope:** Do not introduce any named or individually identifiable character outside [Character Profile]. Unnamed incidental crowd members may appear only as generic background and must not drive a plot beat.
            * **Scene scope:** Keep every scene inside the locations supplied by [Environment Information]. You may add sensory details and movable objects inside those locations, but must not invent or switch to a new named location.
            * **No purple prose:** Avoid overwrought, archaic, or pretentious wording (e.g., avoid "tapestry of souls", "symphony of emotions", "ethereal glow"), unless the specific character's voice genuinely requires it. Keep the prose grounded.
            * **No info dumps:** Do not stop the story to explain the entire world history or a character's complete backstory. Information should surface naturally as needed.
            * **No repetition:** Do not repeat the same inner monologue or description over and over.

            # Output Format
            Output **only** the body of the story — no title, no chapter number. The output should be a polished chapter ready for publication (about 1800-2500 words, depending on content density).

        """

    continue_writing_prompt = """

            # Role
            You are a senior ghostwriter and novelist with chameleon-like style-matching ability who can switch effortlessly between genres. Your specialty is "scene expansion" and "sensory grounding." Your task is to rewrite a rough [Chapter Summary] into a polished, high-quality novel chapter that contains the required plot points in [Must Plots], and that connects seamlessly to the previous chapter as a natural continuation.

            # Context Data and Variables
            Please read the following inputs carefully.
            (**Concrete content is provided below.**)
            1.  **[Previous Chapter Text Reference]**:
                Use this to match narrative voice, pacing, and writing style. The new chapter must begin immediately where this text ends, maintaining perfect continuity.
                ⚠️ **Hard state-continuity constraints (must not be violated)**:
                - [Location] The location at the opening of this chapter must match the location at the end of the previous chapter. If the [Chapter Summary] requires a different location, the chapter must open with a transition paragraph (movement / being escorted / being rescued / waking up already moved, etc.) explaining how the protagonist got from the previous location to the new one. "Suddenly appearing in a new place" is forbidden.
                - [Life and Death] Any character explicitly killed, vanished, sacrificed, turned to ash, murdered, or permanently gone in the previous chapter must NEVER appear in this chapter as a living, normal person speaking or acting. If the [Chapter Summary] includes such a character, you must handle it through an explicit mechanism (hallucination / memory flashback / recording / resonant echo, etc.).
                - [Companions Present] People who were with the protagonist at the end of the previous chapter must either still be present at the start of this chapter, or the previous chapter must have explained why they separated. They cannot vanish or be replaced without reason.
                - [Item / Injury / Ability State] Key items the protagonist was carrying at the end of the previous chapter, severe injuries, locked / awakened ability states must persist into this chapter; any change must be explicitly explained in this chapter.
                - [Causal Chain] Strong causes established in the previous chapter (being wanted, factions declaring war, a key secret being exposed, etc.) must be acknowledged at least once in this chapter; you may not treat them as if they never happened.

            2.  **[Character Profile]**:
                You must strictly adhere to these personalities, speech habits, and physical descriptions. No "Out of Character" behavior.

            3.  **[Environment Information]**:
                Use these details to ground the scene. Ensure the setting description is consistent with established facts (e.g., lighting, layout, smells).

            4.  **[Chapter Summary]**:
                This is the event flow. Do not change key plot beats or the ending direction. Your job is to expand these events into full scenes with dialogue, action, and sensory detail.

            5. **[Must Plots]**:
                These are required plot points that must appear in this chapter; you must include them.


            # Writing Guidelines (key)

            ### 1. Tone and style analysis (adaptive)
            * **Analyze the genre:** Based on the input draft and the previous chapter text, determine the appropriate tone (e.g., for a romance draft, use a warm, intimate voice; for a thriller / suspense, use a sharp, tense voice).
            * **Consistency:** Ensure the narrative voice matches the character described and flows naturally from the [Previous Chapter Text Reference].

            ### 2. "Show, don't tell" (the golden rule)
            * **Concrete imagery:** Do not use abstract adjectives to summarize a scene. Do not write "the atmosphere was electric" — write the specific sounds, smells, and physical reactions (e.g., "the hum of the speakers", "the smell of stale coffee", "goosebumps rising on her arm").
            * **Sensory detail:** Based on the [Environment Information], engage all five senses (sight, sound, smell, touch, taste) to nail the reader to the scene.

            ### 3. Character depth and dialogue
            * **Natural cadence:** Dialogue must sound like real people, not robots. Use subtext, interruptions, pauses, and action beats embedded in dialogue. Avoid blunt "thesis statement" emotional declarations.
            * **Internal state:** Express emotion through physiological reactions (e.g., shallow breathing, finger rubbing, looking away) rather than slapping an emotion label on the character (e.g., "he was nervous").

            ### 4. Expansion and pacing
            * **Unpack the summary:** The [Chapter Summary] often compresses action (e.g., "they talked about their dreams"). You must expand it into a full scene with real dialogue and interaction.
            * **Pacing:** Slow down at key emotional beats or turning points so the reader feels the weight; speed up during transitions.

            # Negative Constraints (strictly forbidden)
            * **Entity scope:** Do not introduce any named or individually identifiable character outside [Character Profile]. Unnamed incidental crowd members may appear only as generic background and must not drive a plot beat.
            * **Scene scope:** Keep every scene inside the locations supplied by [Environment Information]. You may add sensory details and movable objects inside those locations, but must not invent or switch to a new named location.
            * **No purple prose:** Avoid overwrought, archaic, or pretentious wording (e.g., avoid "tapestry of souls", "symphony of emotions", "ethereal glow", "shroud"), unless the specific character's voice genuinely requires it. Keep the prose grounded and real.
            * **No plot deviation:** Do not invent major plot beats not present in the [Chapter Summary].
            * **No repetition:** Do not repeat the same inner monologue or description multiple times.

            # Output Format
            Output **only** the body of the story — no title, no chapter number. The output should be a polished chapter ready for publication (aim for about 1800-2500 words, depending on content density). This is a pacing target, not a reason to cut required plot or continuity material when a modest overrun is necessary.
        """

    tools = []

    model_client = build_langchain_model_client()

    if first_writing:
        prompt = first_writing_prompt
    else:
        prompt = continue_writing_prompt

    writer_agent = create_react_agent(
        model=model_client,
        tools=tools,
        prompt=prompt,
        name="Writer_Agent",
    )


    return writer_agent


def _seed_entity_lookup(event_payload: dict, entity_type: str) -> dict[str, dict]:
    seed_subgraph = event_payload.get("seed_subgraph", {})
    entity_bucket = seed_subgraph.get(entity_type, {})
    items = entity_bucket.get("new", [])
    lookup: dict[str, dict] = {}
    if not isinstance(items, list):
        return lookup
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("temp_id"), str):
            lookup[item["temp_id"]] = item
    return lookup


def _load_work_meta(work_premise_id: str, *, required: bool = False) -> dict:
    from tools.working_memory import get_work_meta

    try:
        meta = get_work_meta(work_premise_id, required=required)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        if required:
            raise
        return {}
    return meta if isinstance(meta, dict) else {}


def _coerce_character_work_id(value) -> int | None:
    parsed = parse_character_ref(value)
    if parsed is not None:
        return int(parsed)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_environment_work_id(value) -> str | None:
    if isinstance(value, int):
        return f"e_{value}"
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    return f"e_{value}" if value.isdigit() else value


def _find_node_by_id(nodes: list[dict], target_id) -> dict | None:
    if target_id is None:
        return None
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if node_id == target_id or str(node_id) == str(target_id):
            return node
    return None


def _find_temp_node_by_seed_name(
    nodes: list[dict],
    seed_item: dict | None,
    existing_ids: set,
) -> dict | None:
    if not isinstance(seed_item, dict):
        return None
    raw_aliases = seed_item.get("aliases", [])
    aliases = raw_aliases if isinstance(raw_aliases, list) else [raw_aliases]
    names = {
        str(value).strip()
        for value in [seed_item.get("name"), *aliases]
        if isinstance(value, str) and value.strip()
    }
    if not names:
        return None
    candidates = [
        node
        for node in nodes
        if isinstance(node, dict)
        and node.get("id") not in existing_ids
        and (
            node.get("name") in names
            or bool(names.intersection(set(
                node.get("aliases", [])
                if isinstance(node.get("aliases", []), list)
                else [node.get("aliases")]
            )))
        )
    ]
    return candidates[0] if len(candidates) == 1 else None


def _compose_current_entity_record(
    temp_ref: str,
    seed_item: dict | None,
    current_node: dict | None,
    *,
    entity_type: str,
) -> dict:
    seed_item = seed_item if isinstance(seed_item, dict) else {}
    profile = seed_item.get("profile", {})
    record = deepcopy(profile) if isinstance(profile, dict) else {}

    if entity_type == "characters":
        if "short_term_goal" not in record and "short_term" in record:
            record["short_term_goal"] = record.pop("short_term")
        else:
            record.pop("short_term", None)
        if "long_term_goal" not in record and "long_term" in record:
            record["long_term_goal"] = record.pop("long_term")
        else:
            record.pop("long_term", None)
    elif entity_type == "environments":
        if "minutia" not in record and "micro_details" in record:
            micro_details = record.pop("micro_details")
            record["minutia"] = micro_details if isinstance(micro_details, list) else [micro_details]
        else:
            record.pop("micro_details", None)

    for key in ("name", "aliases", "branch", "function_slot"):
        if key in seed_item and key not in record:
            record[key] = deepcopy(seed_item[key])

    if isinstance(current_node, dict):
        record.update(deepcopy(current_node))
        record["record_source"] = "current_M_sub"
        record["working_memory_id"] = current_node.get("id")
    else:
        record["record_source"] = "seed_fallback"

    record["temp_ref"] = temp_ref
    return record


def _resolve_temp_entity_records(
    work_premise_id: str,
    event_payload: dict,
    refs: list[str],
    entity_type: str,
) -> list[dict]:
    from data import characters_graph_data, raw_environments_graph_data
    from tools.working_memory import validate_environment_projection

    if entity_type == "characters":
        graph = characters_graph_data(work_premise_id)
        if not isinstance(graph, dict) or not isinstance(graph.get("characters_node"), list):
            raise OSError(f"Failed to read working G_C for Writer: {work_premise_id}")
        nodes = graph.get("characters_node", [])
        map_key = "character_temp_to_id"
        graph_bucket = "characters"
        is_temp_ref = is_temp_character_ref
        coerce_work_id = _coerce_character_work_id
        existing_refs = (
            event_payload.get("seed_subgraph", {}).get("characters", {}).get("existing", []) or []
        )
        existing_ids = {
            cid for cid in (parse_character_ref(ref) for ref in existing_refs) if cid is not None
        }
    elif entity_type == "environments":
        graph = raw_environments_graph_data(work_premise_id)
        if not isinstance(graph, dict) or not isinstance(graph.get("environments_node"), list):
            raise OSError(f"Failed to read working T_E for Writer: {work_premise_id}")
        validate_environment_projection(graph)
        nodes = graph.get("environments_node", [])
        map_key = "environment_temp_to_id"
        graph_bucket = "environments"
        is_temp_ref = is_temp_environment_ref
        coerce_work_id = _coerce_environment_work_id
        existing_ids = set(
            event_payload.get("seed_subgraph", {}).get("environments", {}).get("existing", []) or []
        )
    else:
        raise ValueError(f"Unsupported entity_type: {entity_type}")

    meta = _load_work_meta(work_premise_id, required=True)
    temp_map = meta.get(map_key, {})
    temp_map = dict(temp_map) if isinstance(temp_map, dict) else {}
    graph_temp_map = graph.get("_meta", {}).get("temp_to_id", {})
    if isinstance(graph_temp_map, dict):
        for ref, work_id in graph_temp_map.items():
            temp_map.setdefault(ref, work_id)

    seed_lookup = _seed_entity_lookup(event_payload, graph_bucket)
    resolved: list[dict] = []
    seen: set[str] = set()
    for ref in refs or []:
        if not isinstance(ref, str) or not is_temp_ref(ref) or ref in seen:
            continue
        seen.add(ref)
        seed_item = seed_lookup.get(ref)
        work_id = coerce_work_id(temp_map.get(ref))
        current_node = _find_node_by_id(nodes, work_id)
        if current_node is None:
            current_node = _find_temp_node_by_seed_name(nodes, seed_item, existing_ids)
        if current_node is None:
            raise ValueError(
                f"Temp ref {ref} has no current node in working {entity_type} graph."
            )
        resolved.append(
            _compose_current_entity_record(
                ref,
                seed_item,
                current_node,
                entity_type=entity_type,
            )
        )
    return resolved


def _build_existing_character_prompt(character_refs: list[str], numeric_ids: list[int]) -> str:
    return f"""characters: {character_refs};
numeric_character_ids: {numeric_ids};
Task:
1. `characters` is the list of string ids (e.g., c_1) for characters that appear in this chapter and already exist in the database.
2. `numeric_character_ids` is the corresponding list of numeric database ids, which you can use to query the character store directly.
3. You must look up the character database and retrieve the complete, accurate information for every character.
4. After collecting all character information, output only the detailed information — no extra explanation.
# Every character record must include the full set of attributes from the database.
"""


def _build_existing_environment_prompt(environment_refs: list[str]) -> str:
    return f"""environments: {environment_refs};
Task:
1. `environments` is the list of string ids (e.g., e_1) for environments that appear in this chapter and already exist in the database. You must look up the environment database and retrieve the complete, accurate information for every environment.
2. After collecting all environment information, output only the detailed information — no extra explanation.
"""


def _resolve_chapter_context(
    premise_id: str,
    event_payload: dict,
    chapter_payload: dict,
    *,
    include_scope_records: bool = False,
):
    refs = chapter_payload.get("refs", {})
    character_refs = refs.get("characters", [])
    environment_refs = refs.get("environments", [])

    existing_character_ids = [item for item in character_refs if is_real_character_ref(item)]
    existing_environment_ids = [item for item in environment_refs if is_real_environment_ref(item)]

    current_temp_characters = _resolve_temp_entity_records(
        premise_id, event_payload, character_refs, "characters"
    )
    current_temp_environments = _resolve_temp_entity_records(
        premise_id, event_payload, environment_refs, "environments"
    )
    character_scope_records = deepcopy(current_temp_characters)
    environment_scope_records = deepcopy(current_temp_environments)

    character_info = (
        "Temporary-reference character records resolved from the current working memory "
        f"(M_sub): {json.dumps(current_temp_characters, ensure_ascii=False)};\n"
    )
    env_info = (
        "Temporary-reference environment records resolved from the current working memory "
        f"(M_sub): {json.dumps(current_temp_environments, ensure_ascii=False)};\n"
    )

    if existing_character_ids:
        ca = character_agent.build_character_agent(premise_id)
        character_id_values = [parse_character_ref(item) for item in existing_character_ids]
        numeric_ids = [cid for cid in character_id_values if cid is not None]
        res = ca.invoke({"messages": [HumanMessage(content=_build_existing_character_prompt(existing_character_ids, numeric_ids))]})
        character_info += "Existing character records from the database: " + res["messages"][-1].content + ";"
        # Validation must use structured graph data, never infer an allow-list from the
        # model's free-form lookup response.
        from tools.character_graph_manager import CharacterGraphManager

        stored_records = CharacterGraphManager(premise_id).find_characters_info_by_id(numeric_ids)
        if len(stored_records) != len(set(numeric_ids)):
            raise ValueError("Writer could not resolve every existing character ref for entity validation.")
        character_scope_records.extend(deepcopy(stored_records))

    if existing_environment_ids:
        ea = environment_agent.build_environment_agent(premise_id)
        res = ea.invoke({"messages": [HumanMessage(content=_build_existing_environment_prompt(existing_environment_ids))]})
        env_info += "Existing environment records from the database: " + res["messages"][-1].content + ";"
        from tools.environment_graph_manager import EnvironmentTreeManager

        stored_records = EnvironmentTreeManager(premise_id).find_environments_info(existing_environment_ids)
        if len(stored_records) != len(set(existing_environment_ids)):
            raise ValueError("Writer could not resolve every existing environment ref for entity validation.")
        environment_scope_records.extend(deepcopy(stored_records))

    if include_scope_records:
        return character_info, env_info, character_scope_records, environment_scope_records
    return character_info, env_info


_PLOT_CONTEXT_FIELDS = (
    "chapter",
    "overview",
    "details",
    "importance",
    "event",
    "characters_involved",
    "environment",
    "foreshadowing",
)


_ENGLISH_CHAPTER_MIN_WORDS = 1800
_ENGLISH_CHAPTER_MAX_WORDS = 2500
_ENGLISH_CHAPTER_OVERFLOW_WORDS = 4000
# Chinese does not have whitespace-delimited words.  A 3,000--5,000 Han-character
# chapter is the practical equivalent used by the optional Chinese review mode.
_CHINESE_CHAPTER_MIN_UNITS = 3000
_CHINESE_CHAPTER_MAX_UNITS = 5000
_CHINESE_CHAPTER_OVERFLOW_UNITS = 8000
_DEFAULT_WRITER_REPAIR_ATTEMPTS = 2

_LATIN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[''-][A-Za-z0-9]+)*")
_CJK_CHARACTER_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CAPITALIZED_TOKEN = r"(?:[A-Z][A-Za-z]*(?:[''][A-Za-z]+)?|[A-Z]{2,})"
_CAPITALIZED_SEQUENCE_RE = re.compile(
    rf"(?<![A-Za-z0-9_]){_CAPITALIZED_TOKEN}"
    rf"(?:[ \t]+(?:(?:of|the|and|de|del|la|le|van|von|&)[ \t]+)?{_CAPITALIZED_TOKEN})*"
    rf"(?:['']s)?"
)

_LEADING_PROSE_WORDS = {
    "a", "an", "the", "and", "but", "or", "so", "then", "when", "while",
    "after", "before", "inside", "outside", "across", "behind", "beside",
    "near", "at", "in", "on", "from", "to", "for", "as", "by", "through",
    "under", "over", "into", "with", "without", "meanwhile", "still", "finally",
    "now", "if", "although", "because", "beyond", "around", "between", "above",
    "below", "toward", "towards", "his", "her", "their", "our", "my", "your",
    "eventually", "suddenly", "later", "soon", "next", "perhaps", "maybe",
    "instead", "otherwise", "therefore", "however", "yet", "once",
    "anything", "everything", "something", "nothing", "sometimes", "always",
    "only", "even", "all", "every", "either", "neither", "do", "does", "did",
    "can", "could", "would", "should", "will", "may", "might", "must", "am",
    "is", "are", "was", "were", "have", "has", "had",
}
_SINGLETON_NON_NAME_WORDS = _LEADING_PROSE_WORDS | {
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "them",
    "this", "that", "these", "those", "who", "whom", "whose", "what", "which",
    "where", "why", "how", "there", "here", "yes", "no", "okay", "ok",
    "someone", "anyone", "everyone", "nobody", "people", "one", "two", "three",
    "four", "five", "six", "seven", "eight", "nine", "ten",
}
_GENERIC_ROLE_SINGLETONS = {
    "defender", "guardian", "warrior", "soldier", "officer", "captain", "doctor",
    "councilor", "councillor", "archivist", "scribe", "guard", "technician",
    "bartender", "singer", "musician", "producer", "engineer", "attendant",
    "courier", "messenger", "commander", "merchant", "healer", "patroller",
    "detective", "inspector", "investigator", "agent", "deputy", "sheriff",
    "constable", "sergeant", "lieutenant", "chief", "guide", "mayor",
}
_PERSON_TITLE_WORDS = {
    "mr", "mrs", "ms", "miss", "dr", "doctor", "captain", "officer",
    "professor", "commander", "king", "queen", "prince", "princess",
}
_PERSON_ROLE_PREFIX_WORDS = _PERSON_TITLE_WORDS | _GENERIC_ROLE_SINGLETONS | {
    "sister", "brother", "mother", "father", "uncle", "aunt", "elder",
}
_GENERIC_CAPITALIZED_PHRASES = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "spring", "summer", "autumn",
    "fall", "winter", "north", "south", "east", "west", "earth", "internet",
    "new year", "new years eve", "christmas", "christmas eve", "christmas day",
    "thank god", "good lord", "oh god", "jesus christ", "happy birthday",
    "ladies and gentlemen", "good morning", "good afternoon", "good evening",
    "yes sir", "no sir", "yes maam", "no maam",
    "thank you", "of course", "all right", "no one",
}
_GENERIC_SIGNAGE_WORDS = {
    "access", "authorized", "caution", "closed", "danger", "do", "emergency",
    "enter", "entry", "exit", "hazard", "keep", "no", "not", "notice", "only", "out",
    "personnel", "private", "restricted", "safety", "staff", "stop", "structural",
    "unauthorized", "warning",
}
_GENERIC_SIGNAGE_ACTION_WORDS = {
    "close", "cross", "delay", "enter", "keep", "know", "open", "proceed",
    "remove", "stop", "touch", "turn", "use", "wait",
}
_TRAILING_COORDINATE_PRONOUNS = {"i", "we", "you", "he", "she", "it", "they"}
_LOCATION_SUFFIXES = {
    "city", "town", "village", "district", "street", "road", "avenue", "boulevard",
    "lane", "drive", "park", "square", "plaza", "station", "airport", "harbor",
    "harbour", "hotel", "inn", "cafe", "bar", "club", "theater", "theatre", "hall",
    "room", "center", "centre", "building", "bridge", "river", "lake", "mountain",
    "bay", "beach", "island", "county", "state", "country", "valley", "hospital",
    "school", "university", "church", "temple", "arena", "stadium", "market", "mall",
    "museum", "gallery", "library", "restaurant", "diner", "warehouse", "alley",
    "hallway", "corridor", "lobby", "lounge", "backstage", "vault", "archive",
    "palace", "reef", "trench", "chamber", "cavern", "cave", "grotto", "gate",
    "court", "citadel", "fortress", "tower", "dock", "pier", "shrine", "sanctuary",
    "laboratory", "lab", "office",
}
_LOCATION_PREFIXES = {"lake", "mount", "mt", "fort", "port", "saint", "st"}
_NON_PERSON_ENTITY_SUFFIXES = {
    "company", "corporation", "corp", "inc", "llc", "council", "committee",
    "department", "agency", "network", "records", "studios", "foundation",
    "association", "society", "team", "band", "orchestra", "festival", "showcase",
    "contest", "tournament", "conference", "ceremony", "stream", "vod",
}
_PERSON_ACTION_WORDS = {
    "said", "asked", "replied", "shouted", "whispered", "walked", "entered", "smiled",
    "laughed", "nodded", "looked", "held", "took", "turned", "raised", "stared",
    "watched", "answered", "followed", "joined", "left", "leaned", "stepped", "waved",
}
_PERSON_OBJECT_CUES = {
    "met", "called", "asked", "told", "saw", "heard", "joined", "followed", "helped",
    "thanked", "introduced", "watched", "found", "greeted",
}
_WORK_TITLE_CUES_RE = re.compile(
    r"(?:\b(?:song|track|album|book|novel|film|movie|poem|piece|number|show|play|"
    r"painting|sculpture|story|chapter|episode|tune|setlist)\b(?:\s+(?:as|called|named|titled))?"
    r"|\b(?:played|performed|sang|queued|covered)\b(?:\s+the)?(?:\s+(?:song|track|tune))?)"
    r"""\s+["'\u201C\u201D\u2018\u2019*]?\s*$""",
    re.IGNORECASE,
)


class ChapterOutputContractError(ValueError):
    """The Writer exhausted its bounded repairs without producing a valid chapter."""


def _writer_repair_attempts() -> int:
    """Return a small, hard-bounded number of local Writer repair calls."""
    raw = os.getenv("TOPO_WRITER_REPAIR_ATTEMPTS", str(_DEFAULT_WRITER_REPAIR_ATTEMPTS))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = _DEFAULT_WRITER_REPAIR_ATTEMPTS
    return max(0, min(value, _DEFAULT_WRITER_REPAIR_ATTEMPTS))


def _chapter_length_contract(output_lang: str) -> tuple[int, int, str]:
    if output_lang == "zh":
        return (
            _CHINESE_CHAPTER_MIN_UNITS,
            _CHINESE_CHAPTER_MAX_UNITS,
            "Chinese characters/Latin words",
        )
    return _ENGLISH_CHAPTER_MIN_WORDS, _ENGLISH_CHAPTER_MAX_WORDS, "English words"


def _chapter_overflow_limit(output_lang: str) -> int:
    """Operational runaway guard; the normal maximum is only a pacing target."""
    if output_lang == "zh":
        return _CHINESE_CHAPTER_OVERFLOW_UNITS
    return _ENGLISH_CHAPTER_OVERFLOW_WORDS


def _count_chapter_length_units(text: str, output_lang: str = "en") -> int:
    """Deterministic chapter length used by both initial drafts and rewrites.

    English uses conventional word-like tokens (hyphenated/contracted words count once).
    Chinese review mode counts each Han character plus any embedded Latin word once; this
    avoids the invalid assumption that Chinese prose is whitespace-tokenized.
    """
    if not isinstance(text, str):
        return 0
    latin_words = len(_LATIN_WORD_RE.findall(text))
    if output_lang == "zh":
        return len(_CJK_CHARACTER_RE.findall(text)) + latin_words
    return latin_words


def _iter_scope_values(value):
    if isinstance(value, str):
        if value.strip():
            yield value.strip()
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_scope_values(item)


def _collect_keyed_scope_values(record, accepted_keys: set[str]):
    if not isinstance(record, dict):
        return
    for key, value in record.items():
        if str(key).casefold() in accepted_keys:
            yield from _iter_scope_values(value)
        if isinstance(value, dict):
            yield from _collect_keyed_scope_values(value, accepted_keys)


def _normalize_entity_phrase(value: str) -> str:
    value = str(value or "").strip()
    # Planner proposals occasionally emit compact CamelCase names such as
    # MeridiaCentralReefDistrict. Writers naturally render those as words.
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value).casefold()
    value = re.sub(r"(?:['']s)\b", "", value)
    value = re.sub(r"[^\w\u3400-\u4dbf\u4e00-\u9fff]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _scope_phrase_variants(value: str) -> set[str]:
    """Make conservative aliases for parenthesized and '&'-joined stored names."""
    raw = str(value or "").strip()
    if not raw:
        return set()
    variants = {raw}
    # A stored value like "Neon Lantern (Midnight Showcase)" explicitly authorizes
    # both the canonical name and the parenthesized alias/event label.
    for outer, inner in re.findall(r"^(.*?)\s*\(([^()]+)\)\s*$", raw):
        if outer.strip():
            variants.add(outer.strip())
        if inner.strip():
            variants.add(inner.strip())
    for part in re.split(r"\s*(?:&|/|\||;)\s*", raw):
        if part.strip():
            variants.add(part.strip())
    return {_normalize_entity_phrase(item) for item in variants if _normalize_entity_phrase(item)}


def _build_named_entity_scope(
    character_records: list[dict] | None,
    environment_records: list[dict] | None,
) -> dict[str, set[str]]:
    """Build the exact post-generation allow-list from active structured records only."""
    character_phrases: set[str] = set()
    character_tokens: set[str] = set()
    location_phrases: set[str] = set()
    context_phrases: set[str] = set()

    for record in character_records or []:
        for value in _collect_keyed_scope_values(record, {"name", "aliases"}) or []:
            variants = _scope_phrase_variants(value)
            character_phrases.update(variants)
            # First-name, surname and explicit alias shorthand are all normal prose forms.
            for variant in variants:
                for token in variant.split():
                    if len(token) >= 2:
                        character_tokens.add(token)

    for record in environment_records or []:
        for value in _collect_keyed_scope_values(record, {"name", "aliases", "location"}) or []:
            variants = _scope_phrase_variants(value)
            location_phrases.update(variants)
            # Venue-qualified subspaces are naturally shortened in prose (for example,
            # "Neon Lantern Back Hallway" -> "Back Hallway").  Only derive the final
            # two-word form when the stored value ends in a location-type noun.
            for variant in variants:
                words = variant.split()
                if len(words) >= 3 and words[-1] in _LOCATION_SUFFIXES:
                    # Allow a venue-qualified location to omit its leading
                    # container name: Meridia Central Reef District ->
                    # Central Reef District / Reef District.
                    for width in range(2, min(4, len(words)) + 1):
                        location_phrases.add(" ".join(words[-width:]))

    # Descriptive fields from the active M_sub are authorized writing context,
    # not newly invented entities. Keep bounded normalized n-grams so a phrase
    # such as "Demo CDs" (copied from environment.minutia) is not mistaken for
    # a new proper name merely because it begins a sentence. Exact matching is
    # still required, so unrelated names such as "River City" remain blocked.
    descriptive_keys = {
        "description", "minutia", "atmosphere", "appearance", "personality",
        "identity_ability", "event_drive", "short_term_goal", "long_term_goal",
        "status", "current_plot_participation",
    }
    for record in list(character_records or []) + list(environment_records or []):
        for value in _collect_keyed_scope_values(record, descriptive_keys) or []:
            words = _normalize_entity_phrase(value).split()
            for width in range(2, min(6, len(words)) + 1):
                for start in range(0, len(words) - width + 1):
                    context_phrases.add(" ".join(words[start:start + width]))

    return {
        "character_phrases": character_phrases,
        "character_tokens": character_tokens,
        "location_phrases": location_phrases,
        "context_phrases": context_phrases,
        "location_tokens": {
            token
            for phrase in location_phrases
            for token in phrase.split()
            if len(token) >= 4 and token not in _LOCATION_SUFFIXES
        },
    }


def _strip_leading_prose_words(candidate: str) -> str:
    words = candidate.split()
    while len(words) > 1 and _normalize_entity_phrase(words[0]) in _SINGLETON_NON_NAME_WORDS:
        words.pop(0)
    return " ".join(words)


def _strip_trailing_coordinate_pronoun(candidate: str) -> str:
    """Remove a coordinated pronoun that the capitalized-span regex overcaptures.

    `Iona Rusk and I` must be checked as `Iona Rusk`, while `Jane Doe and I`
    must still leave `Jane Doe` available for rejection.
    """
    words = candidate.split()
    if len(words) >= 3 and _normalize_entity_phrase(words[-1]) in _TRAILING_COORDINATE_PRONOUNS:
        if _normalize_entity_phrase(words[-2]) in {"and", "or"}:
            words = words[:-2]
    return " ".join(words)


def _is_generic_all_caps_sign(candidate: str) -> bool:
    if not candidate.isupper():
        return False
    words = _normalize_entity_phrase(candidate).split()
    if not words or len(words) > 8:
        return False
    return (
        all(word in _GENERIC_SIGNAGE_WORDS for word in words)
        or any(word in _GENERIC_SIGNAGE_ACTION_WORDS for word in words)
    )


def _is_work_title_context(text: str, start: int) -> bool:
    prefix = text[max(0, start - 120):start]
    return bool(_WORK_TITLE_CUES_RE.search(prefix))


def _is_sentence_start(text: str, start: int) -> bool:
    prefix = text[:start]
    if not prefix:
        return True
    # Remove any run of whitespace/opening quotation/bracket characters. The
    # previous two-step rstrip left a space behind after removing a quote, so
    # `asked. "That's...` was incorrectly treated as mid-sentence.
    prefix = re.sub(r"[\s\"'""''([{]+$", "", prefix)
    return not prefix or prefix[-1] in ".!?\n…"


def _candidate_is_allowed(candidate: str, scope: dict[str, set[str]]) -> bool:
    normalized = _normalize_entity_phrase(candidate)
    if not normalized:
        return True
    if normalized in scope["character_phrases"] or normalized in scope["location_phrases"]:
        return True
    if normalized in scope.get("context_phrases", set()):
        return True
    if normalized in scope.get("location_tokens", set()):
        return True
    words = normalized.split()
    if len(words) > 1 and words[0] in _PERSON_ROLE_PREFIX_WORDS:
        titled_tail = " ".join(words[1:])
        if titled_tail in scope["character_phrases"]:
            return True
        if all(part in scope["character_tokens"] for part in words[1:]):
            return True
    pieces = [part for part in normalized.split() if part not in {"and", "of", "the"}]
    return bool(pieces) and all(part in scope["character_tokens"] for part in pieces)


def _find_out_of_scope_named_entities(
    text: str,
    scope: dict[str, set[str]],
) -> list[dict]:
    """Find likely named people/places without treating generic background nouns as names.

    Capitalized multi-token phrases are high-confidence proper-name candidates.  A
    singleton is rejected only with a person/action/vocative cue, which deliberately
    trades recall for fewer false positives at ordinary sentence starts.  Explicit work
    title contexts are exempt because titles are not characters or locations.
    """
    if not isinstance(text, str) or not text:
        return []
    violations: list[dict] = []
    seen: set[tuple[str, int]] = set()

    for match in _CAPITALIZED_SEQUENCE_RE.finditer(text):
        raw_candidate = match.group(0).strip()
        candidate = _strip_trailing_coordinate_pronoun(
            _strip_leading_prose_words(raw_candidate)
        )
        normalized = _normalize_entity_phrase(candidate)
        if not normalized or normalized in _GENERIC_CAPITALIZED_PHRASES:
            continue
        if _candidate_is_allowed(candidate, scope) or _is_work_title_context(text, match.start()):
            continue

        words = normalized.split()
        raw_words = re.findall(_CAPITALIZED_TOKEN, candidate)
        if not words or (len(words) == 1 and candidate.isupper()):
            continue
        if _is_generic_all_caps_sign(candidate):
            continue
        if len(words) == 1 and words[0] in _SINGLETON_NON_NAME_WORDS:
            continue
        if len(words) == 1 and words[0] in _GENERIC_ROLE_SINGLETONS:
            continue
        if (
            len(words) == 1
            and words[0] in _LOCATION_SUFFIXES
            and _is_sentence_start(text, match.start())
        ):
            continue
        if words[-1] in _NON_PERSON_ENTITY_SUFFIXES:
            continue

        kind = "named_entity"
        confidence = "ambiguous"
        if words[-1] in _LOCATION_SUFFIXES or words[0] in _LOCATION_PREFIXES:
            kind = "named_location"
            confidence = "high"
        elif words[0] in _PERSON_ROLE_PREFIX_WORDS:
            kind = "named_character"
            confidence = "high"
        elif len(raw_words) == 1:
            before = text[max(0, match.start() - 45):match.start()]
            after = text[match.end():match.end() + 45]
            # Object cues must be in the same clause. `\W*` used to cross a
            # period and quote, allowing the verb in `Alex asked. "That's...`
            # to classify "That's" as the verb's named-person object.
            before_word = re.search(r"([A-Za-z]+)[ \t,\"'""''([{]*$", before)
            after_word = re.match(r"\W*([A-Za-z]+)", after)
            sentence_start = _is_sentence_start(text, match.start())
            strong_person_cue = (
                bool(before_word and before_word.group(1).casefold() in _PERSON_OBJECT_CUES)
                or bool(re.search(r"\b(?:mr|mrs|ms|miss|dr|doctor|captain|officer|professor)\.?\s*$", before, re.I))
                or bool(re.search(r",\s*$", before) and re.match(r"\s*,", after))
            )
            # At a sentence boundary, capitalization says nothing about entityhood:
            # "She laughed", "People leaned", and "Midrange softened" are ordinary
            # prose.  Require an object/honorific/vocative cue there.  Away from a
            # boundary, a following person action is additional evidence.
            person_cue = strong_person_cue or (
                not sentence_start
                and bool(after_word and after_word.group(1).casefold() in _PERSON_ACTION_WORDS)
            )
            if not person_cue:
                # Ordinary capitalized sentence-start words are not useful NER evidence.
                continue
            kind = "named_character"
            confidence = "high"
        else:
            before = text[max(0, match.start() - 45):match.start()]
            after = text[match.end():match.end() + 45]
            before_word = re.search(r"([A-Za-z]+)[ \t,\"'""''([{]*$", before)
            after_word = re.match(r"\W*([A-Za-z]+)", after)
            if (
                bool(before_word and before_word.group(1).casefold() in _PERSON_OBJECT_CUES)
                or bool(after_word and after_word.group(1).casefold() in _PERSON_ACTION_WORDS)
            ):
                kind = "named_character"
                confidence = "high"

        key = (normalized, match.start())
        if key not in seen:
            seen.add(key)
            violations.append({
                "text": candidate,
                "kind": kind,
                "confidence": confidence,
                "start": match.start(),
            })

    # Minimal Chinese place detection: only explicit movement/location cues plus a strong
    # administrative/geographic suffix.  Generic rooms/corridors are intentionally absent.
    known_locations = scope.get("location_phrases", set())
    chinese_place_re = re.compile(
        r"(?:\u5728|\u53bb|\u5230|\u6765\u81ea|\u9a76\u5411|\u62b5\u8fbe)"
        r"([\u3400-\u4dbf\u4e00-\u9fff]{2,10}(?:\u5e02|\u57ce|\u9547|\u6751|\u533a|\u8857|\u8def|\u5df7|\u6e2f|\u673a\u573a|\u9152\u5e97|\u5267\u9662|\u516c\u56ed|\u5e7f\u573a|\u5927\u5b66|\u6cb3|\u6e56|\u5c71|\u6e7e|\u5c9b))"
    )
    generic_chinese_places = {"\u8fd9\u5ea7\u57ce\u5e02", "\u8fd9\u4e2a\u57ce\u5e02", "\u5c0f\u9547", "\u57ce\u5e02", "\u8857\u9053", "\u9053\u8def"}
    for match in chinese_place_re.finditer(text):
        candidate = match.group(1)
        normalized = _normalize_entity_phrase(candidate)
        if candidate in generic_chinese_places or normalized in known_locations:
            continue
        key = (normalized, match.start(1))
        if key not in seen:
            seen.add(key)
            violations.append({
                "text": candidate,
                "kind": "named_location",
                "confidence": "high",
                "start": match.start(1),
            })

    return violations


def _validate_chapter_output_contract(
    chapter_text: str,
    *,
    output_lang: str,
    named_entity_scope: dict[str, set[str]],
) -> dict:
    minimum, target_maximum, unit_label = _chapter_length_contract(output_lang)
    hard_maximum = _chapter_overflow_limit(output_lang)
    length_count = _count_chapter_length_units(chapter_text, output_lang)
    violations: list[dict] = []
    advisories: list[dict] = []
    if not isinstance(chapter_text, str) or not chapter_text.strip():
        violations.append({"code": "empty_output", "message": "Writer returned an empty chapter."})
    if length_count < minimum:
        violations.append({
            "code": "length_below_minimum",
            "message": f"{length_count} {unit_label}; required minimum is {minimum}.",
            "actual": length_count,
            "minimum": minimum,
            "target_maximum": target_maximum,
            "hard_maximum": hard_maximum,
            "unit": unit_label,
        })
    elif length_count > hard_maximum:
        violations.append({
            "code": "length_extreme_overflow",
            "message": (
                f"{length_count} {unit_label}; exceeds the {hard_maximum} "
                "runaway-output safety limit."
            ),
            "actual": length_count,
            "minimum": minimum,
            "target_maximum": target_maximum,
            "hard_maximum": hard_maximum,
            "unit": unit_label,
        })
    elif length_count > target_maximum:
        advisories.append({
            "code": "length_above_target",
            "message": (
                f"{length_count} {unit_label}; above the suggested {target_maximum} target "
                "but accepted because required story content takes priority."
            ),
            "actual": length_count,
            "target_maximum": target_maximum,
            "hard_maximum": hard_maximum,
            "unit": unit_label,
        })
    unknown_entities = _find_out_of_scope_named_entities(chapter_text, named_entity_scope)
    blocking_entities = [
        item for item in unknown_entities if item.get("confidence", "high") == "high"
    ]
    ambiguous_entities = [
        item for item in unknown_entities if item.get("confidence") == "ambiguous"
    ]
    if blocking_entities:
        rendered = ", ".join(item["text"] for item in blocking_entities[:12])
        violations.append({
            "code": "out_of_scope_named_entities",
            "message": f"Likely out-of-scope named character/location: {rendered}",
            "entities": blocking_entities,
        })
    if ambiguous_entities:
        rendered = ", ".join(item["text"] for item in ambiguous_entities[:12])
        advisories.append({
            "code": "ambiguous_capitalized_entities",
            "message": (
                "Ambiguous capitalized prose was retained without forcing a rewrite: "
                f"{rendered}"
            ),
            "entities": ambiguous_entities,
        })
    return {
        "ok": not violations,
        "output_lang": output_lang,
        "length_count": length_count,
        "length_unit": unit_label,
        "minimum": minimum,
        "maximum": target_maximum,
        "target_maximum": target_maximum,
        "hard_maximum": hard_maximum,
        "advisories": advisories,
        "unknown_entities": unknown_entities,
        "blocking_unknown_entities": blocking_entities,
        "ambiguous_entities": ambiguous_entities,
        "violations": violations,
    }


def _writer_repair_instruction(validation: dict, scope: dict[str, set[str]]) -> str:
    failures = "\n".join(
        f"- {item.get('message', item.get('code', 'invalid output'))}"
        for item in validation.get("violations", [])
    )
    characters = ", ".join(sorted(scope.get("character_phrases", set()))) or "(none)"
    locations = ", ".join(sorted(scope.get("location_phrases", set()))) or "(none)"
    return f"""
Your previous draft failed the deterministic publication contract:
{failures}

Rewrite the entire chapter now. Preserve every required plot beat and continuity fact, but fix every listed failure.
Allowed named character forms (including recorded aliases/shorthand): {characters}
Allowed named locations (including recorded profile.location values): {locations}
Unnamed generic background people/places and clearly introduced work titles are allowed. Do not add any other named or individually identifiable person or place.
Output only the complete replacement story body, with no critique, title, chapter number, notes, or word-count report.
""".strip()


def _generate_validated_chapter(
    writer,
    human_message: HumanMessage,
    *,
    output_lang: str,
    named_entity_scope: dict[str, set[str]],
    trace,
    event_index,
    chapter_no,
) -> tuple[str, dict, dict, int]:
    """Generate, then validate every direct or repair/rewrite candidate before commit."""
    max_repairs = _writer_repair_attempts()
    candidate = ""
    response: dict = {}
    validation: dict = {}

    for attempt in range(max_repairs + 1):
        if attempt == 0:
            messages = [human_message]
        else:
            messages = [
                human_message,
                AIMessage(content=candidate),
                HumanMessage(content=_writer_repair_instruction(validation, named_entity_scope)),
            ]
            trace.log(
                "chapter_write_repair_start",
                agent="WriterAgent",
                phase="chapter_writing",
                event_index=event_index,
                chapter_no=chapter_no,
                content=f"Writer contract repair {attempt}/{max_repairs}",
                payload={"repair_attempt": attempt, "previous_validation": validation},
                status="retry",
            )

        response = writer.invoke({"messages": messages})
        response_messages = response.get("messages", []) if isinstance(response, dict) else []
        if not response_messages:
            candidate = ""
        else:
            candidate = response_messages[-1].content
        validation = _validate_chapter_output_contract(
            candidate,
            output_lang=output_lang,
            named_entity_scope=named_entity_scope,
        )
        if validation["ok"]:
            trace.log(
                "chapter_write_validation_passed",
                agent="WriterAgent",
                phase="chapter_writing",
                event_index=event_index,
                chapter_no=chapter_no,
                content="Writer output passed deterministic length/entity checks",
                payload={"repair_attempts_used": attempt, "validation": validation},
                status="ok",
            )
            return candidate, response, validation, attempt

        has_more = attempt < max_repairs
        trace.log(
            "chapter_write_validation_failed",
            agent="WriterAgent",
            phase="chapter_writing",
            event_index=event_index,
            chapter_no=chapter_no,
            content="Writer output rejected before chapter commit",
            payload={"repair_attempt": attempt, "max_repairs": max_repairs, "validation": validation},
            status="retry" if has_more else "error",
        )
        if not has_more:
            summaries = "; ".join(item["message"] for item in validation["violations"])
            raise ChapterOutputContractError(
                f"Writer output contract failed after {max_repairs} repair attempt(s): {summaries}"
            )

    raise ChapterOutputContractError("Writer output contract failed without a candidate.")


def _find_plot_node_by_id(plot_manager, plot_id: int) -> dict | None:
    node = plot_manager.find_plot_by_id(plot_id)
    if node:
        return node
    for candidate in plot_manager.plots_graph.get("plots_node", []) or []:
        try:
            if int(candidate.get("id")) == int(plot_id):
                return candidate
        except (TypeError, ValueError):
            continue
    return None


def _plot_context_record(ref: str, node: dict | None) -> dict:
    if not isinstance(node, dict):
        return {"ref": ref, "missing_from_M_sub": True}
    record = {"ref": ref}
    for key in _PLOT_CONTEXT_FIELDS:
        if key in node:
            record[key] = deepcopy(node[key])
    return record


def _unresolved_plot_refs(event_payload: dict) -> list[str]:
    refs = (
        event_payload.get("seed_subgraph", {})
        .get("plots", {})
        .get("unresolved_threads", []) or []
    )
    return refs if isinstance(refs, list) else []


def _collect_referenced_plots(
    work_premise_id: str,
    event_payload: dict,
    chapter_payload: dict,
    *,
    mark_open_foreshadowing: bool = True,
) -> list[dict]:
    from tools.plot_graph_manager import PlotGraphManager

    plot_refs = chapter_payload.get("refs", {}).get("plots", []) or []
    if not isinstance(plot_refs, list):
        return []

    unresolved = {
        ref for ref in _unresolved_plot_refs(event_payload) if is_real_plot_ref(ref)
    } if mark_open_foreshadowing else set()
    manager = PlotGraphManager(work_premise_id)
    records: list[dict] = []
    seen: set[str] = set()
    for ref in plot_refs:
        if not is_real_plot_ref(ref) or ref in seen:
            continue
        seen.add(ref)
        plot_id = parse_plot_ref(ref)
        if plot_id is None:
            continue
        record = _plot_context_record(ref, _find_plot_node_by_id(manager, plot_id))
        if ref in unresolved:
            record["is_open_foreshadowing"] = True
        records.append(record)
    return records


def _collect_open_foreshadowing(
    work_premise_id: str,
    event_payload: dict,
    exclude_refs: set[str] | None = None,
) -> list[dict]:
    from tools.plot_graph_manager import PlotGraphManager

    threads_refs = _unresolved_plot_refs(event_payload)
    if not threads_refs:
        return []

    pa = PlotGraphManager(work_premise_id)
    out: list[dict] = []
    seen: set[str] = set()
    exclude_refs = exclude_refs or set()
    for ref in threads_refs:
        if not isinstance(ref, str) or ref in seen or ref in exclude_refs:
            continue
        seen.add(ref)
        if not is_real_plot_ref(ref):
            out.append({
                "ref": ref,
                "materialized": False,
                "note": "Planning-time thread only; no existing canon plot node is claimed.",
            })
            continue
        pid = parse_plot_ref(ref)
        if pid is None:
            continue
        record = _plot_context_record(ref, _find_plot_node_by_id(pa, int(pid)))
        record["is_open_foreshadowing"] = True
        out.append(record)
    return out


def _collect_relationship_history(work_premise_id: str, character_refs: list[str]) -> list[dict]:
    from tools.character_graph_manager import CharacterGraphManager
    char_ids: list[int] = []
    seen_ids: set[int] = set()
    work_meta = _load_work_meta(work_premise_id)
    temp_map = work_meta.get("character_temp_to_id", {})
    temp_map = temp_map if isinstance(temp_map, dict) else {}
    for ref in character_refs:
        cid = None
        if isinstance(ref, str) and is_real_character_ref(ref):
            cid = parse_character_ref(ref)
        elif isinstance(ref, str) and is_temp_character_ref(ref):
            cid = _coerce_character_work_id(temp_map.get(ref))
        if cid is None:
            continue
        cid = int(cid)
        if cid not in seen_ids:
            seen_ids.add(cid)
            char_ids.append(cid)

    if len(char_ids) < 2:
        return []

    ca = CharacterGraphManager(work_premise_id)
    name_by_id: dict[int, str] = {}
    for node in ca.characters_graph.get("characters_node", []):
        try:
            name_by_id[int(node.get("id"))] = node.get("name", "")
        except (TypeError, ValueError):
            continue

    out: list[dict] = []
    seen_pairs: set[tuple[int, int]] = set()
    for i, sid in enumerate(char_ids):
        for tid in char_ids[i + 1 :]:
            pair_key = (min(sid, tid), max(sid, tid))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            rel = (
                ca.find_two_characters_relationship_by_id(sid, tid)
                or ca.find_two_characters_relationship_by_id(tid, sid)
            )
            if not rel:
                continue
            out.append({
                "source": name_by_id.get(int(rel.get("source", sid)), str(rel.get("source", sid))),
                "target": name_by_id.get(int(rel.get("target", tid)), str(rel.get("target", tid))),
                "current_type": rel.get("current_type", ""),
                "change_history": rel.get("change_history", []) or [],
            })
    return out


def _legacy_event_to_single_chapter_payload(chapter_outline: str) -> tuple[dict, dict]:
    raw_event = extract_json.extract_json(llm_output=chapter_outline)
    event_payload = event_schema.normalize_event_payload(raw_event)
    if not event_payload:
        raise ValueError("Could not parse the legacy Event output.")
    chapter_payload = event_schema.get_chapter_payload(event_payload, 1)
    if not chapter_payload:
        raise ValueError("The legacy Event output has no chapter information.")
    return event_payload, chapter_payload


_OUTPUT_LANG_DIRECTIVES = {
    "zh": "\n            IMPORTANT: Write the entire chapter body in fluent, native Chinese, with 3000-5000 Chinese characters (embedded Latin words count as one unit each). You may keep character / place names in their original English form to avoid awkward transliteration.",
    "en": "",
}


def write_chapter_txt(
    premise_id,
    premise,
    event_payload: dict | None = None,
    chapter_payload: dict | None = None,
    chapter_outline: str = "",
    pre_chapter_text: str = "",
    trace_logger=None,
    event_index=None,
    enable_foreshadowing_injection: bool = True,
    enable_hij_injection: bool = True,
    output_lang: str = "en",
):
    trace = get_trace_logger(trace_logger)
    if event_payload is None or chapter_payload is None:
        event_payload, chapter_payload = _legacy_event_to_single_chapter_payload(chapter_outline)
    else:
        event_payload = event_schema.normalize_event_payload(event_payload)
        if not event_payload:
            raise ValueError("event_payload structure is invalid; cannot write chapter.")

    chapter_struct = {
        "chapter_goal": chapter_payload.get("chapter_goal", ""),
        "chapter_conflict": chapter_payload.get("chapter_conflict", ""),
        "chapter_turn": chapter_payload.get("chapter_turn", ""),
        "chapter_hook_out": chapter_payload.get("chapter_hook_out", ""),
    }
    must_plots = {
        "chapter_no": chapter_payload.get("chapter_no"),
        "chapter_struct": chapter_struct,
        "expected_deltas": chapter_payload.get("expected_deltas", {}),
    }
    trace.log(
        "chapter_write_start",
        agent="WriterAgent",
        phase="chapter_writing",
        event_index=event_index,
        chapter_no=chapter_payload.get("chapter_no"),
        content=chapter_payload.get("pure_plot", ""),
        payload={
            "refs": chapter_payload.get("refs", {}),
            "chapter_struct": chapter_struct,
            "expected_deltas": chapter_payload.get("expected_deltas", {}),
        },
    )

    (
        character_info,
        env_info,
        character_scope_records,
        environment_scope_records,
    ) = _resolve_chapter_context(
        premise_id,
        event_payload,
        chapter_payload,
        include_scope_records=True,
    )
    named_entity_scope = _build_named_entity_scope(
        character_scope_records,
        environment_scope_records,
    )
    print('>>> Character information used for writing:\n', character_info)
    print('>>> Environment information used for writing:\n', env_info)

    referenced_plots = _collect_referenced_plots(
        premise_id,
        event_payload,
        chapter_payload,
        mark_open_foreshadowing=enable_foreshadowing_injection,
    )
    referenced_plot_refs = {
        item.get("ref")
        for item in referenced_plots
        if isinstance(item, dict) and isinstance(item.get("ref"), str)
    }
    referenced_plot_block = ""
    if referenced_plots:
        referenced_plot_block = (
            "\n[Referenced Canon Plot Context]: "
            + json.dumps(referenced_plots, ensure_ascii=False)
            + "\n  ⚠️ Every p_K record above is an already-materialized fact from C_P, not a new plot proposal. "
              "Use its overview/details as continuity constraints and do not rewrite established facts."
            + ("\n  Any record marked `is_open_foreshadowing=true` is both a prior canon fact and a thread "
               "selected by this chapter; give it an explicit, reader-visible payoff."
               if enable_foreshadowing_injection else "")
        )

    open_foreshadowing: list[dict] = []
    foreshadowing_block = ""
    if enable_foreshadowing_injection:
        open_foreshadowing = _collect_open_foreshadowing(
            premise_id,
            event_payload,
            exclude_refs=referenced_plot_refs,
        )
        if open_foreshadowing:
            foreshadowing_block = (
                "\n[Open Foreshadowing Threads]: " + json.dumps(open_foreshadowing, ensure_ascii=False) +
                "\n  ⚠️ These are foreshadowing threads / hooks that have already been planted but not yet paid off. In this chapter you must:"
                "\n    a) If any of these refs appear in chapter_payload.refs.plots, **explicitly pay them off** in this chapter (make the reader feel the payoff clearly, not glossed over in a single sentence);"
                "\n    b) Otherwise, **at minimum do not contradict them**, and you may plant additional details for later chapters to extend;"
                "\n    c) Silently dropping any thread is strictly forbidden."
            )

    relationship_history: list[dict] = []
    relationship_block = ""
    if enable_hij_injection:
        relationship_history = _collect_relationship_history(
            premise_id,
            chapter_payload.get("refs", {}).get("characters", []) or [],
        )
        if relationship_history:
            relationship_block = (
                "\n[Character Relationship History (H_ij)]: " + json.dumps(relationship_history, ensure_ascii=False) +
                "\n  ⚠️ These are the current relationship states and evolution histories among the characters appearing in this chapter. Character interactions in this chapter must:"
                "\n    a) Be consistent with `current_type` (do not portray them as more intimate or more distant than what is recorded);"
                "\n    b) Allow past events in `change_history` to surface naturally **through dialogue / memory / action beats** at appropriate moments — do not ignore the past;"
                "\n    c) Any change in relationship must be triggered by a concrete event in this chapter; abrupt relationship leaps are forbidden."
            )

    trace.log(
        "agent_decision",
        agent="WriterAgent",
        phase="chapter_context",
        event_index=event_index,
        chapter_no=chapter_payload.get("chapter_no"),
        content="Chapter context resolved",
        payload={
            "characters_info": character_info,
            "environments_info": env_info,
            "refs": chapter_payload.get("refs", {}),
            "referenced_canon_plots": referenced_plots,
            "open_foreshadowing": open_foreshadowing,
            "relationship_history": relationship_history,
            "enable_foreshadowing_injection": enable_foreshadowing_injection,
            "enable_hij_injection": enable_hij_injection,
            "writer_named_entity_scope": {
                key: sorted(values) for key, values in named_entity_scope.items()
            },
        },
    )

    lang_directive = _OUTPUT_LANG_DIRECTIVES.get(output_lang, "")
    minimum, maximum, unit_label = _chapter_length_contract(output_lang)
    length_instruction = (
        f"aim for {minimum}-{maximum} {unit_label}; fewer than {minimum} will be rejected, "
        "but a modest overrun is acceptable when needed to preserve required plot and continuity"
    )
    if pre_chapter_text == "":
        wa = build_writer_agent(first_writing=True)
        user_prompt = f"""
            [Chapter Summary]: {chapter_payload.get("pure_plot", "")}
            [Must Plots]: {json.dumps(must_plots, ensure_ascii=False)}
            [Character Profile]: {character_info}
            [Environment Information]: {env_info}{referenced_plot_block}{foreshadowing_block}{relationship_block}
            Use the overall premise of the novel as reference: {premise}
            Begin the task now. Output **only** the body of the story — no title, no chapter number. The output should be a polished chapter ready for publication ({length_instruction}).{lang_directive}
            """
    else:
        wa = build_writer_agent(first_writing=False)
        user_prompt = f"""
            [Previous Chapter Text Reference]: {pre_chapter_text}
            [Chapter Summary]: {chapter_payload.get("pure_plot", "")}
            [Must Plots]: {json.dumps(must_plots, ensure_ascii=False)}
            [Character Profile]: {character_info}
            [Environment Information]: {env_info}{referenced_plot_block}{foreshadowing_block}{relationship_block}
            Overall novel premise for reference: {premise}
            Begin the task now. Output **only** the body of the story — no title, no chapter number. The output should be a polished chapter ready for publication ({length_instruction}).{lang_directive}
            """

    human_message = HumanMessage(content=user_prompt)
    chapter_text, res, validation, repair_attempts_used = _generate_validated_chapter(
        wa,
        human_message,
        output_lang=output_lang,
        named_entity_scope=named_entity_scope,
        trace=trace,
        event_index=event_index,
        chapter_no=chapter_payload.get("chapter_no"),
    )
    print(
        ">>> Writer agent output passed validation:",
        f"{validation['length_count']} {validation['length_unit']}; ",
        f"repair_attempts={repair_attempts_used}",
    )
    trace.log(
        "chapter_write_done",
        agent="WriterAgent",
        phase="chapter_writing",
        event_index=event_index,
        chapter_no=chapter_payload.get("chapter_no"),
        content=chapter_text,
        payload={
            "message_count": len(res.get("messages", [])) if isinstance(res, dict) else None,
            "validation": validation,
            "repair_attempts_used": repair_attempts_used,
        },
        status="ok",
    )
    return {
        "chapter_text": chapter_text,
        "characters_info": character_info,
        "envs_info": env_info,
        "event": event_payload.get("event_meta", {}).get("abstract", ""),
        "chapter_payload": chapter_payload,
    }

