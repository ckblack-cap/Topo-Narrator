import ast
import re
import json
from copy import deepcopy
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
from sentence_transformers import SentenceTransformer, util

from data import characters_graph_data, set_characters_graph_data
from agents import AgentGlobalConfig, DEFAULT_MODEL_NAME
from tools.graph_refs import parse_character_ref

# client = AgentGlobalConfig.DeepSeekCLIENT
client = AgentGlobalConfig.GPTCLIENT



VALID_CHARACTER_ATTRIBUTES = {
    "id", "name", "aliases", "description", "short_term_goal", "long_term_goal",
    "importance", "current_plot_participation", "status", "created_chapter", "updated_chapter"
}

VALID_RELATIONSHIP_ATTRIBUTES = {
    "source", "target", "current_type", "change_history"
}


_NO_CHANGE_SENTINELS = {
    "no change",
    "no changes",
    "unchanged",
    "same as before",
    "not changed",
    "n/a",
    "not applicable",
    "\u65e0\u53d8\u5316",
    "\u6ca1\u6709\u53d8\u5316",
    "\u4e0d\u53d8",
    "\u65e0\u9700\u66f4\u65b0",
    "\u4fdd\u6301\u4e0d\u53d8",
}


def _is_no_change_sentinel(value) -> bool:
    """Return True only for an explicit scalar no-op marker from the LLM.

    Passage hot updates occasionally emit values such as ``"No change."`` for
    stable fields.  Persisting that marker destroys the canonical profile, so
    treat it as an omitted field.  This intentionally applies only to complete
    scalar values; ordinary prose containing the words "no change" is kept.
    """
    if not isinstance(value, str):
        return False
    normalized = re.sub(r"[\s.!?\u3002\uFF01\uFF1F]+$", "", value.strip()).casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized in _NO_CHANGE_SENTINELS


def _parse_graph_json_payload(raw_text: str) -> dict:
    """Parse a graph object without corrupting apostrophes inside JSON strings.

    The previous implementation globally replaced every ASCII apostrophe with a
    double quote before ``json.loads``.  A completely valid value such as
    ``"artist's mirror"`` was therefore changed into invalid JSON and triggered
    an unnecessary, lossy LLM repair call.  Prefer strict JSON as emitted; only
    fall back to ``ast.literal_eval`` when the *whole payload* is a Python-style
    dict literal.
    """
    if not isinstance(raw_text, str):
        raise TypeError("graph JSON payload must be text")
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as json_error:
        try:
            parsed = ast.literal_eval(cleaned)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            raise json_error
    if not isinstance(parsed, dict):
        raise ValueError("graph JSON payload must decode to an object")
    return parsed


def _parse_character_query_id(value) -> int | None:
    """Accept the public ``c_K`` form as well as legacy integer/numeric IDs."""
    if isinstance(value, bool):
        return None
    parsed = parse_character_ref(value)
    if parsed is not None:
        return int(parsed)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _relationship_history_error(value) -> str | None:
    if not isinstance(value, list):
        return "change_history must be an array"
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            return f"change_history[{index}] must be an object"
        for field in ("from", "to", "reason"):
            item = entry.get(field)
            if not isinstance(item, str) or not item.strip():
                return f"change_history[{index}].{field} must be nonempty text"
    return None


class CharacterGraphManager:

    def __init__(self, premise_id):
        self.premise_id = premise_id
        self.model_name = DEFAULT_MODEL_NAME
        self.history = []
        self.graph_history = []
        self.characters_graph = characters_graph_data(premise_id)
        self.graph_history.append(deepcopy(self.characters_graph))
        self.embedding_model = None
        self.vector_cache = {}
        self.vector_keys = []
        self.vector_matrix = None
        self.is_vector_stale = True
        self.last_passage_update_ok = None
        self.last_passage_update_error = None


    ##############################################################################
    ##############################################################################


    def extract_passage_characters(self, passage: str):
        task_prompt = f"""
            # Role: You are a Senior Novel Narrative Logic Expert.
            # Task: Analyze the novel chapter and construct a concise and critical [Character Relationship Graph].
            # Core Goal: Extract only the "Core Nodes" that have a substantial impact on the plot development, strictly filtering out background characters and plot devices.
            
            # 1. Character Extraction Standards (Must meet ALL the following conditions to be recorded):
                - **Centrality**: The character must have specific dialogue or actions, and their behavior must **directly drive a plot twist** or **have a substantial impact on main characters**.
                - **Tangibility**:
                    - Includes: Humans, sentient ghosts/creatures, key plot items (e.g., a magic mirror that can speak), key equipment, and singular artifacts.
                    - Excludes: Ordinary pets, static furniture/props.
                - **Non-Collectivity**: Strictly forbid recording vague collective terms like "netizens," "crowd," "police," or "medical staff," unless a specific individual stands out as a key character.
            
            # 2. ⛔️ Exclusion List (Do NOT record if the following applies):
                - **Background/Tool Characters**: Such as a nurse who only speaks one line, a colleague who appears only once, a roadside driver, or a nameless patient.
                - **Atmosphere Group**: Such as "onlookers" or "forum netizens," which are merely part of the environmental description.
            
            # 3. Naming Conventions:
                - **Prioritize Original Names**: If a full name or a fixed screen name/handle (e.g., "Thunder Dharma King") appears in the text, prioritize using it.
                - **Naming Key Nameless Characters**: Applies ONLY to **Core Villains** or **Key Mysterious Figures**. If they are nameless, assign a highly distinctive alias.
                    - ❌ Bad Example: "Old Man", "Patient" (Too vague, easily confused).
                    - ✅ Good Example: "Resurrected Old Man in Morgue", "Young Patient Attacking Nurse".
            
            # 4. Relationship Extraction Standards:
                - Record only **Substantive Interactions** (e.g., Attack, Cry for help, Deceive, Ally).
                - Ignore weak interactions (e.g., Seeing, Passing by, Routine medical inquiry).
            
            # Thinking Steps (Must be explicitly output):
                1. **Initial Screening**: List all noun entities appearing in the text.
                2. **Filtering**: Eliminate tool characters based on the [Exclusion List] and [Centrality] standards (Please explain the reason for elimination, e.g., "The Head Nurse only acts as an information conduit with no independent plotline, excluded").
                3. **Naming**: Standardize the names of the remaining core characters.
                4. **Construction**: Generate the final relationship list.
            
            # Final Answer Format(<name> Json List</name>):
            <name>["Character A name", "Character B name", ...]</name><rel>[["Character A", "Substantive Relationship", "Character B"]]</rel>
            
            # Chapter Content:
            {passage}
        """
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": task_prompt}]
        )
        resp_content = response.choices[0].message.content
        print('Character extraction response:\n', resp_content)
        res = {'names': [], 'relationships': ''}
        pattern = r'<name>(.*?)</name>'
        matches = re.findall(pattern, resp_content, re.DOTALL)
        if matches:
            characters_content = matches[0].strip()
            try:
                characters_list = json.loads(characters_content)
            except json.JSONDecodeError as e:
                print('Character-name JSON is invalid; starting repair')
                repair_prompt = f"""
                   You are a strict JSON syntax repair specialist. Follow every rule below:

                   ### Task requirements
                   1. Given a JSON string with syntax errors, fix every violation so a standard JSON parser can parse the result.
                   2. Fixes include, but are not limited to:
                      - remove trailing commas from objects and arrays;
                      - replace single quotes with double quotes (JSON requires double quotes);
                      - enclose every property name in double quotes;
                      - add missing closing braces or brackets;
                      - fix invalid escapes, newlines, and other special characters;
                      - preserve all original data (text, numbers, and structure); change only syntax errors.

                   ### Output rules (follow exactly)
                   1. Output only the repaired JSON text, with no explanations, Markdown fences, comments, or unnecessary whitespace.
                   2. Do not wrap the result in ```json or ``` fences.
                   3. Do not add any introductory or unrelated text.
                   4. Keep the JSON compact and valid, with correct array and object nesting.

                   ### Malformed JSON to repair
                   {characters_content}""".strip()

                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system",
                         "content": "You are a strict JSON syntax repair specialist. Output only valid JSON, with no other text."},
                        {"role": "user", "content": repair_prompt}
                    ],
                    temperature=0.0,
                )

                corrected_json_str = response.choices[0].message.content.strip()
                if corrected_json_str.startswith("```json"):
                    corrected_json_str = corrected_json_str[7:-3].strip()

                try:
                    characters_list = json.loads(corrected_json_str)
                    print("Character-name JSON repair succeeded; parsing complete")

                except json.JSONDecodeError as e2:
                    raise Exception(f"Parsing still failed after LLM repair: {e2}, malformed content: {corrected_json_str}")
                    exit()

            res['names'] = characters_list
            pattern = r'<rel>(.*?)</rel>'
            matches = re.findall(pattern, resp_content, re.DOTALL)
            if matches:
                rel_content = matches[0].strip().replace("'", '"')
                res['relationships'] = rel_content
            return res
        return {}

    def _parse_json_safe(self, content):
        try:
            return json.loads(content)
        except:
            match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                content = match.group(1)
            else:
                match = re.search(r'<graph>(.*?)</graph>', content, re.DOTALL)
                if match:
                    content = match.group(1)
            try:
                return json.loads(content.strip())
            except:
                try:
                    return json.loads(content.strip().replace("'", '"'))
                except:
                    return None

    def align_character_names(self, new_names: list, passage_summary: str) -> dict:
        print(f"--- Aligning character entities; new names to analyze: {new_names} ---")

        candidate_pool = {}

        name_to_node_map = {node['name']: node for node in self.characters_graph.get('characters_node', [])}

        for new_name in new_names:
            try:
                semantic_candidates = self.search_character(new_name, top_k=3)
            except Exception as e:
                print(f"Vector search warning: {e}")
                semantic_candidates = []

            keyword_candidates = []
            for node in self.characters_graph.get('characters_node', []):
                if new_name in node['name'] or node['name'] in new_name:
                    keyword_candidates.append(node)
                elif 'aliases' in node and any(new_name in alias for alias in node['aliases']):
                    keyword_candidates.append(node)

            all_suspects = []

            for res in semantic_candidates:
                node = name_to_node_map.get(res['name'])
                if node: all_suspects.append(node)

            all_suspects.extend(keyword_candidates)

            for node in all_suspects:
                c_id = str(node['id'])
                if c_id not in candidate_pool:
                    name = node['name']
                    aliases = f", Aliases:{','.join(node.get('aliases', []))}" if node.get('aliases') else ""
                    desc_str = node.get('description', '')[:100].replace('\n', ' ')
                    status_str = f", status:{node.get('status', 'unknown')}" if node.get('status') else ""

                    info_str = f"ID:{c_id} [{name}]{aliases}\n   - features: {desc_str}{status_str}..."
                    candidate_pool[c_id] = info_str

        if not candidate_pool:
            return {name: None for name in new_names}

        existing_roles_text = "\n".join(candidate_pool.values())

        prompt = f"""        
        # Task: Detective-style Character Entity Resolution
            1. Determine if the [Extracted Names from New Chapter] refer to any character in the [Candidate List].
            2. If they refer to an existing character, determine if it is merely a nickname (`ALIAS`) or a formal identity reveal/name change (`RENAME`).
            
        # Core Logic:
            I have pre-screened the most likely "suspects" (candidates) for you using a retrieval algorithm.
            Please combine the context from the passage to reason whether the new name corresponds to one of these candidates.
            
        # Passage Segment (Context):
            {passage_summary}
            
        # Candidate/Suspect List:
            {existing_roles_text}
            
        # Extracted Names from New Chapter:
            {json.dumps(new_names, ensure_ascii=False)}
            
        # Judgment Rules:
            1. **Nickname/Abbreviation**: E.g., "Xun'er" matches "Xiao Xun'er".
            2. **Feature/Behavior Overlap**: If the behavioral traits of "Knocking Ghost" align highly with candidate "Morgue Old Man" (e.g., both wear burial clothes), treat them as the same person.
            3. **New Character**: If the new name does not match any candidates, output null.
            
        # Processing Rules:
            **If Matched:** Set `"id"` to the corresponding character's ID.
            1. **ALIAS**: For informal names, such as nicknames, titles, or vague references (e.g., "Dr. Ding") that are not definitive identity reveals, mark the action as `ALIAS`.
            2. **RENAME**: For identity reveals, such as a character previously known as "Masked Man" being formally revealed as "Zhang San" in the new chapter, mark the action as `RENAME` (indicating a primary name update).
            
            **If Unmatched:**
            3. **New Character**: Mark pure new characters as `null`.
            
        # Output Format (JSON):
            {{
                "Xun'er": {{ "id": 101, "action": "ALIAS" }},  // Just a nickname, add to aliases
                "Knocking Ghost": {{ "id": 105, "action": "RENAME" }}, // Identity reveal, suggest changing primary name
                "Passerby A": null // New character
            }}
        """

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1
            )
            res_content = response.choices[0].message.content

            llm_mapping = self._parse_json_safe(res_content)

            for new_name, info in llm_mapping.items():
                if isinstance(info, dict) and info.get("action") == "ALIAS":
                    target_id = info.get("id")

                    for node in self.characters_graph.get('characters_node', []):
                        if int(node.get("id", -1)) == int(target_id):
                            if "aliases" not in node:
                                node["aliases"] = []

                            if new_name not in node["aliases"] and new_name != node["name"]:
                                node["aliases"].append(new_name)
                                print(f"Added alias for ID:{target_id}: {new_name}")
                            # break


                elif isinstance(info, dict) and info.get("action") == "RENAME":
                    target_id = info.get("id")
                    for node in self.characters_graph.get('characters_node', []):
                        if int(node.get("id", -1)) == int(target_id):
                            old_name = node.get("name")

                            if old_name != new_name:
                                if "aliases" not in node:
                                    node["aliases"] = []
                                if old_name and old_name not in node["aliases"]:
                                    node["aliases"].append(old_name)

                                node["name"] = new_name
                                print(f"Renamed character identity (ID:{target_id}): '{old_name}' -> '{new_name}'")

                                rels = self.characters_graph.get('characters_relationship', [])
                                if rels and len(rels) > 0:
                                    first_source = rels[0].get('source')
                                    if isinstance(first_source, str):
                                        count = 0
                                        for rel in rels:
                                            if rel.get('source') == old_name:
                                                rel['source'] = new_name
                                                count += 1
                                            if rel.get('target') == old_name:
                                                rel['target'] = new_name
                                                count += 1
                                        if count > 0:
                                            print(f"   -> Updated {count} name-based relationship edges")
                            # break
            return llm_mapping
        except Exception as e:
            print(f"Alignment failed: {e}")
            return {}

    def update_graph_by_passage2(self, passage: str, event):
        passage_characters = self.extract_passage_characters(passage)
        raw_names = passage_characters['names']
        raw_relations = passage_characters['relationships']
        print(f"1. Extracted original character names: {raw_names}")


        align_map = self.align_character_names(raw_names, passage)
        if not align_map:
            print('Alignment returned no results; proceed with caution!')
            align_map = {}

        name_to_id_lookup = {}
        related_ids = []

        for name, info in align_map.items():
            if info and isinstance(info, dict) and info.get('id'):
                c_id = int(info['id'])
                name_to_id_lookup[name] = c_id
                related_ids.append(c_id)
            else:
                name_to_id_lookup[name] = None


        current_info = []
        for node in self.characters_graph['characters_node']:
            if int(node.get('id', -1)) in related_ids:
                current_info.append(node)

        current_relation = []
        for edge in self.characters_graph['characters_relationship']:
            if int(edge['source']) in related_ids and int(edge['target']) in related_ids:
                current_relation.append(edge)


        characters_graph_example = '''
        {
        "characters_node":[{
          "name": "Character Name",
          "aliases": ["Alias 1", "Nickname", "Title/Honorific"], 
          "description": "[Global Profile] Intrinsic attributes of the character. Includes: background, personality traits, core abilities, physical appearance. Note: Only modify this field when major transformations occur (e.g., plastic surgery, personality corruption/darkening, identity revelation).",
          "short_term_goal": "[Current Intent] Specific goals the character wants to achieve in this chapter or the near future.",
          "long_term_goal": "[Ultimate Vision] Long-term pursuits spanning the entire book; update frequency is very low.",
          "importance": "Protagonist | Supporting Character | Extra/Mob",
          "current_plot_participation": "Key Figure | Participant | Non-participant", 
          "status": "[Real-time State] Current physiological/psychological state (e.g., severe injury, coma, fear, excitement, imprisoned)."
        }],
        "characters_relationship": [
        {
          "source": "Character A Name", 
          "target": "Character B Name",  
          "current_type": "Current Relationship (e.g., Ally/Enemy/Lover/Master-Apprentice)",
          "change_history": [
            {
              "from": "Old Relationship",
              "to": "New Relationship",
              "reason": "Summary of the specific event causing the relationship change"
            }
          ]
        }]
        }'''

        pattern = r'<character>(.*?)</character>'
        matches = re.findall(pattern, event, re.DOTALL)

        if matches:
            new_character_info = matches[0].strip().replace("'", '"')
            print('Extracted new character information from Event: ', new_character_info)
        else:
            print('Failed to extract new character information from Event!')
            new_character_info = "The author has not provided any information on the newly added characters. Characters not included in the character profile should be regarded as new ones. Please keep this in mind, but exercise extreme caution when designating new characters as protagonists!"

        task_prompt = f"""
        # Role
        You are a Senior **Novel Character Graph Architect**. Your core capability is to precisely extract character information from text and maintain a dynamically evolving relationship network.

        # Goal
        Read the [New Chapter] and, based on the [Existing Graph Info], generate a subgraph containing **incremental updates**.

        # Input Data
        1. **Reference(It may not be accurate) - Pre-extracted Character Names**: {raw_names};
        2. **Reference(It may not be accurate) - Pre-extracted Temporary Relations**: {raw_relations};
        
        3. **Existing - Related Character Profiles**: {current_info};
        4. **Existing - Related Character Relationships**: {current_relation};
           (If a character's information is missing here, it means this character is appearing for the first time in this chapter and needs to be created.)
        5. **New key character information provided by the author**:{new_character_info};
        6. **Text - New Chapter Content**: {passage};

        # Output Schema
        Output the updated subgraph. Please strictly adhere to the following JSON structure:
        {characters_graph_example}

        # Updating Rules (Core Principles)

        1. **Node Updates (Nodes)**:
           - **status vs description**: 
             - `status` is **Transient** (e.g., left arm fractured, spiritual power exhausted, angry). Please update this field actively.
             - `description` is **Stable** (e.g., personality, appearance, hobbies, characteristics, age, identity, and other stable traits). 
               - *New Character Creation*: When establishing a new character, provide a comprehensive descriptive grasp.
               - *Old Character Modification*: Modify only when there are major physical or psychological changes (including but not limited to permanent injury, death, drastic mood shifts, breaking/establishing important relationships), new settings added, or significant setting changes.
           - **short_term_goal**: The character's next intended steps.

        2. **Relationship Updates (Edges)**:
           - Record only **Substantive** relationships (Kinship, Subordinate, Hostile, Emotional). Ignore temporary, meaningless interactions (e.g., "asking for directions").
           - **Change History**: When `current_type` undergoes a fundamental change (e.g., from "Stranger" to "Ally"), you must append a record to `change_history` explaining the `reason`.

        3. **Format Constraints**:
           - `source` and `target` in relationships must strictly use the character's **name** (Primary/Canonical Name). Do NOT use IDs or Aliases.

        # Workflow (Thinking Steps)
        1. **Read & Align**: Read the text, identify characters appearing in the text, and match them with existing character profiles.
        2. **Gap Analysis**: 
           - Did this character get injured in this chapter? -> Update `status`.
           - Did a relationship break? -> Update `relationship`.
        3. **New Discovery**: Identify newly appearing characters in the text and create profile information for them.You need refer to the additional New key character information provided by the author.
        4. **Generate Subgraph**: Organize all **changed old characters** and **newly added characters** (along with corresponding relationships) into JSON.

        # Execute
        Based on the rules above, please output the updated character graph subgraph:

        # Output Format: JSON format character graph subgraph structure wrapped in tags, Format Example: <graph>""" + """{"characters_node":[...], "characters_relationship": [...]}</graph>
        """
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": task_prompt}]
        )
        resp_content = response.choices[0].message.content
        print('3.Character graph update response:\n', resp_content)
        pattern = r'<graph>(.*?)</graph>'
        matches = re.findall(pattern, resp_content, re.DOTALL)

        if matches and False:
            updated_subgraph_content = matches[0].strip().replace("'", '"')

            print('3.Subgraph matching\n', updated_subgraph_content)
            subgraph = json.loads(updated_subgraph_content)

            graph_characters_name_index = self.find_all_characters_name_index()

            for subgraph_node in subgraph['characters_node']:
                if subgraph_node['name'] in graph_characters_name_index.keys():
                    res = self.update_character_info(subgraph_node['name'], subgraph_node)
                    if "Error" in res:
                        print(res)
                        return []

                else:
                    res = self.add_characters([subgraph_node])
                    if "Error" in res:
                        print(res)
                        return []

            for subgraph_edge in subgraph['characters_relationship']:
                if self.find_two_characters_relationship(subgraph_edge['source'], subgraph_edge['target'], exact=True):
                    res = self.update_relationship_by_edge(subgraph_edge)
                    if "Error" in res:
                        print(res)
                        return []
                else:
                    res = self.add_relationship(subgraph_edge)
                    if "Error" in res:
                        print(res)
                        return []

            if not self.save_characters_graph():
                raise OSError(f"Failed to save character graph: {self.premise_id}")
            self.graph_history.append(deepcopy(self.characters_graph))
            print('Updated character graph:\n', self.characters_graph)
            print('————————Character graph update complete!————————')
            names = []
            for character in subgraph['characters_node']:
                names.append(character['name'])
            self.is_vector_stale = True
            return names
            # return True
        # return []

        if matches:
            updated_subgraph_content = matches[0].strip()
            try:
                subgraph = _parse_graph_json_payload(updated_subgraph_content)
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed at position {e.pos}, starting LLM-assisted repair")
                task_prompt = f"""
                You are a strict JSON syntax repair specialist. Follow every rule below:

                ### Task requirements
                1. Given a JSON string with syntax errors, fix every violation so a standard JSON parser can parse the result.
                2. Fixes include, but are not limited to:
                   - remove trailing commas from objects and arrays;
                   - replace single quotes with double quotes (JSON requires double quotes);
                   - enclose every property name in double quotes;
                   - add missing closing braces or brackets;
                   - fix invalid escapes, newlines, and other special characters;
                   - preserve all original data (text, numbers, and structure); change only syntax errors.

                ### Output rules (follow exactly)
                1. Output only the repaired JSON text, with no explanations, Markdown fences, comments, or unnecessary whitespace.
                2. Do not wrap the result in ```json or ``` fences.
                3. Do not add any introductory or unrelated text.
                4. Keep the JSON compact and valid, with correct array and object nesting.

                ### Malformed JSON to repair
                {updated_subgraph_content}""".strip()

                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system",
                         "content": "You are a strict JSON syntax repair specialist. Output only valid JSON, with no other text."},
                        {"role": "user", "content": task_prompt}
                    ],
                    temperature=0.0,
                )

                corrected_json_str = response.choices[0].message.content.strip()
                if corrected_json_str.startswith("```json"):
                    corrected_json_str = corrected_json_str[7:-3].strip()

                try:
                    subgraph = json.loads(corrected_json_str)
                    print("JSON repair succeeded; parsing complete")

                except json.JSONDecodeError as e2:
                    raise Exception(f"Parsing still failed after LLM repair: {e2}, malformed content: {corrected_json_str}")
                    exit()




            current_max_id = self.get_max_character_id()

            for subgraph_node in subgraph.get('characters_node', []):
                name = subgraph_node['name']

                target_id = name_to_id_lookup.get(name)
                if target_id is None:
                    target_id = self.find_character_id_by_name(name)

                if target_id:
                    subgraph_node['id'] = target_id
                    res = self.update_character_info(target_id, subgraph_node)
                    if "Error" in res: print(res)

                    name_to_id_lookup[name] = target_id
                else:
                    current_max_id += 1
                    subgraph_node['id'] = current_max_id

                    res = self.add_characters([subgraph_node])
                    if "Error" in res: print(res)

                    name_to_id_lookup[name] = current_max_id


            for subgraph_edge in subgraph.get('characters_relationship', []):
                s_name = subgraph_edge.get('source')
                t_name = subgraph_edge.get('target')



                s_id = name_to_id_lookup.get(s_name)
                t_id = name_to_id_lookup.get(t_name)

                if s_id is None: s_id = self.find_character_id_by_name(s_name)
                if t_id is None: t_id = self.find_character_id_by_name(t_name)

                if type(s_name) == int and s_id is None:
                    s_id = s_name
                if type(t_name) == int and t_id is None:
                    t_id = t_name

                if s_id and t_id:
                    subgraph_edge['source'] = s_id
                    subgraph_edge['target'] = t_id

                    if self.find_two_characters_relationship_by_id(s_id, t_id):
                        self.update_relationship_by_edge(subgraph_edge)
                    else:
                        self.add_relationship(subgraph_edge)
                else:
                    print(f"Warning: could not resolve IDs for relationship {s_name}->{t_name}; skipping.")

            if not self.save_characters_graph():
                raise OSError(f"Failed to save character graph: {self.premise_id}")
            self.graph_history.append(deepcopy(self.characters_graph))
            self.is_vector_stale = True

            print('————————Character graph update complete!————————')
            return list(name_to_id_lookup.keys())

        print("Invalid output: <graph> tag not found!")
        return []

    def _coerce_proposal_shape_to_canonical(self, raw_node: dict) -> dict | None:
        if not isinstance(raw_node, dict):
            return None
        node = dict(raw_node)
        profile = node.pop("profile", None)
        if isinstance(profile, dict):
            if "description" not in node and profile.get("description"):
                node["description"] = profile["description"]
            if "short_term_goal" not in node and profile.get("short_term"):
                node["short_term_goal"] = profile["short_term"]
            if "long_term_goal" not in node and profile.get("long_term"):
                node["long_term_goal"] = profile["long_term"]
            if "importance" not in node and profile.get("importance"):
                node["importance"] = profile["importance"]
        if "role" in node and "importance" not in node:
            node["importance"] = node.pop("role")
        if "goal" in node and "short_term_goal" not in node:
            node["short_term_goal"] = node.pop("goal")
        cleaned = {k: v for k, v in node.items() if k in VALID_CHARACTER_ATTRIBUTES}
        if not cleaned.get("name"):
            return None
        return cleaned

    def update_graph_by_passage(self, passage: str, characters_info):

        import re
        self.last_passage_update_ok = False
        self.last_passage_update_error = None

        def _fail_passage_update(message: str, exception_type=ValueError):
            """Fail closed so the caller can roll the whole three-graph batch back."""
            self.last_passage_update_ok = False
            self.last_passage_update_error = message
            raise exception_type(message)
        def extract_all_id_numbers(text: str, deduplicate: bool = True) -> list[int]:
            pattern = r'id\s*["\']?\s*:\s*(\d+)'

            matches = re.findall(pattern, text, re.IGNORECASE)

            id_list = []
            for id_str in matches:
                try:
                    id_num = int(id_str)
                    id_list.append(id_num)
                except ValueError:
                    print(f"Skipping invalid ID string: {id_str}")
                    continue

            if deduplicate and id_list:
                id_list = list(set(id_list))

            return id_list
        try:
            related_ids = extract_all_id_numbers(characters_info)
            current_relation = []
            for edge in self.characters_graph['characters_relationship']:
                if int(edge['source']) in related_ids and int(edge['target']) in related_ids:
                    current_relation.append(edge)
            characters_relationship = f"\n# Existing relationships between characters [relationships present]: {current_relation}\n"
        except:
            print(">>> Character ID extraction failed")
            characters_relationship = ""

        # characters_graph_example = '''
        # {
        # "characters_node":[{
        #   "id": 1 , // int type
        #   "name": "Character Name",
        #   "aliases": ["Alias 1", "Nickname", "Title/Honorific"],
        #   "description": "[Global Profile] Intrinsic attributes of the character. Includes: background, personality traits, core abilities, physical appearance. Note: Only modify this field when major transformations occur (e.g., plastic surgery, personality corruption/darkening, identity revelation).",
        #   "short_term_goal": "[Current Intent] Specific goals the character wants to achieve in this chapter or the near future.",
        #   "long_term_goal": "[Ultimate Vision] Long-term pursuits spanning the entire book; update frequency is very low.",
        #   "importance": "Protagonist | Supporting Character | Extra/Mob",
        #   "current_plot_participation": "Key Figure | Participant | Non-participant",
        #   "status": "[Real-time State] Current physiological/psychological state (e.g., severe injury, coma, fear, excitement, imprisoned)."
        # }],
        # "characters_relationship": [
        # {
        #   "source": 1 , // source character id(int type), not name(str type)
        #   "target": 3 , // source character id(int type), not name(str type)
        #   "current_type": "Current Relationship (e.g., Ally/Enemy/Lover/Master-Apprentice)",
        #   "change_history": [
        #     {
        #       "from": "Old Relationship",
        #       "to": "New Relationship",
        #       "reason": "Summary of the specific event causing the relationship change"
        #     }
        #   ]
        # }]
        # }'''

        characters_graph_example = '''
            {
            "characters_node":[{
              "id": 1 , // integer
              "name": "Character name",
              "aliases": ["alias 1", "alias 2", "nickname", "title / honorific"],
              "description": "[Global setting] Inherent character attributes: background, personality traits, core abilities, appearance. Note: only update this field when the character undergoes a major transformation (e.g., reconstructive surgery, moral darkening / corruption, identity reveal).",
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

        task_prompt = f"""
        # Role
        You are a senior **novel character graph management expert**. Your core capability is to precisely analyze and extract character-information changes from a new chapter, and update the character network based on the existing character information.
        # Task
        Read the [New Chapter] and, using the [Character Profile] as the existing record, generate the **updated** subgraph.

        # Character Profile
           [Character Profile] :{characters_info}
           Note: the [Character Profile] is the complete authorized character set projected
           into the current working memory (M_sub). It may contain both Event-proposed first-
           appearance characters and historical characters, but every one already has an
           authoritative internal id in this working graph.
           {characters_relationship}
        # New Chapter Content
            [New Chapter] : {passage};

        # Output Schema
        Output the updated subgraph. Strictly follow the JSON structure below:
        {characters_graph_example}

        # Update Rules (Core Principles)

        1. **Node updates (Nodes)**:
           - **[MANDATORY] id-reuse rule**: for every character that already exists in the [Character Profile] (matched strictly by name **or** aliases), you **MUST** use that character's original id from the [Character Profile]. It is **strictly forbidden** to assign a new id to an existing character.
             - ✅ Correct: [Character Profile] has {{"id":1, "name":"Alice"}}; in this chapter Alice is despondent → output {{"id":1, "name":"Alice", "status":"despondent"}}
             - ❌ Wrong: [Character Profile] has {{"id":1, "name":"Alice"}}; Alice appears again → output {{"id":99, "name":"Alice", ...}} (id drift splits one character into two)
           - **status vs description**:
             - `status` is **transient** (e.g., left arm fractured, mana exhausted, enraged). Update this field actively; **any existing character whose mood / state changes in this chapter MUST appear under the original id with an updated status**.
               - `description` is **stable** (e.g., personality, appearance, hobbies, traits, age, identity, and other stable attributes).
                - *No creation in hot update*: NEVER emit a character absent from the [Character Profile]. New characters must be proposed by the Event Planner before writing and cannot be introduced through this passage-update path. Incidental unnamed people in the prose remain untracked.
                - *Existing character modification*: only modify when there is a major physical / psychological change (permanent injury, death, drastic mood shift, breaking / forming an important relationship) or when new setting information is revealed.

        2. **Relationship updates (Edges)**:
           - Record only **substantive** relationships (kinship, subordination, hostility, emotional bond). Ignore temporary, meaningless interactions (e.g., "asking for directions").
           - **Change history**: when `current_type` undergoes a fundamental change (e.g., from "stranger" to "ally"), you MUST append a record to `change_history` explaining the `reason`.

        3. **Format constraints**:
           - `source` and `target` in relationships must strictly use the character's **integer id** (do NOT use names).

        4. **[MANDATORY] Protagonist Singleton**:
           - A character with importance="Protagonist" is **globally unique**. If the main narrative figure in this chapter (POV character / primary plot driver) is the same person as the existing protagonist in the [Character Profile] — **even when the name spelling differs** (e.g., "Alice" vs "Alyse", "Alex" vs "Alec") — you MUST use the existing protagonist's id and append the differing spelling to that protagonist's `aliases` list.
           - It is **strictly forbidden** to emit a second character with importance="Protagonist". If you are uncertain, **default to treating the figure as an alias variant of the same protagonist**, not a new protagonist.

        # Workflow (thinking steps)
        1. **Character matching (the most important step)**: for each character appearing in the chapter text, look it up in the [Character Profile] in this order:
           a. Strict name match → hit → use its id
           b. Aliases-list match → hit → use its id
           c. Fuzzy approximate match (partial-spelling overlap, phonetic similarity, contextual consistency) → hit → use its id, and add the spelling used in the chapter into aliases
           d. None of the above → do not emit that person; this hot update cannot create nodes
           In particular: **protagonist candidates** (POV character / primary driver) must be force-matched to the existing protagonist via rule c, to avoid splitting.
        2. **Diff analysis**:
           - Did this character get wounded / undergo a drastic emotional change / break a relationship / change state in this chapter? → update `status` / `relationship`.
        3. **Execution**: for authorized characters that changed, emit the update under the original id; authorized characters with no change need not be output. Never create a node here.
        4. **Emit the subgraph**: collect only **changed authorized characters** and relationships whose endpoints are both in the [Character Profile].

        # Execute
        Based on the rules above, output the updated character-graph subgraph:

        # Output format: a tag-wrapped JSON character-graph subgraph. Example: <graph>""" + """{"characters_node":[...], "characters_relationship": [...]}</graph>
        """
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": task_prompt}]
        )
        resp_content = response.choices[0].message.content
        print('>>> Character graph update response:\n', resp_content)
        pattern = r'<graph>(.*?)</graph>'
        matches = re.findall(pattern, resp_content, re.DOTALL)

        if matches:
            updated_subgraph_content = matches[0].strip().replace("'", '"')
            if updated_subgraph_content.startswith("```json"):
                updated_subgraph_content = updated_subgraph_content[7:-3]
            try:
                subgraph = json.loads(updated_subgraph_content)
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed at position {e.pos}, starting LLM-assisted repair")
                task_prompt = f"""
                You are a strict JSON syntax repair specialist. Follow every rule below:

                ### Task requirements
                1. Given a JSON string with syntax errors, fix every violation so a standard JSON parser can parse the result.
                2. Fixes include, but are not limited to:
                   - remove trailing commas from objects and arrays;
                   - replace single quotes with double quotes (JSON requires double quotes);
                   - enclose every property name in double quotes;
                   - add missing closing braces or brackets;
                   - fix invalid escapes, newlines, and other special characters;
                   - preserve all original data (text, numbers, and structure); change only syntax errors.

                ### Output rules (follow exactly)
                1. Output only the repaired JSON text, with no explanations, Markdown fences, comments, or unnecessary whitespace.
                2. Do not wrap the result in ```json or ``` fences.
                3. Do not add any introductory or unrelated text.
                4. Keep the JSON compact and valid, with correct array and object nesting.

                ### Malformed JSON to repair
                {updated_subgraph_content}""".strip()

                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system",
                         "content": "You are a strict JSON syntax repair specialist. Output only valid JSON, with no other text."},
                        {"role": "user", "content": task_prompt}
                    ],
                    temperature=0.0,
                )

                corrected_json_str = response.choices[0].message.content.strip()
                if corrected_json_str.startswith("```json"):
                    corrected_json_str = corrected_json_str[7:-3].strip()

                try:
                    subgraph = json.loads(corrected_json_str)
                    print("JSON repair succeeded; parsing complete")

                except json.JSONDecodeError as e2:
                    raise Exception(f"Parsing still failed after LLM repair: {e2}, malformed content: {corrected_json_str}")
                    exit()
            if not isinstance(subgraph, dict):
                _fail_passage_update("Character hot update output must be a JSON object.")
            node_updates = subgraph.get('characters_node', [])
            if not isinstance(node_updates, list):
                _fail_passage_update("characters_node must be an array.")

            inv_name = []
            for subgraph_node in node_updates:
                if not isinstance(subgraph_node, dict):
                    _fail_passage_update(f"Character hot update node must be an object: {subgraph_node!r}")
                cid = subgraph_node.get('id')
                cname = subgraph_node.get('name')
                if cname is None or str(cname).strip() == "":
                    _fail_passage_update(
                        "Character hot update node lacks a nonempty name; refusing to skip it silently."
                    )
                cname = str(cname).strip()
                subgraph_node['name'] = cname

                if cid is None:
                    recovered = self.find_character_id_by_name(cname)
                    if recovered is not None:
                        cid = int(recovered)
                        subgraph_node['id'] = cid
                        print(f"🔧 [ID Recover] LLM omitted id; name='{cname}' resolved to id={cid}.")
                existing_records = self.find_characters_info_by_id([cid]) if cid is not None else []
                if not existing_records:
                    recovered = self.find_character_id_by_name(cname)
                    if recovered is not None:
                        cid = int(recovered)
                        subgraph_node['id'] = cid
                        existing_records = self.find_characters_info_by_id([cid])

                if not existing_records:
                    _fail_passage_update(
                        f"The passage contains unauthorized character {cname!r} (id={cid!r}); "
                        "passage hot update cannot create new characters outside R_active."
                    )

                existing_node = existing_records[0]
                existing_name = str(existing_node.get('name', '')).strip()
                existing_aliases = existing_node.get('aliases', []) or []
                existing_aliases = existing_aliases if isinstance(existing_aliases, list) else []

                if cname != existing_name and cname not in existing_aliases:
                    other_id = self.find_character_id_by_name(cname)
                    if other_id is None:
                        _fail_passage_update(
                            f"Character hot update identity conflict: id={cid!r} belongs to {existing_name!r}, "
                            f"but the output names unauthorized entity {cname!r}."
                        )
                    cid = int(other_id)
                    subgraph_node['id'] = cid
                    redirected = self.find_characters_info_by_id([cid])
                    if not redirected:
                        _fail_passage_update(
                            f"Character hot update could not resolve registered character {cname!r} with id={cid}."
                        )
                    existing_node = redirected[0]
                    existing_name = str(existing_node.get('name', '')).strip()
                    existing_aliases = existing_node.get('aliases', []) or []
                    existing_aliases = existing_aliases if isinstance(existing_aliases, list) else []

                subgraph_node['name'] = existing_name
                invalid_attributes = set(subgraph_node.keys()) - VALID_CHARACTER_ATTRIBUTES
                if invalid_attributes:
                    _fail_passage_update(
                        "Character hot update contains invalid fields: "
                        + ", ".join(sorted(invalid_attributes))
                    )

                for field in list(subgraph_node):
                    if field not in {"id", "name"} and _is_no_change_sentinel(
                        subgraph_node[field]
                    ):
                        subgraph_node.pop(field)

                # Passage deltas are patches, never complete replacement records.
                # Keep accumulated aliases and reject scalar/type drift before it
                # reaches the generic node.update implementation.
                if "aliases" in subgraph_node:
                    aliases = subgraph_node["aliases"]
                    if not isinstance(aliases, list) or any(
                        not isinstance(alias, str) for alias in aliases
                    ):
                        _fail_passage_update(
                            f"Aliases for character {existing_name!r} must be an array of strings."
                        )
                    merged_aliases = list(existing_aliases)
                    for alias in aliases:
                        alias = alias.strip()
                        if alias and alias != existing_name and alias not in merged_aliases:
                            merged_aliases.append(alias)
                    subgraph_node["aliases"] = merged_aliases

                text_fields = {
                    "description", "short_term_goal", "long_term_goal",
                    "importance", "current_plot_participation", "status",
                }
                for field in list(text_fields & set(subgraph_node)):
                    value = subgraph_node[field]
                    if value is None or (isinstance(value, str) and not value.strip()):
                        subgraph_node.pop(field)
                        continue
                    if not isinstance(value, str):
                        _fail_passage_update(
                            f"Field {field} for character {existing_name!r} must be text."
                        )
                    subgraph_node[field] = value.strip()

                # Role importance is proposal/global-profile canon, not transient
                # chapter state.  A hot update may echo it but cannot promote or
                # demote an existing character.
                if (
                    "importance" in subgraph_node
                    and subgraph_node["importance"] != existing_node.get("importance")
                ):
                    _fail_passage_update(
                        f"Character hot update cannot change importance: {existing_name!r} "
                        f"{existing_node.get('importance')!r} -> "
                        f"{subgraph_node['importance']!r}."
                    )

                res = self.update_character_info(cid, subgraph_node)
                res_text = res.strip() if isinstance(res, str) else ""
                if (
                    not res_text
                    or res_text.startswith("Error")
                    or res_text.casefold().startswith("error")
                ):
                    _fail_passage_update(f"Character node update failed: {res}")
                inv_name.append(existing_name)

            def _resolve_to_int_id(val):
                if val is None or isinstance(val, bool):
                    return None
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
                val_str = str(val).strip()
                if not val_str:
                    return None
                if val_str.startswith('c_tmp_'):
                    meta = self.characters_graph.get('_meta', {}) if isinstance(self.characters_graph, dict) else {}
                    temp_map = meta.get('temp_to_id', {}) if isinstance(meta, dict) else {}
                    if val_str in temp_map:
                        try:
                            return int(temp_map[val_str])
                        except (TypeError, ValueError):
                            pass
                if val_str.startswith('c_') and val_str[2:].isdigit():
                    return int(val_str[2:])
                try:
                    found = self.find_character_id_by_name(val_str)
                    if found is not None:
                        return int(found)
                except Exception:
                    pass
                return None

            relationship_updates = subgraph.get('characters_relationship', [])
            if not isinstance(relationship_updates, list):
                _fail_passage_update("characters_relationship must be an array.")
            for subgraph_edge in relationship_updates:
                if not isinstance(subgraph_edge, dict):
                    _fail_passage_update(f"Character relationship must be an object: {subgraph_edge!r}")
                raw_s = subgraph_edge.get('source')
                raw_t = subgraph_edge.get('target')
                s_id = _resolve_to_int_id(raw_s)
                t_id = _resolve_to_int_id(raw_t)

                if s_id is None or t_id is None:
                    _fail_passage_update(
                        f"Could not resolve relationship source={raw_s!r} target={raw_t!r} "
                        "to character IDs in the working graph."
                    )
                subgraph_edge['source'] = s_id
                subgraph_edge['target'] = t_id

                if self.find_two_characters_relationship_by_id(s_id, t_id):
                    edge_result = self.update_relationship_by_edge(subgraph_edge)
                else:
                    edge_result = self.add_relationship(subgraph_edge)
                edge_result_text = edge_result.strip() if isinstance(edge_result, str) else ""
                if (
                    not edge_result_text
                    or edge_result_text.startswith("Error")
                    or edge_result_text.casefold().startswith("error")
                ):
                    _fail_passage_update(f"Character relationship update failed: {edge_result}")

            if not self.save_characters_graph():
                _fail_passage_update(
                    f"Failed to save character graph: {self.premise_id}", OSError
                )
            self.graph_history.append(deepcopy(self.characters_graph))
            self.is_vector_stale = True
            self.last_passage_update_ok = True

            print('————————Character graph update complete!————————')
            return inv_name

        print("Invalid output: <graph> tag not found!")
        self.last_passage_update_error = "LLM output did not contain <graph>...</graph>"
        return []

    ##############################################################################
    ##############################################################################

    def _ensure_vectors_ready(self):
        if self.embedding_model is None:
            from tools.embedding_singleton import get_embedding_model
            self.embedding_model = get_embedding_model()

        if not self.is_vector_stale:
            return

        print("Refreshing the character vector index...")

        names = []
        texts_to_embed = []

        for node in self.characters_graph.get('characters_node', []):
            name = node['name']

            aliases = ",".join(node.get('aliases', []))
            desc = node.get('description', '')
            status = node.get('status', '')
            short_term_goal = node.get('short_term_goal', '')
            long_term_goal = node.get('long_term_goal', '')

            importance = node.get('importance', '')
            plot_role = node.get('current_plot_participation', '')

            rich_text = (
                f"Name: {name}; "
                f"Character importance: {importance}; "
                f"Plot role: {plot_role}; "
                f"Aliases: {aliases}; "
                f"Description: {desc}; "
                f"Current status: {status}; "
                f"Short-term goal: {short_term_goal}; "
                f"Long-term goal: {long_term_goal};"
            )
            if name == "Selina":
                print(f"DEBUG - Selina vector text: {rich_text}")


            names.append(name)
            texts_to_embed.append(rich_text)

        if not texts_to_embed:
            self.vector_keys = []
            self.vector_matrix = None
            self.is_vector_stale = False
            return

        embeddings = self.embedding_model.encode(texts_to_embed, convert_to_tensor=False)

        self.vector_keys = names
        self.vector_matrix = np.array(embeddings)
        self.is_vector_stale = False
        print(f"Vector index refresh complete; indexed {len(names)} characters.")

    def search_character(self, query: str, top_k: int = 3) -> list[dict]:
        self._ensure_vectors_ready()

        if self.vector_matrix is None or len(self.vector_keys) == 0:
            return []

        query_vector = self.embedding_model.encode(query, convert_to_tensor=False)

        scores = util.cos_sim(query_vector, self.vector_matrix)[0]

        top_results = []

        scores_np = scores.numpy()
        top_indices = np.argsort(scores_np)[::-1][:top_k]

        for idx in top_indices:
            score = float(scores_np[idx])

            if score < 0.35:
                continue

            name = self.vector_keys[idx]

            node = next((n for n in self.characters_graph['characters_node'] if n['name'] == name), {})

            top_results.append({
                "id": node.get('id', -1),
                "name": name,
                "score": round(score, 4),
                "aliases": node.get('aliases', [])[:3],
                "description_snippet": node.get('description', 'unknown'),
                "importance": node.get('importance', 'unknown'),
                "status": node.get('status', 'unknown'),
                "current_plot_participation": node.get('current_plot_participation', 'unknown'),
                "short_term_goal": node.get('short_term_goal', 'unknown'),
                "long_term_goal": node.get('long_term_goal', 'unknown')
            })

        return top_results

        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------

    def _format_search_result(self, node: dict, score: float) -> dict:
        return {
            "id": node.get('id', -1),
            "name": node.get('name', 'unknown'),
            "score": round(score, 4),
            "aliases": node.get('aliases', [])[:3],
            "description_snippet": node.get('description', '')[:100],
            "importance": node.get('importance', 'unknown'),
            "status": node.get('status', 'unknown'),
            "current_plot_participation": node.get('current_plot_participation', ''),
            "short_term_goal": node.get('short_term_goal', 'unknown'),
            "long_term_goal": node.get('long_term_goal', 'unknown')
        }

    def search_character_by_attribute(self, attribute_name: str, query_value: str, top_k: int = 3) -> list[dict]:
        """
        Hybrid search: search a specific attribute with [exact-match first + semantic fuzzy fallback].

        :param attribute_name: the attribute name to query (e.g., 'name', 'importance', 'short_term_goal').
                               If it is 'name' or 'aliases', both fields will be searched automatically.
        :param query_value: the user-supplied query value (e.g., 'Protagonist', 'main character', 'wants revenge').
        :param top_k: number of results to return.
        :return: list of matching characters.
        """
        target_attrs = []
        if attribute_name in ['name', 'aliases']:
            target_attrs = ['name', 'aliases']
        else:
            target_attrs = [attribute_name]

        print(f"Running hybrid search | attributes: {target_attrs} | query value: {query_value}")

        # ---------------------------------------------------------
        # ---------------------------------------------------------
        exact_matches = []
        seen_ids = set()

        for node in self.characters_graph.get('characters_node', []):
            is_match = False

            for attr in target_attrs:
                node_val = node.get(attr)

                if node_val is None:
                    continue

                if isinstance(node_val, list):
                    if query_value in node_val:
                        is_match = True
                else:
                    if str(node_val) == query_value:
                        is_match = True

            if is_match and node['id'] not in seen_ids:
                exact_matches.append(self._format_search_result(node, 1.0))
                seen_ids.add(node['id'])

        if exact_matches:
            print(f"Exact match found {len(exact_matches)} results; skipping fuzzy search.")
            return exact_matches

        # ---------------------------------------------------------
        # ---------------------------------------------------------
        print("No exact match; starting fuzzy search for the selected attributes...")

        if self.embedding_model is None:
            from tools.embedding_singleton import get_embedding_model
            self.embedding_model = get_embedding_model()

        candidates_nodes = []
        candidates_texts = []

        for node in self.characters_graph.get('characters_node', []):
            text_segments = []
            for attr in target_attrs:
                val = node.get(attr)
                if val:
                    if isinstance(val, list):
                        text_segments.append(",".join(val))
                    else:
                        text_segments.append(str(val))

            combined_text = " ".join(text_segments)

            if combined_text.strip():
                candidates_nodes.append(node)
                candidates_texts.append(combined_text)

        if not candidates_texts:
            return []

        query_vector = self.embedding_model.encode(query_value, convert_to_tensor=True)
        corpus_vectors = self.embedding_model.encode(candidates_texts, convert_to_tensor=True)

        scores = util.cos_sim(query_vector, corpus_vectors)[0].cpu().numpy()

        top_indices = np.argsort(scores)[::-1][:top_k]

        fuzzy_results = []
        for idx in top_indices:
            score = float(scores[idx])

            if score < 0.4:
                continue

            node = candidates_nodes[idx]
            fuzzy_results.append(self._format_search_result(node, score))

        return fuzzy_results



    def get_max_character_id(self) -> int:
        id_floor = int(self.characters_graph.get("_meta", {}).get("id_floor", 0))
        if not self.characters_graph.get('characters_node'):
            return id_floor
        return max(id_floor, max(int(node.get('id', 0)) for node in self.characters_graph['characters_node']))

    def find_character_index_by_id(self, target_id: int) -> int:
        normalized_id = _parse_character_query_id(target_id)
        if normalized_id is None:
            return -1
        for index, node in enumerate(self.characters_graph['characters_node']):
            if int(node.get('id', -1)) == normalized_id:
                return index
        return -1

    def find_character_id_by_name(self, name: str) -> int:
        normalized_id = _parse_character_query_id(name)
        if normalized_id is not None:
            return normalized_id if self.find_character_index_by_id(normalized_id) != -1 else None
        for node in self.characters_graph['characters_node']:
            if node['name'] == name or name in node.get('aliases', []):
                return int(node['id'])
        return None

    def find_characters_info(self, characters: list) -> dict:
        result = {}
        selected_characters_info = []
        selected_characters_relationship = []

        for character_node in self.characters_graph['characters_node']:
            print('character_node[name]:', character_node['name'])
            for character_name in characters:
                print('character_name:', character_name)
                if character_node['name'] in character_name:
                    selected_characters_info.append(character_node)

        for rel_edge in self.characters_graph['characters_relationship']:
            if rel_edge['source'] in characters and rel_edge['target'] in characters:
                selected_characters_relationship.append(rel_edge)

        result['characters_info'] =  selected_characters_info if selected_characters_info else ['No character information found in the character graph']
        result['characters_relationship'] = selected_characters_relationship if selected_characters_relationship else ['No relationships found in the character graph']
        return result

    def find_all_characters_name_index(self) -> dict[str, int]:
        characters_name_index: dict[str, int] = {}
        for character_node in self.characters_graph['characters_node']:
            name = character_node.get('name')
            cid = character_node.get('id')
            if name is None or cid is None:
                continue
            try:
                characters_name_index[name] = int(cid)
            except (TypeError, ValueError):
                continue
        return characters_name_index

    def find_character_all_relationships(self, character: str) -> list[dict]:
        related_relationships = []
        for relationship in self.characters_graph.get('characters_relationship', []):
            if relationship.get('source') == character or relationship.get('target') == character:
                related_relationships.append(relationship)
        return related_relationships

    def find_two_characters_relationship(self, character1_name: str, character2_name: str, exact=False) -> dict | None:
        for relationship in self.characters_graph.get('characters_relationship', []):
            source = relationship.get('source')
            target = relationship.get('target')
            if exact:
                if source == character1_name and target == character2_name:
                    return relationship
            else:
                if (source == character1_name and target == character2_name) or \
                        (source == character2_name and target == character1_name):
                    return relationship
        return None

    def find_two_characters_relationship_by_id(self, s_id, t_id):
        s_id = _parse_character_query_id(s_id)
        t_id = _parse_character_query_id(t_id)
        if s_id is None or t_id is None:
            return None
        for rel in self.characters_graph.get('characters_relationship', []):
            if int(rel['source']) == s_id and int(rel['target']) == t_id:
                return rel
        return None

    def find_characters_info_with_relationship_by_name(self, characters_name: list) -> dict:
        result = {}
        selected_characters_info = []
        selected_characters_relationship = []
        characters_id = []

        for character_node in self.characters_graph['characters_node']:
            if character_node['name'] in characters_name or set(character_node['aliases']) & set(characters_name):
                selected_characters_info.append(character_node)
                characters_id.append(character_node['id'])

        for rel_edge in self.characters_graph['characters_relationship']:
            if int(rel_edge['source']) in characters_id and int(rel_edge['target']) in characters_id:
                selected_characters_relationship.append(rel_edge)

        result['characters_info'] = selected_characters_info if selected_characters_info else [
            'No character information found in the character graph']
        result[
            'characters_relationship'] = selected_characters_relationship if selected_characters_relationship else [
            'No relationships found in the character graph']
        return result


    def fuzzy_search_characters(self, query:str, top_k:int=3) -> list[dict]:
        """
        Semantic matching: Match candidate roles through vague features and key information
        :param query: Vague role features or role key information
        :param top_k: Return the top 3 matching information by default
        :return: Matching results
        """
        search_res =  self.search_character(query, top_k)

        return search_res

    def find_characters_info_by_id(self, characters_id: list) -> list[dict]:
        """
        Retrieve the detailed information of the characters from the character graph based on their ids
        :param characters_id: A list of ids of the characters to be queried
        :return: A list composed of all information of the corresponding characters
        """
        print('>>> Calling the character lookup by ID')
        selected_characters_info = []

        query_values = characters_id if isinstance(characters_id, list) else [characters_id]
        target_ids = {
            normalized_id
            for cid in query_values
            if (normalized_id := _parse_character_query_id(cid)) is not None
        }

        for character_node in self.characters_graph['characters_node']:
            if int(character_node['id']) in target_ids:
                print('----Looking up character ', character_node['name'])
                selected_characters_info.append(character_node)

        return selected_characters_info

    def find_relationship_by_id(self, character1_id: str, character2_id: str, exact=False) -> dict | str:
        """
        Retrieve the detailed relationship dict between two characters. If there is a relationship, return it; if there is no relationship, return None
        :param character1_id: the id of character one
        :param character2_id: the id of character two
        :param exact: whether to strictly match the source character and target character. False for loose matching (undirected relationship edge), True for strict directed matching (directed relationship edge)
        :return: the result of the relationship dict
        """
        character1_id = _parse_character_query_id(character1_id)
        character2_id = _parse_character_query_id(character2_id)
        if character1_id is None or character2_id is None:
            return 'Invalid character ref; expected c_K or an integer ID.'

        for relationship in self.characters_graph.get('characters_relationship', []):
            source = relationship.get('source')
            target = relationship.get('target')
            if exact:
                if int(source) == int(character1_id) and int(target) == int(character2_id):
                    return relationship
            else:
                if (int(source) == int(character1_id) and int(target) == int(character2_id)) or \
                        (int(source) == int(character2_id) and int(target) == int(character1_id)):
                    return relationship
        return 'No relationship found between these two characters.'

    def find_characters_info_by_name(self, characters_name: list) -> list:
        """
        Retrieve detailed information of characters from the character map based on their exact names or one of their exact aliases
        :param characters_name: A list consisting of the names or aliases of the characters to be queried
        :return: All information of the corresponding characters
        """

        query_values = characters_name if isinstance(characters_name, list) else [characters_name]
        requested_ids = {
            normalized_id
            for value in query_values
            if (normalized_id := _parse_character_query_id(value)) is not None
        }
        requested_names = {
            str(value).strip()
            for value in query_values
            if _parse_character_query_id(value) is None and str(value).strip()
        }
        selected_characters_info = []

        for character_node in self.characters_graph['characters_node']:
            if (
                int(character_node['id']) in requested_ids
                or character_node['name'] in requested_names
                or set(character_node.get('aliases', [])) & requested_names
            ):
                selected_characters_info.append(character_node)

        return selected_characters_info

    # def find



    # ---------------------------------------------------

    def add_characters(self, new_characters: list) -> str:
        if not isinstance(new_characters, list):
            return f"Error: Input must be a list; got {type(new_characters)}."

        if not new_characters:
            return"Warning: the input list is empty; no characters were added."

        existing_name_to_id = self.find_all_characters_name_index()  # {name: id}
        existing_names = set(existing_name_to_id.keys())
        names_in_input_list = set()

        to_add: list[dict] = []
        to_update: list[tuple[int, dict]] = []  # (existing_id, character_info)
        next_character_id = self.get_max_character_id()

        for character_info in new_characters:
            if not isinstance(character_info, dict):
                return f"Error: Input item '{character_info}' is not a valid dictionary."
            invalid_attributes = set(character_info.keys()) - VALID_CHARACTER_ATTRIBUTES
            if invalid_attributes:
                return f"Error: Invalid or misspelled character attributes: {', '.join(invalid_attributes)}."

            name = character_info.get('name')
            if not name:
                return f"Error: Character dictionary {character_info} is missing a nonempty name."

            if name in names_in_input_list:
                return f"Error: The submitted list contains duplicate character name '{name}'."
            names_in_input_list.add(name)

            if name in existing_names:
                existing_id = existing_name_to_id[name]
                character_info['id'] = int(existing_id)
                to_update.append((existing_id, character_info))
                continue

            next_character_id += 1
            character_info['id'] = next_character_id
            to_add.append(character_info)

        update_msgs = []
        for existing_id, info in to_update:
            res = self.update_character_info(existing_id, dict(info))
            update_msgs.append(f"{info.get('name')}(id={existing_id}): {res}")
            print(f"🔁 Duplicate name; falling back to update: {info.get('name')} (id={existing_id})")

        if to_add:
            self.characters_graph['characters_node'].extend(to_add)
            print('Adding: ', [c.get('name') for c in to_add])

        added_names = [c.get('name') for c in to_add]
        updated_names = [info.get('name') for _id, info in to_update]
        parts = []
        if added_names:
            parts.append(f"Added {len(added_names)}: {', '.join(added_names)}")
        if updated_names:
            parts.append(f"Updated {len(updated_names)} existing names: {', '.join(updated_names)}")
        return "; ".join(parts) if parts else "Warning: no characters were processed."

    
    def add_relationship(self, new_relationship: dict) -> str:
        if not isinstance(new_relationship, dict):
            return f"Error: Input must be a dictionary; got {type(new_relationship)}."

        s_id = new_relationship.get('source')
        t_id = new_relationship.get('target')

        if not s_id or not t_id: return "Error: missing ID"

        if self.find_two_characters_relationship_by_id(s_id, t_id):
            return "Error: relationship already exists"
        current_type = new_relationship.get('current_type')

        if not all([s_id, t_id, current_type]):
            return f"Error: Relationship dictionary {new_relationship} is missing source, target, or current_type."
        if not isinstance(current_type, str) or not current_type.strip():
            return "Error: current_type must be nonempty text."

        if s_id == t_id:
            return f"Error: A character cannot have a relationship with itself ('{s_id}' -> '{t_id}')."

        existing_s = self.find_character_index_by_id(s_id)
        existing_t = self.find_character_index_by_id(t_id)

        if existing_s == -1:
            return f"Error: Source character '{s_id}' does not exist in the character graph."
        if existing_t == -1:
            return f"Error: Target character '{t_id}' does not exist in the character graph."

        if self.find_two_characters_relationship_by_id(s_id, t_id) is not None:
            return f"Error: Relationship between '{s_id}' and '{t_id}' already exists; do not add it again."

        history = new_relationship.get('change_history', [])
        history_error = _relationship_history_error(history)
        if history_error:
            return f"Error: {history_error}."
        if not history:
            history = [{
                "from": "none",
                "to": current_type.strip(),
                "reason": "Relationship introduced by chapter extraction.",
            }]
        new_relationship['change_history'] = deepcopy(history)

        self.characters_graph['characters_relationship'].append(new_relationship)
        return f"Relationship added successfully: '{s_id}' and '{t_id}' have relationship '{new_relationship.get('current_type')}'."

    
    # def update_character_info(self, character_name: str, _updated_info: dict) -> str:
    #     """
    #     """
    #     updated_info = deepcopy(_updated_info)
    #
    #
    #     if not isinstance(updated_info, dict) or not updated_info:
    #
    #     if 'name' in updated_info:
    #         updated_info.pop('name')
    #     print('updated_info: ', updated_info)
    #
    #
    #     incoming_attributes = set(updated_info.keys())
    #     invalid_attributes = incoming_attributes - VALID_CHARACTER_ATTRIBUTES
    #
    #     if invalid_attributes:
    #
    #     characters_name_map = self.find_all_characters_name_index()
    #     if character_name not in characters_name_map:
    #
    #     character_index = characters_name_map[character_name]
    #     character_node = self.characters_graph['characters_node'][character_index]
    #     character_node.update(updated_info)
    #

    # def update_character_info(self, character_id: int, updated_info: dict) -> str:
    #     """
    #     """
    #     index = self.find_character_index_by_id(character_id)
    #     if index == -1:
    #
    #     node = self.characters_graph['characters_node'][index]
    #
    #     if 'id' in updated_info:
    #         updated_info.pop('id')
    #
    #     if 'name' in updated_info and updated_info['name'] != node['name']:
    #         exit()
    #
    #     node.update(updated_info)

    def update_character_info(self, character_id: int, updated_info: dict) -> str:
        index = self.find_character_index_by_id(character_id)
        if index == -1:
            return f"Error: ID {character_id} does not exist."

        node = self.characters_graph['characters_node'][index]

        if 'id' in updated_info:
            updated_info.pop('id')

        if 'name' in updated_info:
            new_name = updated_info['name']
            old_name = node['name']

            if new_name != old_name:
                print(f"🔄 Rename detected (ID {character_id}): '{old_name}' -> '{new_name}'")

                if 'aliases' not in node:
                    node['aliases'] = []

                if old_name and old_name not in node['aliases']:
                    node['aliases'].append(old_name)
                    print(f"   -> Archived old name '{old_name}' in the aliases list.")

                if hasattr(self, 'is_vector_stale'):
                    self.is_vector_stale = True


        # cleaned_info = {k: v for k, v in updated_info.items() if k in valid_keys}
        # node.update(cleaned_info)

        node.update(updated_info)

        if hasattr(self, 'is_vector_stale'):
            self.is_vector_stale = True

        return f"Successfully updated ID {character_id}"

    
    def update_relationship(self, source_name: str, target_name: str, updated_data: dict, update_reason:str) -> str:
        if not isinstance(updated_data, dict) or not updated_data:
            return "Error: Update information must be a nonempty dictionary."
        if 'source' in updated_data or 'target' in updated_data:
            if updated_data['source'] != source_name or updated_data['target'] != target_name:
                return "Error: Changing the source or target character name of a relationship is not allowed."

        incoming_attributes = set(updated_data.keys())
        invalid_attributes = incoming_attributes - VALID_RELATIONSHIP_ATTRIBUTES

        if invalid_attributes:
            return f"Error: Invalid or misspelled relationship attributes: {', '.join(invalid_attributes)}. Check them and retry."

        for relationship in self.characters_graph.get('characters_relationship', []):
            source = relationship.get('source')
            target = relationship.get('target')

            # if (source == source_name and target == target_name) or (source == target_name and target == source_name):
            if source == source_name and target == target_name :
                if 'current_type' in updated_data:
                    old_type = relationship.get('current_type')
                    new_type = updated_data['current_type']
                    if old_type != new_type:
                        history_entry = {"from": old_type, "to": new_type, "reason": update_reason}
                        relationship.setdefault('change_history', []).append(history_entry)
                    relationship.update(updated_data)
                    return f"Success: updated the relationship between '{source_name}' and '{target_name}'."
                else:
                    return f"Error: An update without current_type is invalid!"
        return f"Error: Relationship between '{source_name}' and '{target_name}' not found."

    # def update_relationship_by_edge(self, updated_data:dict) -> str:
    #     index = 0
    #     for relationship in self.characters_graph.get('characters_relationship', []):
    #         source = relationship.get('source')
    #         target = relationship.get('target')
    #         if source == updated_data['source'] and target ==  updated_data['target']:
    #             current_rel = self.characters_graph['characters_relationship'][index]
    #             self.characters_graph['characters_relationship'][index] = deepcopy(relationship)
    #         index+=1

    def update_relationship_by_edge(self, updated_data: dict) -> str:
        if not isinstance(updated_data, dict):
            return "Error: Relationship update must be a dictionary"
        try:
            s_id = int(updated_data['source'])
            t_id = int(updated_data['target'])
        except (KeyError, TypeError, ValueError):
            return "Error: Relationship update lacks valid source/target"
        if 'current_type' in updated_data:
            current_type = updated_data.get('current_type')
            if not isinstance(current_type, str) or not current_type.strip():
                return "Error: current_type must be nonempty text"
        incoming_history = updated_data.get('change_history', [])
        history_error = _relationship_history_error(incoming_history)
        if history_error:
            return f"Error: {history_error}"

        def _as_history_entries(value):
            if value is None:
                return []
            return value if isinstance(value, list) else [value]

        def _history_key(entry):
            if isinstance(entry, dict):
                core = tuple(str(entry.get(k, "")).strip() for k in ("from", "to", "reason"))
                if any(core):
                    return ("transition",) + core
            return ("json", json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str))

        for rel in self.characters_graph.get('characters_relationship', []):
            if int(rel['source']) == s_id and int(rel['target']) == t_id:
                merged_history = []
                seen_history = set()

                def _append_unique(entry):
                    key = _history_key(entry)
                    if key in seen_history:
                        return False
                    seen_history.add(key)
                    merged_history.append(deepcopy(entry))
                    return True

                for entry in _as_history_entries(rel.get('change_history')):
                    _append_unique(entry)

                fresh_incoming = []
                for entry in _as_history_entries(updated_data.get('change_history')):
                    if _append_unique(entry):
                        fresh_incoming.append(entry)

                old_type = rel.get('current_type')
                new_type = updated_data.get('current_type', old_type)
                if old_type != new_type:
                    has_current_transition = any(
                        isinstance(entry, dict)
                        and entry.get('from') == old_type
                        and entry.get('to') == new_type
                        for entry in fresh_incoming
                    )
                    if not has_current_transition:
                        reason = next(
                            (
                                str(entry.get('reason')).strip()
                                for entry in reversed(fresh_incoming)
                                if isinstance(entry, dict) and str(entry.get('reason') or '').strip()
                            ),
                            "Relationship type updated by chapter extraction.",
                        )
                        _append_unique({"from": old_type, "to": new_type, "reason": reason})

                for key, value in updated_data.items():
                    if key not in {'source', 'target', 'change_history'}:
                        rel[key] = deepcopy(value)
                rel['change_history'] = merged_history
                return "Relationship updated successfully"
        return "Error: Specified edge not found"




    #####################################################################################
    #####################################################################################

    def rollback_last_graph(self):
        self.graph_history.pop()
        self.characters_graph = self.graph_history[len(self.graph_history) - 1]
        if not self.save_characters_graph():
            raise OSError(f"Failed to save rolled-back character graph: {self.premise_id}")

    def save_characters_graph(self):
        save_state = set_characters_graph_data(self.premise_id, self.characters_graph)
        if save_state:
            print('----------Saved successfully!----------')
        else:
            print('----------Save failed!----------')
        return bool(save_state)



