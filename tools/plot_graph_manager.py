import os
import json
import re
from copy import deepcopy
from agents import AgentGlobalConfig, DEFAULT_MODEL_NAME
from data import plots_graph_data, set_plots_graph_data
from tools.graph_refs import make_plot_ref, parse_plot_ref
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer, util
import numpy as np

# client = AgentGlobalConfig.DeepSeekCLIENT
client = AgentGlobalConfig.GPTCLIENT

VALID_PLOT_ATTRIBUTES = {
    "chapter", "overview", "details", "importance", "event",
    "characters_involved", "environment", "foreshadowing", "canon_evidence"
}

VALID_REL_TYPES = {"e_temp", "e_cas", "e_res"}

LEGACY_REL_TYPE_ALIAS = {
    "sequence": "e_temp",
    "caused_by": "e_cas",
    "resolves": "e_res",
}

MAX_PLOT_ADJACENCY_RESULTS = 50


def _parse_plot_query_id(value) -> int | None:
    """Accept the public ``p_K`` form as well as legacy integer/numeric IDs."""
    if isinstance(value, bool):
        return None
    parsed = parse_plot_ref(value)
    if parsed is not None:
        return int(parsed)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _parse_formal_plot_ref(value) -> int | None:
    """Strictly accept one canonical public ``p_K`` ref.

    Unlike the legacy node lookup helper, adjacency traversal is an Agent-facing
    graph operation.  It must not reinterpret an integer, a numeric string, a
    temporary ref, surrounding whitespace, or a zero-padded alias as another
    plot node.
    """
    if not isinstance(value, str) or value != value.strip():
        return None
    parsed = parse_plot_ref(value)
    if parsed is None or value != make_plot_ref(parsed):
        return None
    return int(parsed)


def normalize_edge_type(rel_type: str) -> str | None:
    if not isinstance(rel_type, str):
        return None
    canonical = LEGACY_REL_TYPE_ALIAS.get(rel_type.strip(), rel_type.strip())
    return canonical if canonical in VALID_REL_TYPES else None

class PlotGraphManager:
    def __init__(self, premise_id):
        self.premise_id = premise_id
        self.model_name = DEFAULT_MODEL_NAME
        self.history = []
        self.graph_history = []
        self.plots_graph = plots_graph_data(premise_id)
        self.graph_history.append(deepcopy(self.plots_graph))
        self.embedding_model = None
        self.vector_cache = {}
        self.vector_keys = []
        self.vector_matrix = None
        self.is_vector_stale = True

    ###############################################################################
    ###############################################################################


    def _ensure_plot_vectors_ready(self):
        if self.embedding_model is None:
            from tools.embedding_singleton import get_embedding_model
            self.embedding_model = get_embedding_model()

        if not self.is_vector_stale:
            return

        print("Refreshing the plot vector index...")

        plot_ids = []
        texts_to_embed = []
        self._plot_id_to_node = {}
        self._plot_meta = []

        for node in self.plots_graph.get("plots_node", []):
            pid = str(node.get("id", "")).strip()
            if not pid:
                continue

            chapter = node.get("chapter", None)
            overview = node.get("overview", "")
            details = node.get("details", "")
            importance = node.get("importance", "")
            event = node.get("event", "")
            env = node.get("environment", node.get("scene", ""))
            chars = node.get("characters_involved", [])
            if isinstance(chars, list):
                chars_text = ", ".join([str(x) for x in chars])
            else:
                chars_text = str(chars)

            foreshadowing = node.get("foreshadowing", False)
            foreshadowing_text = "foreshadowing present" if foreshadowing else "no foreshadowing"
            canon_evidence = node.get("canon_evidence", [])
            evidence_spans = []
            if isinstance(canon_evidence, list):
                for record in canon_evidence:
                    if not isinstance(record, dict):
                        continue
                    spans = record.get("prose_evidence", [])
                    if isinstance(spans, list):
                        evidence_spans.extend(
                            span.strip()
                            for span in spans
                            if isinstance(span, str) and span.strip()
                        )
            evidence_text = " | ".join(dict.fromkeys(evidence_spans))

            rich_text = (
                f"Plot ID: {pid}; "
                f"Chapter: {chapter}; "
                f"Overview: {overview}; "
                f"Event type: {event}; "
                f"Characters: {chars_text}; "
                f"Scene: {env}; "
                f"Importance: {importance}; "
                f"Foreshadowing: {foreshadowing_text}; "
                f"Details: {details}; "
                f"Passage evidence: {evidence_text}"
            )

            plot_ids.append(pid)
            texts_to_embed.append(rich_text)

            self._plot_id_to_node[pid] = node
            self._plot_meta.append({
                "id": pid,
                "chapter": chapter,
                "importance": importance,
                "foreshadowing": bool(foreshadowing),
                "characters_involved": node.get("characters_involved", []),
                "environment": env,
                "event": event,
            })

        if not texts_to_embed:
            self.vector_keys = []
            self.vector_matrix = None
            self.is_vector_stale = False
            return

        embeddings = self.embedding_model.encode(texts_to_embed, convert_to_tensor=False)
        self.vector_keys = plot_ids
        self.vector_matrix = np.array(embeddings)
        self.is_vector_stale = False

        print(f"Plot vector index refreshed; indexed {len(plot_ids)} plot nodes.")

    def _match_filters(self, meta: dict, filters: dict) -> bool:
        if not filters:
            return True

        ch = meta.get("chapter", None)

        if "chapter_min" in filters and ch is not None and ch < filters["chapter_min"]:
            return False
        if "chapter_max" in filters and ch is not None and ch > filters["chapter_max"]:
            return False

        if "importance" in filters and filters["importance"]:
            if meta.get("importance") != filters["importance"]:
                return False

        if "foreshadowing" in filters:
            if meta.get("foreshadowing") != bool(filters["foreshadowing"]):
                return False

        if "character" in filters and filters["character"]:
            chars = meta.get("characters_involved", [])
            if isinstance(filters["character"], str):
                if filters["character"] not in chars:
                    return False
            else:
                if not any(c in chars for c in filters["character"]):
                    return False

        if "environment" in filters and filters["environment"]:
            if filters["environment"] not in (meta.get("environment") or ""):
                return False

        if "event" in filters and filters["event"]:
            if filters["event"] not in (meta.get("event") or ""):
                return False

        return True

    def search_plot(self, query: str, top_k: int = 5, score_threshold: float = 0.35, filters: dict | None = None) -> list[dict]:
        """
        Plot semantic retrieval of plots: Match the top 5 (by default) most suspicious plot nodes based on plot characteristics and certain key information
        :param query: Such as "Treasure hunting in the market and discovering a black iron piece" / "Being followed" / "Chapter 10 Foreshadowing"
        :param top_k: Optional, default 5; match the top top_k results with scores higher than the threshold
        :param score_threshold: Optional, default 0.35; default score threshold
        :param filters: Optional, default no filtering information; filter search results based on the attributes of a specific plot node
        {
        "chapter_min": 10,
        "chapter_max": 12,
        "importance": "Main plot",
        "foreshadowing": True,
        "character": "Xiao Yan",
        "environment": "Wutan City",
        "event": "Being followed"
        }
        :return: Detailed information of the searched suspicious plot nodes
        """
        self._ensure_plot_vectors_ready()

        if self.vector_matrix is None or len(self.vector_keys) == 0:
            return []

        candidate_indices = []
        if filters:
            for i, meta in enumerate(self._plot_meta):
                if self._match_filters(meta, filters):
                    candidate_indices.append(i)
        else:
            candidate_indices = list(range(len(self._plot_meta)))

        if not candidate_indices:
            return []

        query_vec = self.embedding_model.encode(query, convert_to_tensor=False)

        sub_matrix = self.vector_matrix[candidate_indices]
        scores = util.cos_sim(query_vec, sub_matrix)[0]
        scores_np = scores.cpu().numpy()

        # TopK
        sub_top = np.argsort(scores_np)[::-1][:top_k]

        results = []
        for sub_idx in sub_top:
            score = float(scores_np[sub_idx])
            if score < score_threshold:
                continue

            global_idx = candidate_indices[sub_idx]
            pid = self.vector_keys[global_idx]
            node = self._plot_id_to_node.get(pid, {})

            overview = node.get("overview", "")
            details = node.get("details", "")
            details_snippet = details[:160] + ("..." if len(details) > 160 else "")
            canon_evidence_snippets = []
            raw_canon_evidence = node.get("canon_evidence", [])
            if isinstance(raw_canon_evidence, list):
                for record in raw_canon_evidence:
                    if not isinstance(record, dict):
                        continue
                    spans = record.get("prose_evidence", [])
                    if not isinstance(spans, list):
                        continue
                    for span in spans:
                        if isinstance(span, str) and span.strip():
                            snippet = span.strip()
                            if len(snippet) > 240:
                                snippet = snippet[:240] + "..."
                            if snippet not in canon_evidence_snippets:
                                canon_evidence_snippets.append(snippet)
                        if len(canon_evidence_snippets) >= 3:
                            break
                    if len(canon_evidence_snippets) >= 3:
                        break

            chars = node.get("characters_involved", [])
            if isinstance(chars, list):
                chars = chars[:6]

            results.append({
                "id": int(node.get("id", pid)) if str(node.get("id", pid)).isdigit() else pid,
                "chapter": node.get("chapter", None),
                "overview": overview,
                "score": round(score, 4),
                "importance": node.get("importance", ""),
                "event": node.get("event", ""),
                "environment": node.get("environment", ""),
                "characters_involved": chars,
                "foreshadowing": bool(node.get("foreshadowing", False)),
                "details_snippet": details_snippet,
                "canon_evidence": canon_evidence_snippets,
            })

        return results


    def update_graph_by_passage_backup(self, passage):
        plots_graph_example = """
        {
            "plots_node": [
                {
                "id":1,  // Plot ID (unique, increasing)
                "chapter": 1,  // Chapter number; a chapter may contain multiple plot nodes
                "overview": "Plot overview",
                "details": "Plot details: state which characters did what, and in which scene",
                "importance": "main plot / subplot",
                "event": "Several plot nodes may belong to the same event", 
                "characters_involved": [
                    "Character name (use the existing name when present in characters; otherwise use the new name)",
                ],
                "scene": "Scene name (use an existing name from scenes, or the name given in the passage)",
                "foreshadowing": true,  // Whether this is foreshadowing (true); use sparingly
                },
            ],
            "plots_relationship": [
                // Only caused_by and resolves relationships (legacy format, for compatibility)
                ["2", "5", "caused_by"], // Causal direction: plot 2 (cause) -> plot 5 (effect)
                ["5", "3", "resolves"]   // Resolution direction: plot 5 (payoff) -> plot 3 (setup)
            ]
        }
        """

        task_prompt = f"""
        # Role: You are an avid fiction reader updating the novel's plot graph from a new chapter.
        # The plot graph node schema is: {plots_graph_example}
        # Task: Extract every plot beat from the latest chapter (who did what and where), then update the plot graph.
        # Task steps (show the steps explicitly):
        ## 1.1 First identify the scenes in the chapter. **Each plot node may contain at most one scene.** A scene may contain several plot nodes when it has many beats; a scene change requires a new plot node.
        ## 1.2 In details, describe the scene, its characters, and important setting information.
        ## 2.Identify the plot beats within each scene using step 1.
        ## 3.Identify the characters in each plot beat.
        ## 4.Build plot nodes according to the schema.
        ## 5.Build edges between plot nodes according to the schema. Only major beats need relationships; most nodes have none.
        ## 6.Check the output format. Remove stray quotation marks and other invalid characters from field values.
        # Output format: Return a JSON plot subgraph inside <graph> tags. The outer object must contain plots_node and plots_relationship. Example: <graph>{{"plots_node": [], "plots_relationship": []}}</graph>
        # New chapter: {passage}
        """
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": task_prompt}]
            )
            resp_content = response.choices[0].message.content
            pattern = r'<graph>(.*?)</graph>'
            matches = re.findall(pattern, resp_content, re.DOTALL)
            if matches:
                updated_subgraph_content = matches[0].strip().replace("'", '"')
                print('updated_subgraph_content\n', updated_subgraph_content)
                subgraph_dict = json.loads(updated_subgraph_content)
                current_graph_node_count = len(self.plots_graph['plots_node'])
                for node in subgraph_dict['plots_node']:
                    node['id'] = node['id'] + current_graph_node_count
                for rela in subgraph_dict['plots_relationship']:
                    rela[0] = str(int(rela[0]) + current_graph_node_count)
                    rela[1] = str(int(rela[1]) + current_graph_node_count)
                self.plots_graph['plots_node'].extend((subgraph_dict['plots_node']))
                self.plots_graph['plots_relationship'].extend(subgraph_dict['plots_relationship'])
                if not self.save_plots_graph():
                    raise OSError(f"Failed to save plot graph: {self.premise_id}")
                print('————————Plot graph update complete!————————')
                self.graph_history.append(deepcopy(self.plots_graph))
                return True
            print("Invalid output: character graph tag not found!")
            return False
        except Exception as e:
            print('History on error:\n', self.graph_history)
            print('Graph on error:\n', self.plots_graph)
            print('Plot graph error: ', str(e))
            return False


    def get_max_plot_id(self) -> int:
        id_floor = int(self.plots_graph.get("_meta", {}).get("id_floor", 0))
        if not self.plots_graph.get('plots_node'):
            return id_floor
        return max(id_floor, max(node['id'] for node in self.plots_graph['plots_node']))

    def get_recent_plots_context(self, limit=5) -> str:
        if not self.plots_graph.get('plots_node'):
            return "No prior plot nodes (first chapter)."

        recent_nodes = self.plots_graph['plots_node'][-limit:]
        context_str = []
        for node in recent_nodes:
            is_fore = "[foreshadowing]" if node.get('foreshadowing') else ""
            context_str.append(f"ID:{node['id']} {is_fore} [overview]:{node['overview']}")
        return "\n".join(context_str)

    def update_graph_by_passage(self, passage: str, chapter_num: int, characters_name: list, envs_name: list):
        print(f"--- Starting plot update for chapter {chapter_num} ---")

        current_max_id = self.get_max_plot_id()
        start_id = current_max_id + 1
        recent_context = self.get_recent_plots_context(limit=8)

        plots_graph_example = """
        {
            "plots_node": [
                {
                "id": 105,
                "chapter": 10,
                "overview": "Xiao Yan discovers a mysterious black iron piece at a stall",
                "details": "...",
                "importance": "Main Story",
                "event": "Market Treasure Hunt",
                "characters_involved": ["Xiao Yan", "Stall Owner"],
                "environment": "Corner of Wutan City Market",   // REQUIRED, scene-anchored
                "foreshadowing": true
                },
                {
                "id": 106,
                "chapter": 10,
                "overview": "Xiao Yan is stalked after leaving the market",
                "details": "...",
                "importance": "Main Story",
                "event": "Encountering Stalkers",
                "characters_involved": ["Xiao Yan", "Mu Li"],
                "environment": "Streets of Wutan City",
                "foreshadowing": false
                }
            ],
            "plots_relationship": [
                // e_temp: chronological backbone (preferred over scattered ordering)
                ["105", "106", "e_temp"],
                // e_cas: strong causal dependency
                ["105", "106", "e_cas"]
            ]
        }
        """

        task_prompt = f"""
        # Role: You are a Novel Logic Expert currently constructing the [Causal Plot Chain C_P] for a novel.
        # Task: Analyze the given new chapter, decompose it into multiple fine-grained scene-anchored "Plot Nodes," and establish logical relationships between them and historical plots, as well as **among themselves**.

        # Key Parameters (Strict Adherence):
        1. **Current Chapter Number**: {chapter_num}
        2. **Start ID**: {start_id} (The first new node ID MUST be {start_id}; subsequent IDs must increment. Usage of other IDs is strictly forbidden.)
        3. **Current Chapter Info**:
           - Character List: {characters_name}
           - Environment List: {envs_name}
        4. **Recent Historical Context**:
        {recent_context}

        # Hard Rule: Scene-Anchored Nodes
        Every plot node MUST be anchored to exactly one physical scene from the provided Environment List
        (`environment` field non-empty). When the environment switches mid-scene, split into multiple nodes.

        # Task Steps:
        1. **Scene Splitting**: Identify the scenes appearing in the chapter. If the environment switches or the core conflict changes, you MUST create a new node.
        2. **Node Construction**: Extract `characters_involved` (select from the provided Character List), `environment` (select from the provided Environment List), `details`, and other fields.
        3. **Edge Construction (three edge types — STRICT)**:
           - **e_temp (temporal backbone)**:
             - Connect adjacent steps (v_t, v_{{t+1}}) along the chronological backbone of the chapter and to the closest historical node when applicable.
             - Example: ["105", "106", "e_temp"]

           - **e_cas (causal dependency — high threshold)**:
             - Use ONLY when the preceding event is a necessary condition or direct trigger for the following event.
             - ❌ Reject "He ate" -> "He went out". (this is e_temp only)
             - ✅ Accept "He was poisoned" -> "He sought the antidote". (e_cas)

           - **e_res (resolution / payoff — very high threshold)**:
             - Use ONLY when this node pays off a previously-marked foreshadowing node (`foreshadowing: true`).
             - Format: ["payoff_id", "setup_id", "e_res"]
             - Within-chapter `e_res` should be avoided unless this chapter contains a major reveal.

        ## Special Task: Foreshadowing Detection
        Please apply literary logic to review every extracted plot node. If a node meets one of the following [Foreshadowing Features], you MUST set `foreshadowing` to `true` and note the reason in `details`:

        1. **Chekhov's Gun (High Detail, Low Utility)**: 
           - Is there an item/person described with specific physical details but serves [absolutely no practical function] in this chapter (not used, did not advance the plot)?
        2. **Abnormal Behavior**: 
           - Did a character behave strangely in a way that contradicts their persona without explanation?
           - Did a character hesitate to speak or give a vague prophecy?
        3. **Open Loops**:
           - Did an environmental anomaly (e.g., sudden noise, sudden chill) occur without a discovered cause?

        **Note**: Do not treat every detail as foreshadowing. Only information that is **"Conspicuous but currently useless"** qualifies. Set `false` for ordinary scenic descriptions.

        # Output Subgraph Schema:
        {plots_graph_example}

        # Chapter Text:
        {passage}

        # Requirements:
        - Explicitly output your analysis process.
        - The final answer must be a Plot Dictionary wrapped in <graph>...</graph> tags.
        - The output must be valid JSON.
        - The dictionary must ONLY contain two lists: `plots_node` and `plots_relationship`.
        """

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": task_prompt}],
                temperature=0.3
            )
            resp_content = response.choices[0].message.content
            print('Plot extraction response:\n', resp_content)

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


                final_nodes = []
                for node in subgraph.get('plots_node', []):
                    node['chapter'] = chapter_num
                    if 'foreshadowing' not in node:
                        node['foreshadowing'] = False
                    if 'scene' in node and 'environment' not in node:
                        node['environment'] = node.pop('scene')
                    if not node.get('environment'):
                        node['environment'] = envs_name[0] if envs_name else ""
                    if not node.get('environment'):
                        print(f"Warning: discarding a node without environment {node.get('overview','')}")
                        continue
                    final_nodes.append(node)

                final_rels = []
                valid_ids = {str(n['id']) for n in self.plots_graph.get('plots_node', [])} | \
                            {str(n['id']) for n in final_nodes}

                for rel in subgraph.get('plots_relationship', []):
                    if len(rel) < 3:
                        continue
                    s_id = str(rel[0])
                    t_id = str(rel[1])
                    r_type = normalize_edge_type(rel[2])
                    if r_type is None:
                        print(f"Warning: discarding an invalid edge type {rel}")
                        continue
                    if s_id in valid_ids and t_id in valid_ids:
                        final_rels.append([s_id, t_id, r_type])
                    else:
                        print(f"Warning: discarding an invalid relationship {rel}, ID does not exist")

                if 'plots_node' not in self.plots_graph: self.plots_graph['plots_node'] = []
                if 'plots_relationship' not in self.plots_graph: self.plots_graph['plots_relationship'] = []

                self.plots_graph['plots_node'].extend(final_nodes)
                self.plots_graph['plots_relationship'].extend(final_rels)

                if not self.save_plots_graph():
                    raise OSError(f"Failed to save plot graph: {self.premise_id}")
                self.graph_history.append(deepcopy(self.plots_graph))

                new_foreshadows = [n['overview'] for n in final_nodes if n['foreshadowing']]
                print(f'————————Plot graph update complete; added {len(final_nodes)} nodes————————')
                if new_foreshadows:
                    print(f"New foreshadowing found: {new_foreshadows}")
                return True

            print("Invalid output: plot graph <graph> tag not found!")
            return False

        except Exception as e:
            print('Plot graph update failed: ', str(e))
            import traceback
            traceback.print_exc()
            return False

    #####################################################################################
    #####################################################################################

    
    def find_plot_by_id(self, plot_id: int | str) -> dict | None:
        """
        Retrieve the detailed information of a single plot node based on its unique ID.
        :param plot_id: Public p_K ref or legacy integer/numeric ID.
        :return: dict | None Returns the complete information dictionary of the plot if found; otherwise returns None.
        """
        plot_id = _parse_plot_query_id(plot_id)
        if plot_id is None:
            return None
        for node in self.plots_graph.get('plots_node', []):
            if not isinstance(node, dict):
                continue
            if _parse_plot_query_id(node.get('id')) == plot_id:
                return node
        return None


    def query_plot_adjacency(
        self,
        plot_ref: str,
        direction: str = "both",
        edge_type: str | None = None,
        limit: int = 20,
    ) -> dict:
        """Read one bounded C_P adjacency around a formal plot ref.

        Use this for historical order, cause, and payoff queries. ``plot_ref``
        must be the canonical public form ``p_K``; integers, numeric strings,
        temporary refs, and aliases are rejected rather than guessed.

        ``direction`` is relative to ``plot_ref`` and must be ``incoming``,
        ``outgoing``, or ``both``. ``edge_type`` may be omitted, or be exactly
        one of ``e_temp``, ``e_cas``, and ``e_res``. The stored C_P direction is
        preserved: ``e_temp`` and ``e_cas`` normally point earlier/cause to
        later/effect, while ``e_res`` is ``payoff -> setup``.

        The result pairs every matching edge with its adjacent node, includes
        each node's ``foreshadowing`` flag, and is capped at 50 edges even if a
        larger limit is requested. This method never mutates or saves C_P.
        """

        def error_result(
            message: str,
            graph_node_count: int | None = None,
            graph_edge_count: int | None = None,
        ) -> dict:
            return {
                "ok": False,
                "error": message,
                "query_ref": plot_ref if isinstance(plot_ref, str) else None,
                "direction": direction if isinstance(direction, str) else None,
                "edge_type": edge_type,
                "limit": 0,
                "center_node": None,
                "neighbors": [],
                "total_matching_edges": 0,
                "truncated": False,
                "graph_node_count": graph_node_count,
                "graph_edge_count": graph_edge_count,
            }

        query_id = _parse_formal_plot_ref(plot_ref)
        if query_id is None:
            return error_result(
                "plot_ref must be one canonical formal ref such as 'p_7'; "
                "integer, numeric-string, temporary, padded, and whitespace aliases are forbidden."
            )

        if not isinstance(direction, str):
            return error_result("direction must be 'incoming', 'outgoing', or 'both'.")
        normalized_direction = direction.strip().lower()
        if normalized_direction not in {"incoming", "outgoing", "both"}:
            return error_result("direction must be 'incoming', 'outgoing', or 'both'.")

        normalized_filter = None
        if edge_type is not None:
            if not isinstance(edge_type, str) or edge_type != edge_type.strip():
                return error_result("edge_type must be e_temp, e_cas, e_res, or null.")
            normalized_filter = edge_type
            if normalized_filter not in VALID_REL_TYPES:
                return error_result("edge_type must be e_temp, e_cas, e_res, or null.")

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            return error_result("limit must be a positive integer.")
        bounded_limit = min(limit, MAX_PLOT_ADJACENCY_RESULTS)

        node_by_id: dict[int, dict] = {}
        for node in self.plots_graph.get("plots_node", []):
            if not isinstance(node, dict):
                continue
            node_id = _parse_plot_query_id(node.get("id"))
            if node_id is not None and node_id not in node_by_id:
                node_by_id[node_id] = node

        valid_relations: list[tuple[int, int, str]] = []
        for relation in self.plots_graph.get("plots_relationship", []):
            if not isinstance(relation, (list, tuple)) or len(relation) < 3:
                continue
            source_id = _parse_plot_query_id(relation[0])
            target_id = _parse_plot_query_id(relation[1])
            canonical_type = normalize_edge_type(relation[2])
            if source_id is None or target_id is None or canonical_type is None:
                continue
            if source_id == target_id or source_id not in node_by_id or target_id not in node_by_id:
                continue
            valid_relations.append((source_id, target_id, canonical_type))

        graph_node_count = len(node_by_id)
        graph_edge_count = len(valid_relations)
        center = node_by_id.get(query_id)
        if center is None:
            return error_result(
                f"Plot node {plot_ref} does not exist in C_P.",
                graph_node_count=graph_node_count,
                graph_edge_count=graph_edge_count,
            )

        def public_node(node: dict, node_id: int) -> dict:
            result = deepcopy(node)
            result["ref"] = make_plot_ref(node_id)
            return result

        matches: list[dict] = []
        for source_id, target_id, canonical_type in valid_relations:
            if normalized_filter is not None and canonical_type != normalized_filter:
                continue

            relative_direction = None
            neighbor_id = None
            if source_id == query_id:
                relative_direction = "outgoing"
                neighbor_id = target_id
            elif target_id == query_id:
                relative_direction = "incoming"
                neighbor_id = source_id
            if relative_direction is None:
                continue
            if normalized_direction != "both" and relative_direction != normalized_direction:
                continue

            matches.append({
                "direction": relative_direction,
                "edge": {
                    "source": make_plot_ref(source_id),
                    "target": make_plot_ref(target_id),
                    "edge_type": canonical_type,
                },
                "node": public_node(node_by_id[neighbor_id], neighbor_id),
            })

        total_matches = len(matches)
        return {
            "ok": True,
            "error": None,
            "query_ref": plot_ref,
            "direction": normalized_direction,
            "edge_type": normalized_filter,
            "limit": bounded_limit,
            "center_node": public_node(center, query_id),
            "neighbors": matches[:bounded_limit],
            "total_matching_edges": total_matches,
            "truncated": total_matches > bounded_limit,
            "graph_node_count": graph_node_count,
            "graph_edge_count": graph_edge_count,
        }

    
    def find_all_plots_id_index(self) -> dict[int, int]:
        return {node.get('id'): i for i, node in enumerate(self.plots_graph.get('plots_node', []))}

    
    def find_plots_by_property(self, query: dict) -> list[dict]:
        """
        Search for plots based on one or more precise attribute conditions.
        :param query: dict The query dictionary, where the keys are the attribute names of the plot, and the values are the precise values expected to match. For example: {'chapter': 1, 'importance': 'main plot'}.
        :return: list[dict] Returns a list of scene information dictionaries that fully match the query conditions.
        """
        results = []
        if not query:
            return results
        query = dict(query)
        if "id" in query:
            normalized_id = _parse_plot_query_id(query["id"])
            if normalized_id is None:
                return []
            query["id"] = normalized_id
        for node in self.plots_graph.get('plots_node', []):
            if not isinstance(node, dict):
                continue
            if all(
                (_parse_plot_query_id(node.get(key)) == value)
                if key == "id"
                else (node.get(key) == value)
                for key, value in query.items()
            ):
                results.append(node)
        return results

    
    def find_plots_by_character(self, character_name: str) -> list[dict]:
        """
        Get all the plots involving a specific character, and organize them into a story timeline in chronological order (by chapter and ID).
        :param character_name: str The name of the character to be queried.
        :return: list[dict] Returns a list of information dictionaries of all plots involving the character, sorted in the order of the story's development.
        """
        involved_plots = []
        for node in self.plots_graph.get('plots_node', []):
            if character_name in node.get('characters_involved', []):
                involved_plots.append(node)
        return sorted(involved_plots, key=lambda p: (p.get('chapter', 0), p.get('id', 0)))

    
    def find_plots_by_scene(self, scene_name: str) -> list[dict]:
        """
        Retrieve all plots that occur in a specific scene and sort them in chronological order (by chapter and ID).
        :param scene_name: str The name of the scene to be queried.
        :return: list[dict] Returns a list of information dictionaries of all plots that occur in the scene, sorted in the order of the story's development.
        """
        scene_plots = []
        for node in self.plots_graph.get('plots_node', []):
            if scene_name == node.get('environment', node.get('scene')):
                scene_plots.append(node)
        return sorted(scene_plots, key=lambda p: (p.get('chapter', 0), p.get('id', 0)))

    
    def find_unresolved_foreshadowing(self) -> list[dict]:
        """
        Find all plots marked as foreshadowing (foreshadowing: True) that have not been 'resolved' by any plot.
        :return: list[dict] Returns a list of information dictionaries for all unresolved foreshadowing plots.
        """
        nodes_in_order: list[tuple[int, dict]] = []
        node_ids: set[int] = set()
        for node in self.plots_graph.get("plots_node", []):
            if not isinstance(node, dict):
                continue
            node_id = _parse_plot_query_id(node.get("id"))
            if node_id is None:
                continue
            node_ids.add(node_id)
            if node.get("foreshadowing"):
                nodes_in_order.append((node_id, node))

        foreshadowing_ids = {node_id for node_id, _ in nodes_in_order}
        resolved_ids: set[int] = set()
        for relation in self.plots_graph.get("plots_relationship", []):
            if not isinstance(relation, (list, tuple)) or len(relation) < 3:
                continue
            if normalize_edge_type(relation[2]) != "e_res":
                continue
            payoff_id = _parse_plot_query_id(relation[0])
            setup_id = _parse_plot_query_id(relation[1])
            if (
                payoff_id is None
                or setup_id is None
                or payoff_id == setup_id
                or payoff_id not in node_ids
                or setup_id not in foreshadowing_ids
            ):
                continue
            resolved_ids.add(setup_id)

        return [node for node_id, node in nodes_in_order if node_id not in resolved_ids]


    def fuzzy_search_plot(self, query: str, top_k: int = 3, filters:dict|None=None) -> list[dict]:
        search_res = self.search_plot(query, top_k)
        return search_res


    
    def add_plots(self, new_plots: list[dict]) -> str:
        if not isinstance(new_plots, list) or not new_plots:
            return "Error: Input must be a nonempty list."

        all_nodes = self.plots_graph.get('plots_node', [])
        current_max_id = max([n.get('id', 0) for n in all_nodes]) if all_nodes else 0

        plots_to_add = []
        for i, plot_info in enumerate(new_plots):
            if 'id' in plot_info:
                return f"Error: Do not specify IDs for new plot nodes; IDs are generated automatically."

            invalid_attrs = set(plot_info.keys()) - VALID_PLOT_ATTRIBUTES
            if invalid_attrs:
                return f"Error: Invalid attribute names: {', '.join(invalid_attrs)}."

            plot_info['id'] = current_max_id + 1 + i
            plots_to_add.append(plot_info)

        all_nodes.extend(plots_to_add)
        return f"Success: added {len(plots_to_add)} plot nodes starting at ID {plots_to_add[0]['id']}."

    
    def add_plot_relationship(self, source_id: int, target_id: int, relationship_type: str) -> str:
        canonical = normalize_edge_type(relationship_type)
        if canonical is None:
            return f"Error: Relationship type '{relationship_type}' is invalid; use e_temp, e_cas, or e_res."

        if source_id == target_id:
            return f"Error: A plot node cannot relate to itself (ID: {source_id})."

        existing_ids = set(self.find_all_plots_id_index().keys())
        if source_id not in existing_ids or target_id not in existing_ids:
            return f"Error: Source ID '{source_id}' or target ID '{target_id}' does not exist."

        if canonical == 'e_res':
            target_plot = self.find_plot_by_id(target_id)
            if not target_plot or not target_plot.get('foreshadowing'):
                return f"Error: e_res must target a setup node with foreshadowing=True (ID: {target_id})."

        new_rel = [str(source_id), str(target_id), canonical]
        self.plots_graph['plots_relationship'].append(new_rel)
        return f"Success: added a '{canonical}' relationship from plot {source_id} to {target_id}."

    
    def update_plot_info(self, plot_id: int, updated_info: dict) -> str:
        if not isinstance(updated_info, dict) or not updated_info:
            return "Error: Update information must be a nonempty dictionary."

        if 'id' in updated_info:
            return "Error: A plot node ID cannot be changed."

        invalid_attrs = set(updated_info.keys()) - VALID_PLOT_ATTRIBUTES
        if invalid_attrs:
            return f"Error: Invalid attribute names: {', '.join(invalid_attrs)}."

        plot_index_map = self.find_all_plots_id_index()
        if plot_id not in plot_index_map:
            return f"Error: Plot node with ID '{plot_id}' not found."

        index = plot_index_map[plot_id]
        self.plots_graph['plots_node'][index].update(updated_info)
        return f"Success: updated plot node {plot_id}."

    
    def delete_plot(self, plot_id: int) -> str:
        plot_index_map = self.find_all_plots_id_index()
        if plot_id not in plot_index_map:
            return f"Error: Plot node with ID '{plot_id}' not found."

        del self.plots_graph['plots_node'][plot_index_map[plot_id]]

        str_plot_id = str(plot_id)
        relations = self.plots_graph.get('plots_relationship', [])
        updated_relations = [rel for rel in relations if str_plot_id not in (rel[0], rel[1])]
        self.plots_graph['plots_relationship'] = updated_relations

        return f"Success: deleted plot node {plot_id} and all its relationships."

    #####################################################################################
    #####################################################################################


    def save_plots_graph(self):
        save_state = set_plots_graph_data(self.premise_id, self.plots_graph)
        if save_state:
            print('----------Saved successfully!----------')
        else:
            print('----------Save failed!----------')
        return bool(save_state)

    def rollback_last_graph(self):
        self.graph_history.pop()
        self.plots_graph = self.graph_history[len(self.graph_history) - 1]
        if not self.save_plots_graph():
            raise OSError(f"Failed to save rolled-back plot graph: {self.premise_id}")

