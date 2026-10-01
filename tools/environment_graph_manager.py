
import os
import re
import ast
import json
from copy import deepcopy

from agents import AgentGlobalConfig, DEFAULT_MODEL_NAME
from data import environments_graph_data, set_environments_graph_data
from tools.environment_ids import (
    CONC_BRANCH,
    PHYS_BRANCH,
    ensure_environment_graph_ids,
    find_lca,
    get_environment_name_by_ref,
    get_environment_node_by_ref,
    get_next_environment_ref,
    walk_ancestors,
)
from tools.graph_refs import make_environment_ref, normalize_environment_ref

os.environ['HF_ENDPOINT'] = ''
from sentence_transformers import SentenceTransformer, util
import numpy as np


client = AgentGlobalConfig.GPTCLIENT


def _normalize_environment_query_ref(value):
    """Canonicalize public e_K, integer, and numeric-string lookup values."""
    # ``bool`` is an ``int`` subclass in Python; without this guard ``True``
    # would silently become e_1 and could make an invalid Manager query look
    # grounded by returning the physical root.
    if isinstance(value, bool):
        return None
    normalized = normalize_environment_ref(value, allow_temp=False)
    if normalized is not None:
        return normalized
    if isinstance(value, str) and value.strip().isdigit():
        numeric_id = int(value.strip())
        return make_environment_ref(numeric_id) if numeric_id > 0 else None
    return value.strip() if isinstance(value, str) else value


VALID_PHYS_ATTRIBUTES = {
    "id", "name", "branch", "parent_id", "children_ids",
    "description", "location", "minutia", "atmosphere",
}
VALID_CONC_ATTRIBUTES = {
    "id", "name", "branch", "parent_id", "children_ids",
    "description", "attributes",
}


_ENV_NO_CHANGE_SENTINELS = {
    "no change", "no changes", "unchanged", "same as before", "n/a",
    "\u65e0\u53d8\u5316", "\u6ca1\u6709\u53d8\u5316", "\u4e0d\u53d8", "\u65e0\u9700\u66f4\u65b0", "\u4fdd\u6301\u4e0d\u53d8",
}


def _is_environment_no_change_sentinel(value) -> bool:
    if not isinstance(value, str):
        return False
    normalized = re.sub(r"[\s.!?\u3002\uFF01\uFF1F]+$", "", value.strip()).casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized in _ENV_NO_CHANGE_SENTINELS


def _plan_minutia_delta(
    existing: list[str],
    *,
    legacy_minutia=None,
    minutia_add=None,
    minutia_remove=None,
) -> tuple[list[str], list[str], list[str]]:
    """Validate and calculate an exact physical-scene minutia state delta.

    ``minutia`` is retained as a legacy alias for additions.  Removals are
    deliberately stricter: every requested string must be copied byte-for-byte
    (at the Python string level) from the current node state.  This prevents a
    lossy chapter summary from replacing the proposal-backed environment file.
    """
    if not isinstance(existing, list) or any(not isinstance(item, str) for item in existing):
        raise ValueError("The current environment node minutia must be an array of strings.")

    additions: list[str] = []
    for field_name, raw_items in (
        ("minutia", legacy_minutia),
        ("minutia_add", minutia_add),
    ):
        if raw_items is None:
            raw_items = []
        if not isinstance(raw_items, list) or any(
            not isinstance(detail, str) for detail in raw_items
        ):
            raise ValueError(f"Environment hot update field {field_name} must be an array of strings.")
        for detail in raw_items:
            normalized = detail.strip()
            if normalized and normalized not in additions:
                additions.append(normalized)

    if minutia_remove is None:
        minutia_remove = []
    if not isinstance(minutia_remove, list) or any(
        not isinstance(detail, str) for detail in minutia_remove
    ):
        raise ValueError("Environment hot update field minutia_remove must be an array of strings.")

    removals: list[str] = []
    for detail in minutia_remove:
        if not detail or detail != detail.strip():
            raise ValueError(
                "Environment hot update minutia_remove entries must exactly match existing nonempty minutia."
            )
        if detail not in removals:
            removals.append(detail)

    overlap = [detail for detail in removals if detail in additions]
    if overlap:
        raise ValueError(
            "Environment hot update cannot add and remove the same minutia: " + ", ".join(overlap)
        )
    missing = [detail for detail in removals if detail not in existing]
    if missing:
        raise ValueError(
            "Environment hot update cannot remove nonexistent minutia (exact match required): "
            + ", ".join(missing)
        )

    next_minutia = [detail for detail in existing if detail not in removals]
    for detail in additions:
        if detail not in next_minutia:
            next_minutia.append(detail)
    return next_minutia, additions, removals


def _parse_environment_array_payload(raw_text: str) -> list[dict]:
    """Parse extraction output without replacing apostrophes inside strings."""
    if not isinstance(raw_text, str):
        raise TypeError("environment payload must be text")
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as json_error:
        try:
            parsed = ast.literal_eval(cleaned)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            raise json_error
    if not isinstance(parsed, list):
        raise ValueError("environment payload must decode to an array")
    return parsed


class EnvironmentTreeManager:

    def __init__(self, premise_id):
        self.premise_id = premise_id
        self.model_name = DEFAULT_MODEL_NAME
        self.history = []
        self.graph_history = []
        self.environments_graph = environments_graph_data(premise_id)
        self.environments_graph, _ = ensure_environment_graph_ids(self.environments_graph)
        self.graph_history.append(deepcopy(self.environments_graph))
        self.embedding_model = None
        self.vector_keys: list[str] = []
        self.vector_matrix = None
        self.is_vector_stale = True


    def _nodes(self) -> list[dict]:
        return self.environments_graph.get("environments_node", [])

    def _nodes_by_id(self) -> dict[str, dict]:
        return {n.get("id"): n for n in self._nodes()}

    def _refresh_after_mutation(self):
        self.environments_graph, _ = ensure_environment_graph_ids(self.environments_graph)
        self.is_vector_stale = True


    def _ensure_env_vectors_ready(self):
        if self.embedding_model is None:
            from tools.embedding_singleton import get_embedding_model
            self.embedding_model = get_embedding_model()

        if not self.is_vector_stale:
            return

        keys: list[str] = []
        texts: list[str] = []

        for node in self._nodes():
            nid = node.get("id")
            name = node.get("name", "").strip()
            branch = node.get("branch", PHYS_BRANCH)
            if not nid or not name:
                continue
            if branch == PHYS_BRANCH:
                minutia = node.get("minutia", []) or []
                minutia_text = ", ".join(str(x) for x in minutia)
                rich = (
                    f"[physical] id={nid} name={name} location={node.get('location','')} "
                    f"description={node.get('description','')} details={minutia_text} "
                    f"atmosphere={node.get('atmosphere','')}"
                )
            else:
                attrs = node.get("attributes", {}) or {}
                attr_text = "; ".join(f"{k}={v}" for k, v in attrs.items())
                rich = (
                    f"[conceptual] id={nid} name={name} description={node.get('description','')} "
                    f"attributes={attr_text}"
                )
            keys.append(nid)
            texts.append(rich)

        if not texts:
            self.vector_keys = []
            self.vector_matrix = None
            self.is_vector_stale = False
            return

        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        self.vector_keys = keys
        self.vector_matrix = np.array(embeddings)
        self.is_vector_stale = False

    def search_environment(self, query: str, top_k: int = 5, score_threshold: float = 0.30) -> list[dict]:
        self._ensure_env_vectors_ready()
        if self.vector_matrix is None or not self.vector_keys:
            return []

        query_vector = self.embedding_model.encode(query, convert_to_tensor=False)
        scores = util.cos_sim(query_vector, self.vector_matrix)[0].cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[dict] = []
        nodes_by_id = self._nodes_by_id()
        for idx in top_indices:
            score = float(scores[idx])
            if score < score_threshold:
                continue
            nid = self.vector_keys[idx]
            node = nodes_by_id.get(nid)
            if not node:
                continue
            results.append({
                "id": nid,
                "name": node.get("name", ""),
                "branch": node.get("branch", PHYS_BRANCH),
                "score": round(score, 4),
                "description_snippet": (node.get("description", "") or "")[:160],
            })
        return results

    def fuzzy_search_env(self, query: str, top_k: int = 3) -> list[dict]:
        """Semantic fuzzy retrieval exposed to the Environment Agent."""
        return self.search_environment(query, top_k)


    def find_environments_info(self, environments_ref_or_name: list) -> list[dict]:
        """Batch lookup by e_K/integer ID or exact name."""
        query_values = (
            environments_ref_or_name
            if isinstance(environments_ref_or_name, list)
            else [environments_ref_or_name]
        )
        ref_set = {_normalize_environment_query_ref(value) for value in query_values}
        results: list[dict] = []
        for node in self._nodes():
            if node.get('id') in ref_set or node.get('name') in ref_set:
                results.append(node)
        return results

    def find_all_environments_name_index(self) -> dict[str, str]:
        """Index of name -> id."""
        return {n.get("name"): n.get("id") for n in self._nodes() if n.get("name")}

    def find_environments_by_property(self, query: dict) -> list[dict]:
        """Exact match by field; only top-level fields are matched."""
        results: list[dict] = []
        if not query:
            return results
        query = dict(query)
        for field in ("id", "parent_id"):
            if field in query and query[field] is not None:
                query[field] = _normalize_environment_query_ref(query[field])
        for node in self._nodes():
            if all(node.get(k) == v for k, v in query.items()):
                results.append(node)
        return results

    def query_ancestors_with_inheritance(self, node_id: str) -> dict:
        """
        Attribute-inheritance traversal for the concept tree (T_conc):
        starting from node_id, walk up the parent chain and merge `attributes`
        (child fields override the same fields on parents). Returns a dict containing
        `effective_attributes` and an ancestor-chain summary. Also callable for
        physical-tree nodes — in that case, it aggregates location/atmosphere
        context along the ancestor chain to aid retrieval.
        """
        node_id = _normalize_environment_query_ref(node_id)
        nodes_by_id = self._nodes_by_id()
        target = nodes_by_id.get(node_id)
        if not target:
            return {"id": node_id, "found": False}

        ancestors = walk_ancestors(self.environments_graph, node_id)
        chain = list(reversed(ancestors)) + [target]

        if target.get("branch") == CONC_BRANCH:
            effective: dict = {}
            for n in chain:
                attrs = n.get("attributes", {}) or {}
                effective.update(attrs)
            return {
                "id": node_id,
                "found": True,
                "branch": CONC_BRANCH,
                "name": target.get("name", ""),
                "effective_attributes": effective,
                "ancestor_chain": [
                    {"id": n.get("id"), "name": n.get("name"), "attributes": n.get("attributes", {})}
                    for n in chain
                ],
            }

        return {
            "id": node_id,
            "found": True,
            "branch": PHYS_BRANCH,
            "name": target.get("name", ""),
            "ancestor_chain": [
                {"id": n.get("id"), "name": n.get("name"), "location": n.get("location", "")}
                for n in chain
            ],
        }

    def check_containment_path(self, from_id: str, to_id: str) -> dict:
        """
        Connectivity check for the physical tree (T_phys):
        if `from` and `to` belong to the same physical subtree and can reach each other
        via their lowest common ancestor, the path is treated as legal; otherwise a
        teleportation risk is flagged.
        """
        nodes_by_id = self._nodes_by_id()
        a = nodes_by_id.get(from_id)
        b = nodes_by_id.get(to_id)
        if not a or not b:
            return {"reachable": False, "reason": "node not found", "from_id": from_id, "to_id": to_id}
        if a.get("branch") != PHYS_BRANCH or b.get("branch") != PHYS_BRANCH:
            return {"reachable": False, "reason": "containment only applies to the physical tree (T_phys)"}
        if from_id == to_id:
            return {"reachable": True, "lca_id": from_id, "path": [from_id]}

        lca_id = find_lca(self.environments_graph, from_id, to_id)
        if not lca_id:
            return {"reachable": False, "reason": "the two nodes are not in the same physical subtree; teleportation risk"}

        up = [from_id] + [n["id"] for n in walk_ancestors(self.environments_graph, from_id)]
        up_to_lca: list[str] = []
        for nid in up:
            up_to_lca.append(nid)
            if nid == lca_id:
                break

        down_chain = [to_id] + [n["id"] for n in walk_ancestors(self.environments_graph, to_id)]
        down_to_lca: list[str] = []
        for nid in down_chain:
            down_to_lca.append(nid)
            if nid == lca_id:
                break
        path = up_to_lca + list(reversed(down_to_lca[:-1]))
        return {"reachable": True, "lca_id": lca_id, "path": path}

    def list_descendants(self, node_id: str) -> list[dict]:
        """Return all descendant nodes under node_id's subtree (including itself; brief info, root first)."""
        nodes_by_id = self._nodes_by_id()
        if node_id not in nodes_by_id:
            return []
        out: list[dict] = []
        stack = [node_id]
        seen: set[str] = set()
        while stack:
            nid = stack.pop(0)
            if nid in seen:
                continue
            seen.add(nid)
            node = nodes_by_id.get(nid)
            if not node:
                continue
            out.append({
                "id": nid,
                "name": node.get("name", ""),
                "branch": node.get("branch", PHYS_BRANCH),
                "depth_hint": len(walk_ancestors(self.environments_graph, nid)),
            })
            stack.extend(node.get("children_ids", []))
        return out


    def add_phys_node(
        self,
        name: str,
        parent_id: str | None = None,
        description: str = "",
        location: str = "",
        minutia: list[str] | None = None,
        atmosphere: str = "",
    ) -> str:
        if not name:
            return "Error: name cannot be empty."
        if any(n.get("name") == name for n in self._nodes()):
            return f"Error: name '{name}' already exists."
        if parent_id is None:
            parent_id = self.environments_graph.get("phys_root_id")
        if parent_id not in self._nodes_by_id():
            return f"Error: parent_id '{parent_id}' does not exist."
        if self._nodes_by_id()[parent_id].get("branch") != PHYS_BRANCH:
            return "Error: the parent node must belong to T_phys."

        new_id = get_next_environment_ref(self.environments_graph)
        node = {
            "id": new_id,
            "name": name,
            "branch": PHYS_BRANCH,
            "parent_id": parent_id,
            "children_ids": [],
            "description": description,
            "location": location,
            "minutia": list(minutia or []),
            "atmosphere": atmosphere,
        }
        self.environments_graph["environments_node"].append(node)
        self._refresh_after_mutation()
        return new_id

    def add_conc_node(
        self,
        name: str,
        parent_id: str | None = None,
        description: str = "",
        attributes: dict | None = None,
    ) -> str:
        if not name:
            return "Error: name cannot be empty."
        if any(n.get("name") == name for n in self._nodes()):
            return f"Error: name '{name}' already exists."
        if parent_id is None:
            parent_id = self.environments_graph.get("conc_root_id")
        if parent_id not in self._nodes_by_id():
            return f"Error: parent_id '{parent_id}' does not exist."
        if self._nodes_by_id()[parent_id].get("branch") != CONC_BRANCH:
            return "Error: the parent node must belong to T_conc."

        new_id = get_next_environment_ref(self.environments_graph)
        node = {
            "id": new_id,
            "name": name,
            "branch": CONC_BRANCH,
            "parent_id": parent_id,
            "children_ids": [],
            "description": description,
            "attributes": dict(attributes or {}),
        }
        self.environments_graph["environments_node"].append(node)
        self._refresh_after_mutation()
        return new_id

    def add_environments(self, new_environments: list[dict]) -> str:
        if not isinstance(new_environments, list) or not new_environments:
            return "Error: input must be a nonempty list."

        for env in new_environments:
            if not isinstance(env, dict):
                continue
            name = env.get("name")
            if not name:
                return f"Error: node {env} is missing name."
            branch = env.get("branch") or PHYS_BRANCH
            parent_id = env.get("parent_id")
            if branch == PHYS_BRANCH:
                ret = self.add_phys_node(
                    name=name,
                    parent_id=parent_id,
                    description=env.get("description", ""),
                    location=env.get("location", ""),
                    minutia=env.get("minutia", []),
                    atmosphere=env.get("atmosphere", ""),
                )
            else:
                ret = self.add_conc_node(
                    name=name,
                    parent_id=parent_id,
                    description=env.get("description", ""),
                    attributes=env.get("attributes", {}),
                )
            if isinstance(ret, str) and ret.startswith("Error"):
                return ret
        return f"Success: added {len(new_environments)} environment nodes."


    def update_environment_info(self, environment_ref: str, updated_info: dict) -> str:
        if not isinstance(updated_info, dict) or not updated_info:
            return "Error: updated_info must be a nonempty dictionary."
        node = get_environment_node_by_ref(self.environments_graph, environment_ref)
        if not node:
            return f"Error: node '{environment_ref}' not found."

        patch = {k: v for k, v in updated_info.items() if k not in ("id", "branch", "parent_id", "children_ids")}
        valid = VALID_PHYS_ATTRIBUTES if node.get("branch") == PHYS_BRANCH else VALID_CONC_ATTRIBUTES
        virtual_phys_fields = {"minutia_add", "minutia_remove"}
        valid_input = valid | virtual_phys_fields if node.get("branch") == PHYS_BRANCH else valid
        invalid = set(patch.keys()) - valid_input
        if invalid:
            return f"Error: invalid attributes {invalid} (branch={node.get('branch')})."

        if node.get("branch") == PHYS_BRANCH and any(
            field in patch for field in ("minutia", "minutia_add", "minutia_remove")
        ):
            try:
                next_minutia, _, _ = _plan_minutia_delta(
                    list(node.get("minutia", []) or []),
                    legacy_minutia=patch.pop("minutia", None),
                    minutia_add=patch.pop("minutia_add", None),
                    minutia_remove=patch.pop("minutia_remove", None),
                )
            except ValueError as exc:
                return f"Error: {exc}"
            patch["minutia"] = next_minutia
        if "attributes" in patch and isinstance(patch["attributes"], dict):
            merged = dict(node.get("attributes", {}) or {})
            merged.update(patch["attributes"])
            patch["attributes"] = merged

        node.update(patch)
        self._refresh_after_mutation()
        return f"Success: node '{node.get('name')}' updated."

    def reparent_node(self, node_id: str, new_parent_id: str) -> str:
        nodes_by_id = self._nodes_by_id()
        node = nodes_by_id.get(node_id)
        new_parent = nodes_by_id.get(new_parent_id)
        if not node or not new_parent:
            return "Error: node_id or new_parent_id does not exist."
        if node.get("branch") != new_parent.get("branch"):
            return "Error: moving across branches is not allowed."
        if node_id == new_parent_id:
            return "Error: a node cannot be its own parent."
        ancestor_ids = {n.get("id") for n in walk_ancestors(self.environments_graph, new_parent_id)}
        if node_id in ancestor_ids:
            return "Error: a node cannot be moved under its own descendant."
        node["parent_id"] = new_parent_id
        self._refresh_after_mutation()
        return f"Success: node '{node.get('name')}' reparented under '{new_parent.get('name')}'."

    def init_world_concept_tree(self, taxonomy: list[dict]) -> str:
        if not isinstance(taxonomy, list):
            return "Error: taxonomy must be a list."
        conc_root_id = self.environments_graph.get("conc_root_id")
        created = 0

        def _recurse(items: list[dict], parent_id: str):
            nonlocal created
            for item in items:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                if item["name"] in self.find_all_environments_name_index():
                    new_id = self.find_all_environments_name_index()[item["name"]]
                else:
                    new_id = self.add_conc_node(
                        name=item["name"],
                        parent_id=parent_id,
                        description=item.get("description", ""),
                        attributes=item.get("attributes", {}),
                    )
                    if isinstance(new_id, str) and new_id.startswith("Error"):
                        continue
                    created += 1
                children = item.get("children")
                if isinstance(children, list) and children:
                    _recurse(children, new_id)

        _recurse(taxonomy, conc_root_id)
        return f"Success: added {created} T_conc nodes."


    def delete_environment(self, environment_ref: str) -> str:
        node = get_environment_node_by_ref(self.environments_graph, environment_ref)
        if not node:
            return f"Error: node '{environment_ref}' not found."
        if node.get("id") in (self.environments_graph.get("phys_root_id"), self.environments_graph.get("conc_root_id")):
            return "Error: root nodes cannot be deleted."
        parent_id = node.get("parent_id")
        for child in self._nodes():
            if child.get("parent_id") == node.get("id"):
                child["parent_id"] = parent_id
        self.environments_graph["environments_node"] = [
            n for n in self._nodes() if n.get("id") != node.get("id")
        ]
        self._refresh_after_mutation()
        return f"Success: deleted node '{node.get('name')}'."


    def extract_environments(self, passage: str) -> list[dict]:
        mutable_phys_states = [
            {
                "name": node.get("name", ""),
                "minutia": list(node.get("minutia", []) or []),
                "atmosphere": node.get("atmosphere", ""),
            }
            for node in self._nodes()
            if node.get("branch") == PHYS_BRANCH
            and node.get("id") != self.environments_graph.get("phys_root_id")
        ]
        task_prompt = f"""
# Role: You are a novel environment architect.
# Task: From the chapter text, identify the physical scenes where characters **actually are present** and where plot events occur;
  match each scene to an already-authorized node in the current physical-space working tree (T_phys).

# Output requirements
- Strictly output a JSON array wrapped by <environments>[...]</environments>;
- Each element has the structure:
  {{
    "name": "exact existing T_phys node name",     // copy exactly from the index below
    "minutia_add": ["new concrete state/detail observed in this chapter"],
    "minutia_remove": ["exact current minutia string invalidated by this chapter"],
    "atmosphere": "current atmosphere, or empty string when unchanged"
  }}

# Current mutable physical-scene state (copy names/removals exactly):
{json.dumps(mutable_phys_states, ensure_ascii=False, indent=2)}

# Hard boundary
- This is a state-refinement hot update, not a creation step.
- NEVER invent, rename, or propose a physical scene absent from the index above.
- If prose uses a descriptive alias for a listed scene, output the exact indexed name.
- Emit at most one object for each scene name.
- Incidental mentioned places where no character is physically present must not be output.
- This hot update must not rewrite a scene's stable description, geographic location,
  branch, or parent. Emit only newly observed minutia and a genuinely changed atmosphere.
- Use minutia_remove only when the chapter explicitly invalidates physical state (for
  example, an item is taken or an unlocked door becomes locked). Every removal must be
  copied EXACTLY from that scene's current minutia array. For a state transition, remove
  the exact old state and add the exact new state; never replace the array with a summary.
- Use empty arrays and an empty atmosphere when nothing changed.

# Chapter text
{passage}
"""
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": task_prompt}],
                temperature=0.1,
            )
            text = response.choices[0].message.content
            match = re.search(r'<environments>(.*?)</environments>', text, re.DOTALL)
            if not match:
                return []
            try:
                return _parse_environment_array_payload(match.group(1))
            except (json.JSONDecodeError, ValueError, TypeError):
                return []
        except Exception as e:
            print(f"Environment extraction failed: {e}")
            return []

    def update_graph_by_passage(self, passage: str) -> list[str]:
        candidates = self.extract_environments(passage)
        if not candidates:
            return []
        if not isinstance(candidates, list):
            raise ValueError("Environment hot update output must be an array.")

        name_to_id = self.find_all_environments_name_index()
        env_names: list[str] = []
        seen_names: set[str] = set()
        plans: list[tuple[str, dict]] = []

        for item in candidates:
            if not isinstance(item, dict):
                raise ValueError(f"Environment hot update node must be an object: {item!r}")
            name = (item.get("name") or "").strip()
            if not name:
                raise ValueError("Environment hot update node lacks a nonempty name; refusing to skip it silently.")
            if name in seen_names:
                raise ValueError(f"Environment hot update returned scene {name!r} more than once; refusing order-dependent updates.")
            seen_names.add(name)
            environment_id = name_to_id.get(name)
            if environment_id is None:
                raise ValueError(
                    f"The passage contains unauthorized physical scene {name!r}; passage hot update cannot create "
                    "new environment nodes outside R_active."
                )

            environment_node = self._nodes_by_id().get(environment_id)
            if not environment_node or environment_node.get("branch") != PHYS_BRANCH:
                raise ValueError(
                    f"Environment hot update scene {name!r} is not a T_phys node in the current M_sub."
                )
            if environment_id in {
                self.environments_graph.get("phys_root_id"),
                self.environments_graph.get("conc_root_id"),
            }:
                raise ValueError("Environment hot update cannot modify T_E root nodes.")

            allowed_fields = {
                "name", "description", "location", "minutia", "minutia_add",
                "minutia_remove", "atmosphere",
            }
            invalid_fields = set(item) - allowed_fields
            if invalid_fields:
                raise ValueError(
                    "Environment hot update contains invalid fields: " + ", ".join(sorted(invalid_fields))
                )

            existing_minutia = list(environment_node.get("minutia", []) or [])
            next_minutia, additions, removals = _plan_minutia_delta(
                existing_minutia,
                legacy_minutia=item.get("minutia", []),
                minutia_add=item.get("minutia_add", []),
                minutia_remove=item.get("minutia_remove", []),
            )

            raw_atmosphere = item.get("atmosphere", "")
            if raw_atmosphere is None:
                raw_atmosphere = ""
            if not isinstance(raw_atmosphere, str):
                raise ValueError("Environment hot update atmosphere must be text.")
            atmosphere = raw_atmosphere.strip()

            patch = {}
            if next_minutia != existing_minutia:
                if additions:
                    patch["minutia_add"] = additions
                if removals:
                    patch["minutia_remove"] = removals
            if (
                atmosphere
                and not _is_environment_no_change_sentinel(atmosphere)
                and atmosphere != environment_node.get("atmosphere", "")
            ):
                patch["atmosphere"] = atmosphere

            plans.append((environment_id, patch))
            env_names.append(name)

        changed_plans = [(environment_id, patch) for environment_id, patch in plans if patch]
        if not changed_plans:
            return env_names

        graph_before = deepcopy(self.environments_graph)
        stale_before = self.is_vector_stale
        try:
            for environment_id, patch in changed_plans:
                update_result = self.update_environment_info(environment_id, patch)
                update_text = update_result.strip() if isinstance(update_result, str) else ""
                if (
                    not update_text
                    or update_text.startswith("Error")
                    or update_text.casefold().startswith("error")
                ):
                    raise ValueError(f"Environment node update failed: {update_result}")
            if not self.save_env_graph():
                raise OSError(f"Failed to save the environment tree: {self.premise_id}")
        except (ValueError, OSError):
            self.environments_graph = graph_before
            self.is_vector_stale = stale_before
            raise
        self.graph_history.append(deepcopy(self.environments_graph))
        return env_names


    def save_env_graph(self):
        self.environments_graph, _ = ensure_environment_graph_ids(self.environments_graph)
        ok = set_environments_graph_data(self.premise_id, self.environments_graph)
        print('----------Environment tree saved successfully!----------' if ok else '----------Environment tree save failed!----------')
        return bool(ok)

    def rollback_last_graph(self):
        if len(self.graph_history) >= 2:
            self.graph_history.pop()
        self.environments_graph = deepcopy(self.graph_history[-1])
        if not self.save_env_graph():
            raise OSError(f"Failed to save the rolled-back environment tree: {self.premise_id}")


EnvironmentGraphManager = EnvironmentTreeManager
