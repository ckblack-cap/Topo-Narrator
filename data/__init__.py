import os
import json
import threading

from tools.environment_ids import ensure_environment_graph_ids

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

def data_path():
    return DATA_DIR

def check_init_data(pid):
    characters_data = {
        "characters_node": [],
        "characters_relationship": []
    }

    environments_data, _ = ensure_environment_graph_ids({"environments_node": []})

    plots_data = {
        "plots_node": [],
        "plots_relationship": []
    }

    STORY_DIR = os.path.join(DATA_DIR, str(pid))

    os.makedirs(STORY_DIR, exist_ok=True)

    CHARACTERS_FILE = os.path.join(STORY_DIR, 'characters.json')
    ENVIRONMENTS_FILE = os.path.join(STORY_DIR, 'environments.json')
    PLOTS_FILE = os.path.join(STORY_DIR, 'plots.json')

    if not os.path.exists(CHARACTERS_FILE):
        if not set_file(CHARACTERS_FILE, characters_data):
            raise OSError(f"inti error {CHARACTERS_FILE}")

    if not os.path.exists(ENVIRONMENTS_FILE):
        if not set_file(ENVIRONMENTS_FILE, environments_data):
            raise OSError(f"inti error {ENVIRONMENTS_FILE}")

    if not os.path.exists(PLOTS_FILE):
        if not set_file(PLOTS_FILE, plots_data):
            raise OSError(f"inti error  {PLOTS_FILE}")

def characters_graph_data(pid):
    check_init_data(pid)
    STORY_DIR = os.path.join(DATA_DIR, str(pid))
    CHARACTERS_FILE = os.path.join(STORY_DIR, 'characters.json')
    return get_file(CHARACTERS_FILE)

def set_characters_graph_data(pid, new_data):
    check_init_data(pid)
    STORY_DIR = os.path.join(DATA_DIR, str(pid))
    CHARACTERS_FILE = os.path.join(STORY_DIR, 'characters.json')
    return set_file(CHARACTERS_FILE, new_data)

def environments_graph_data(pid):

    data = raw_environments_graph_data(pid)
    if isinstance(data, dict):
        normalized, _ = ensure_environment_graph_ids(data)
        return normalized
    return data


def raw_environments_graph_data(pid):
    check_init_data(pid)
    STORY_DIR = os.path.join(DATA_DIR, str(pid))
    ENVIRONMENTS_FILE = os.path.join(STORY_DIR, 'environments.json')
    return get_file(ENVIRONMENTS_FILE)

def set_environments_graph_data(pid, new_data):
    check_init_data(pid)
    STORY_DIR = os.path.join(DATA_DIR, str(pid))
    ENVIRONMENTS_FILE = os.path.join(STORY_DIR, 'environments.json')
    normalized, _ = ensure_environment_graph_ids(new_data if isinstance(new_data, dict) else {})
    return set_file(ENVIRONMENTS_FILE, normalized)

def plots_graph_data(pid):
    check_init_data(pid)
    STORY_DIR = os.path.join(DATA_DIR, str(pid))
    PLOTS_FILE = os.path.join(STORY_DIR, 'plots.json')
    return get_file(PLOTS_FILE)

def set_plots_graph_data(pid, new_data):
    check_init_data(pid)
    STORY_DIR = os.path.join(DATA_DIR, str(pid))
    PLOTS_FILE = os.path.join(STORY_DIR, 'plots.json')
    return set_file(PLOTS_FILE, new_data)

def get_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{file_path}: {exc}")
        return False

def set_file(file_path, new_data):
    temp_path = f"{file_path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, file_path)
        return True
    except (OSError, TypeError, ValueError) as exc:
        print(f"{file_path}: {exc}")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        return False


