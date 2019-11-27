import gzip
import json
import os
import pickle
from typing import Any, List, Dict, Union

import numpy as np


def load_text_file(path: str) -> str:
    with open(path, "r") as f:
        text = f.read()
    return text


def save_text_file(string: str, path: str) -> None:
    with open(path, "w") as f:
        f.write(string)
    return


def pickle_obj(obj: object, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    return


def unpickle_obj(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def numpy_store(data: np.ndarray, path: str) -> None:
    np.save(path, data)
    return


def numpy_load(path: str) -> np.ndarray:
    return np.load(path)


def store_json(data, path: str):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except ValueError as e:
        print("Invalid path: %s" % e)
        return None


def load_json(path: str) -> Union[List, Dict]:
    try:
        with open(path, "r") as read_file:
            data = json.load(read_file)
    except ValueError as e:
        print("Invalid json: %s" % e)
        return None
    return data


def indent_json(path_in: str) -> None:
    data = load_json(path_in)

    with open(path_in, "w") as write_file:
        json.dump(data, write_file, indent=4)
    return


def unindent_json(path_in: str) -> None:
    data = load_json(path_in)

    with open(path_in, "w") as write_file:
        json.dump(data, write_file)
    return


def merge_jsons(paths_in: List[str], path_out: str) -> None:
    data = []
    for path_in in paths_in:
        data += [load_json(path_in)]

    store_json(data, path_out)


def aggregate_jsons(root_path: str, path_out: str) -> List:
    json_list = []

    for root, dirs, files in os.walk(root_path):
        for f in files:
            if os.path.splitext(f)[1].lower() == ".json":
                json_list.append(os.path.join(root, f))

    merge_jsons(json_list, path_out)
