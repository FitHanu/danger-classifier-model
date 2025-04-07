import json
import os
import pandas as pd
from pathlib import Path
from logging_cfg import get_logger

l = get_logger(__name__)

cwd = Path(__file__).resolve().parent.parent
datasets_file_path = os.path.join(cwd, "datasets.json")
config_file_path = os.path.join(cwd, "config.json")


def get_dataset_info(key: str):
    """
    Get dataset meta from `datasets.json` based on the key
    """
    with open(datasets_file_path) as f:
        data = json.load(f)
        try:
            obj = data[key]
        except KeyError:
            msg = f"No dataset found in {datasets_file_path} for key: {key}"
            raise KeyError(msg)
        return {
            "key":            key,
            "name":           obj["name"],
            "format":         obj["format"],
            "kaggle_path":    obj["kaggle_path"],
            "url":            obj["url"],
            "csv_meta_path":  obj["csv_meta_path"],
            "data_path":      obj["data_path"],
        }


def append_empty_mapping_to_config(ds, overwrite: bool = False):
    """
    Append empty class mapping to `config.json`. Should be called with `overwrite=False`\n
    Unless you want to reset the class mapping to remap manually
    """

    from ds.dataset import DataSet

    # if not isinstance(ds, DataSet):
    #     raise TypeError("ds must be an instance of ds.DataSet")

    l.info(f"Writing config to {config_file_path}, Overiding: {overwrite}")
    parent_property = "class_mapping"
    ds_key = ds.get_key()

    def is_key_empty():
        """
        True if
        1. key is empty in config.json
        2. content in key object is empty
        """
        if parent_property not in data:
            return True
        return not bool(data[parent_property].get(ds_key))

    with open(config_file_path, "r") as f:
        data = json.load(f)
        if parent_property not in data:
            data[parent_property] = {}
        else:
            if not is_key_empty() and overwrite == False:
                l.warning(
                    f"Overwrite is set to off but {ds_key} is already exists. Skipping ..."
                )
                return
        data[parent_property][ds_key] = {}
        l.info(f"Appending empty classes to config.{parent_property}.{ds_key} ...")
        """
        Append empty 
        "ds_key": {
            class1: "",
            class2: "",
            ...
        }
        template for manual dataset class mapping
        """
        column = ds.class_names
        for label in column:
            if label not in data[parent_property][ds_key]:
                data[parent_property][ds_key][label] = 0
    with open(config_file_path, "w") as file:
        json.dump(data, file, indent=2)
    l.info(
        f"Appended {column.size} empty classes to config.{parent_property}.{ds_key}."
    )


def init_default_class_name():
    """
    Initialize default class names to `config.json`
    """
    from ds.dataset import DataSet

    parent_property = "class_mapping"
    default_cn_property = "default"
    df = pd.read_csv(DataSet.DEFAULT_CLASSNAME_PATH)
    with open(config_file_path, "r") as f:
        data = json.load(f)
        data[parent_property][default_cn_property] = {}
        for _, row in df.iterrows():
            data[parent_property][default_cn_property][row["id"]] = row["class_name"]
    with open(config_file_path, "w") as file:
        json.dump(data, file, indent=2)
    l.info(
        f"default class names written to config.{parent_property}.{default_cn_property}"
    )


def get_post_class_mapping(key: str):
    """
    Get class mapping from `config.json` based on the dataset key\n
    Returns a `dict` of class mapping for the dataset\n
    Throws `KeyError` if key is not found
    """
    parent_property = "class_mapping"
    default_class = {}
    dataset_class_map = {}
    result = {}
    with open(config_file_path, "r") as f:
        data = json.load(f)
        try:
            if key not in data[parent_property]:
                msg = f"No class mapping found for key: {key}, have you initialized it to {config_file_path}?"
                raise KeyError(msg)
            default_class = data[parent_property]["default"]
            default_class = {int(k): v for k, v in default_class.items()}
            dataset_class_map = data[parent_property][key]
        except Exception as e:
            l.error(e)
            raise e

    for k, v in dataset_class_map.items():
        if v != 0:
            result[k] = [v, default_class[v]]

    return result


def get_default_class_mapping():
    """
    Get default class mapping from `config.json`\n
    Returns a `dict` of default class mapping
    """
    parent_property = "class_mapping"
    default_class = {}
    with open(config_file_path, "r") as f:
        data = json.load(f)
        default_class = data[parent_property]["default"]
        default_class = {int(k): v for k, v in default_class.items()}
    return default_class


def get_config_json(key: str = None) -> dict:
    """
    Get `config.json` content for the given key\n
    Returns a `dict` of `config.json`
    """
    is_key = key != None and key != ""
    with open(config_file_path, "r") as f:
        data = json.load(f)
        if is_key:
            try:
                return data[key]
            except KeyError:
                msg = f"No content found in {config_file_path} for key: {key}"
                raise KeyError(msg)
        else:
            return data
