from ingest import load_dataset_from_config
import pandas as pd

def prepare_dataset(cfg):
    df = load_dataset_from_config(cfg)

    dataset_cfg = cfg["dataset"]

    # asignar columnas si vienen sin header (ej: german.data)
    if df.columns.size != len(dataset_cfg["columns"]):
        df.columns = dataset_cfg["columns"]

    # mapping del target
    target_col = dataset_cfg["target"]
    mapping = dataset_cfg["target_mapping"]

    df[target_col] = df[target_col].astype(str).map(mapping)

    return df