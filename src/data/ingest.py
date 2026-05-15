import gcp_utils as gcp
import pandas as pd

def load_data(creds):
    """
    Ejecuta una query en BigQuery y devuelve los resultados como un DataFrame de pandas.

    Args:
        creds (str): La ruta al archivo de credenciales.
    Returns:
        pd.DataFrame: Los resultados de la query.
    """

    query = """
        select 
            *
        from dwh.lkp_clientes_vig_snp
    """
    return gcp.query_bigquery(query, creds, date_param=fec_proceso)

def load_dataset_from_config(cfg):
    source_type = cfg["data_source"]["type"]
    path = cfg["data_source"]["path"]

    if source_type == "py":
        return load_data(creds=cfg['credentials'])

    elif source_type == "sql":
        return gcp.file_bigquery(
            file_path=path,
            creds=cfg['credentials']
        )

    elif source_type == "csv":
        return pd.read_csv(path)

    else:
        raise ValueError(f"Tipo desconocido: {source_type}")