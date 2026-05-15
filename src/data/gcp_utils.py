from google.cloud import storage
from google.cloud import bigquery
from datetime import datetime, timedelta
import os
import json


def file_bigquery(file_path, creds, date_param=None):
    """
    Ejecuta una query en BigQuery y devuelve los resultados como un DataFrame de pandas.

    Args:
        file_path (str): La ruta al archivo SQL.
        creds (str): La ruta al archivo de credenciales.
        date_param (str, optional): El parámetro de fecha para reemplazar en la consulta.
    Returns:
        pd.DataFrame: Los resultados de la query.
    """
    # Inicializar el cliente con las credenciales
    client = bigquery.Client.from_service_account_json(creds)

    with open(file_path, 'r') as archivo:
        query = archivo.read()

    if date_param is not None:
        query = query.replace("{{date}}", date_param)

    result = client.query(query).result().to_dataframe(create_bqstorage_client=True)

    return result


def query_bigquery(query, creds, date_param=None):
    """
    Ejecuta una query en BigQuery y devuelve los resultados como un DataFrame de pandas.

    Args:
        query (str): La consulta SQL a ejecutar.
        creds (str): La ruta al archivo de credenciales.
    Returns:
        pd.DataFrame: Los resultados de la query.
    """
    # Inicializar el cliente con las credenciales
    client = bigquery.Client.from_service_account_json(creds)
    
    if date_param is not None:
        query = query.replace("{{date}}", date_param)

    result = client.query(query).result().to_dataframe(create_bqstorage_client=True)

    return result


def read_config():
    """
    This fucntion read the config_file.json file located in respective bucket.
    
    Parameters
        env (str): The name of the environment. It can be dev or prd.

    Returns
        cfg (dict): The config file dictionary
    """

    env = os.getenv('ENV')

    credentials = "deployment/gcp_key.json"
    project_id = os.getenv('GCP_PROJECT_ID')
    config_folder = 'extra_files/'
    file = 'config_file.json'

    print("Enviroment:", env)

    if(env == 'dev'):
        config_bucket = 'pipeline-dev'
    elif (env == 'prd'):
        config_bucket = 'pipeline-prd'  
    else:
        raise Exception("The selected environment is not valid")

    try:
        storage_client = storage.Client.from_service_account_json(credentials)
        bucket_client = storage_client.get_bucket(config_bucket)
        blob = bucket_client.blob(config_folder + file)
        cfg = json.loads(blob.download_as_string(client=None))
        cfg['credentials'] = credentials
        cfg['project_id'] = project_id
        cfg['env'] = env
        
        return cfg

    except IOError: 
        raise Exception("Cannot read config_file or it may not exist")