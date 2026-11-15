"""Read configuration without printing secrets or connecting at import time."""
import os
from dotenv import load_dotenv
from psycopg2.extensions import make_dsn

load_dotenv()

def get_db_url(config=None):
    config = os.environ if config is None else config
    url = config.get('DATABASE_URL')
    if url:
        return url
    names = ('POSTGRES_USERNAME','POSTGRES_PASSWORD','POSTGRES_SERVER','POSTGRES_DATABASE')
    if not all(config.get(name) for name in names):
        raise ValueError('Configure DATABASE_URL or all four POSTGRES_* settings.')
    return make_dsn(user=config[names[0]],password=config[names[1]],
                    host=config[names[2]],dbname=config[names[3]],
                    sslmode=config.get('PGSSLMODE','require'))
