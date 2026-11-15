"""Explicit connection smoke check; never runs during import or test discovery."""
import psycopg2
from utils import get_db_url

def main():
    try:
        conn = psycopg2.connect(get_db_url(),connect_timeout=10)
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT 1')
                assert cur.fetchone() == (1,)
            print('Database connection successful.')
        finally:
            conn.close()
    except (psycopg2.Error,ValueError):
        raise SystemExit('Database connection failed. Check configuration and connectivity.')

if __name__ == '__main__':
    main()
