import csv
import os
from pathlib import Path
from decimal import Decimal
from uuid import uuid4
import bcrypt
import psycopg2
from psycopg2 import sql
import pytest
from populate_db import parse_source, load_source
from query_service import validate_query, run_query, QueryRejected
from utils import get_db_url

HEADER = ['Name','Address','City','Country','Region','ProductName','ProductCategory',
          'ProductCategoryDescription','ProductUnitPrice','QuantityOrderded','OrderDate']
BASE = ['Alex Lee','1 Main','Buffalo','USA','North America','Widget','Tools','Hardware','2.50','2','20250101']

def source_file(tmp_path, rows=None):
    path = tmp_path/'source.tsv'
    with path.open('w',newline='') as handle:
        writer = csv.writer(handle,delimiter='\t')
        writer.writerow(HEADER); writer.writerows(rows or [BASE])
    return path

@pytest.mark.parametrize('date',['20250101','2025-01-01','01/01/2025'])
def test_supported_dates(tmp_path,date):
    row = BASE.copy(); row[-1] = date
    source = parse_source(source_file(tmp_path,[row]))
    assert not source.rejections
    assert source.orders[0][3].isoformat() == '2025-01-01'
    assert source.orders[0][1][2] == Decimal('2.50')

@pytest.mark.parametrize('field,value,reason',[
    (5,'Widget;Gadget','mismatched_item_counts'),(9,'-2','invalid_quantity'),
    (9,'0','invalid_quantity'),(9,'1.5','invalid_quantity'),
    (8,'NaN','invalid_price'),(8,'-1','invalid_price'),(8,'0.00001','invalid_price'),
    (10,'2025-02-30','invalid_date'),(0,'','missing_customer_or_geography')])
def test_rejections_are_accounted_for(tmp_path,field,value,reason):
    row = BASE.copy(); row[field] = value
    source = parse_source(source_file(tmp_path,[BASE,row]))
    assert source.source_rows == 2
    assert len(source.orders) == 1
    assert source.rejections == [{'source_row':3,'reason':reason}]
    assert 'Alex' not in str(source.report())


def test_shared_names_retain_identity_and_product_price(tmp_path):
    other = BASE.copy(); other[1] = '2 Main'; other[8] = '3.00'
    source = parse_source(source_file(tmp_path,[BASE,other]))
    assert len({r[0] for r in source.orders}) == 2
    assert len({r[1] for r in source.orders}) == 2


def test_rejected_batch_never_opens_database(tmp_path,monkeypatch):
    row = BASE.copy();row[9]='-1'
    source = parse_source(source_file(tmp_path,[BASE,row]))
    monkeypatch.setattr(psycopg2,'connect',lambda *a,**k:pytest.fail('Opened database for rejected source'))
    with pytest.raises(ValueError,match='no database changes'):
        load_source(source,'not-a-dsn')

@pytest.mark.parametrize('query',[
    'SELECT COUNT(*) FROM orderdetail',
    'WITH monthly AS (SELECT orderdate, SUM(quantityordered) AS units FROM orderdetail GROUP BY orderdate) SELECT * FROM monthly',
    "SELECT date_trunc('month',orderdate),round(SUM(quantityordered*productunitprice)::numeric,2) FROM orderdetail JOIN product USING(productid) GROUP BY 1",
    "SELECT '; DROP TABLE customer' AS example",
    'SELECT 1; -- trailing comment',
])
def test_allowed_queries(query):
    assert validate_query(query)

@pytest.mark.parametrize('query',[
    'DELETE FROM customer','SELECT 1; DROP TABLE customer',
    'WITH x AS (DELETE FROM customer RETURNING *) SELECT * FROM x',
    'SELECT * INTO backup FROM customer','SELECT * FROM customer FOR UPDATE',
    "SELECT pg_sleep(30)","SELECT set_config('search_path','public',false)",
    'SELECT * FROM pg_catalog.pg_authid','SELECT * FROM information_schema.tables',
    'SELECT * FROM another_schema.customer','SELECT * FROM secret_table',
    "SELECT 'x'::custom_type",'SELECT * FROM generate_series(1,1000000)',
    'COPY customer TO STDOUT','SELECT public.count(*) FROM customer',
])
def test_unsafe_queries_rejected_before_connect(query,monkeypatch):
    monkeypatch.setattr(psycopg2,'connect',lambda *a,**k:pytest.fail('Opened connection for rejected query'))
    with pytest.raises(QueryRejected):
        run_query(query,'not-a-dsn')


def test_quoted_connection_password():
    dsn = get_db_url({'POSTGRES_USERNAME':'reader','POSTGRES_PASSWORD':"a b'@/c",'POSTGRES_SERVER':'localhost','POSTGRES_DATABASE':'sales'})
    assert psycopg2.extensions.parse_dsn(dsn)['password'] == "a b'@/c"


def test_ui_without_secrets_starts_at_configuration_notice(monkeypatch):
    from streamlit.testing.v1 import AppTest
    for key in ('HASHED_PASSWORD','DATABASE_URL','GOOGLE_API_KEY'):
        monkeypatch.delenv(key,raising=False)
    app = AppTest.from_file('streamlit_app.py').run()
    assert not app.exception
    assert any('HASHED_PASSWORD' in notice.value for notice in app.info)


def test_ui_manual_sql_without_ai(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('HASHED_PASSWORD',bcrypt.hashpw(b'test-login',bcrypt.gensalt()).decode())
    app = AppTest.from_file('streamlit_app.py').run()
    app.text_input[0].input('test-login')
    app.button[0].click().run()
    assert not app.exception
    assert any(b.label == 'Run reviewed query' for b in app.button)
    assert any('AI generation is not configured' in n.value for n in app.info)

@pytest.fixture
def database():
    url = os.getenv('TEST_DATABASE_URL')
    if not url:
        pytest.skip('TEST_DATABASE_URL not set; PostgreSQL integration runs in CI')
    schema = 'test_sales_'+uuid4().hex[:12]
    yield url,schema
    conn = psycopg2.connect(url)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(sql.Identifier(schema)))
    finally:
        conn.close()


def test_postgres_load_is_repeatable_and_changed_snapshot_refused(tmp_path,database):
    url,schema=database
    other=BASE.copy();other[1]='2 Main';other[8]='3.00'
    source=parse_source(source_file(tmp_path,[BASE,other]))
    assert load_source(source,url,schema)['inserted_line_items']==2
    assert load_source(source,url,schema)['status']=='unchanged'
    changed=BASE.copy();changed[9]='3'
    with pytest.raises(ValueError,match='different snapshot'):
        load_source(parse_source(source_file(tmp_path,[changed])),url,schema)
    conn=psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL('SELECT COUNT(*),COUNT(DISTINCT customerid),COUNT(DISTINCT productid) FROM {}.orderdetail').format(sql.Identifier(schema)))
            assert cur.fetchone()==(2,2,2)
    finally:conn.close()


def test_postgres_failed_load_rolls_back(tmp_path,database):
    url,schema=database
    source=parse_source(source_file(tmp_path))
    load_source(source,url,schema)
    broken=BASE.copy();broken[7]='Conflicting description'
    with pytest.raises(ValueError,match='descriptions conflict'):
        load_source(parse_source(source_file(tmp_path,[broken])),url,schema,dataset='new-batch')
    conn=psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL('SELECT COUNT(*) FROM {}.etl_loads').format(sql.Identifier(schema)))
            assert cur.fetchone()==(1,)
    finally:conn.close()


def test_postgres_unmanaged_schema_is_preserved(tmp_path,database):
    url,schema=database
    conn=psycopg2.connect(url)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL('CREATE SCHEMA {}; CREATE TABLE {}.region (existing_data TEXT)').format(sql.Identifier(schema),sql.Identifier(schema)))
                cur.execute(sql.SQL("INSERT INTO {}.region VALUES ('keep')").format(sql.Identifier(schema)))
        with pytest.raises(ValueError,match='unmanaged'):
            load_source(parse_source(source_file(tmp_path)),url,schema)
        with conn.cursor() as cur:
            cur.execute(sql.SQL('SELECT * FROM {}.region').format(sql.Identifier(schema)))
            assert cur.fetchone()==('keep',)
    finally:conn.close()


def test_postgres_reader_results_limits_and_recovery(tmp_path,database):
    url,schema=database
    load_source(parse_source(source_file(tmp_path,[BASE,BASE])),url,schema)
    reader='reader_'+uuid4().hex[:12]
    admin=psycopg2.connect(url)
    try:
        with admin:
            with admin.cursor() as cur:
                cur.execute(sql.SQL('CREATE ROLE {} LOGIN PASSWORD %s').format(sql.Identifier(reader)),('test-only',))
                cur.execute(sql.SQL('GRANT USAGE ON SCHEMA {} TO {}; GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}').format(sql.Identifier(schema),sql.Identifier(reader),sql.Identifier(schema),sql.Identifier(reader)))
        config=psycopg2.extensions.parse_dsn(url);config.update(user=reader,password='test-only')
        reader_url=psycopg2.extensions.make_dsn(**config)
        df=run_query('SELECT * FROM orderdetail',reader_url,schema,max_rows=1)
        assert len(df)==1 and df.attrs['truncated']
        with pytest.raises(psycopg2.Error):
            run_query('SELECT 1/0 FROM orderdetail',reader_url,schema)
        assert len(run_query('SELECT COUNT(*) FROM orderdetail; -- comment',reader_url,schema))==1
        with pytest.raises(QueryRejected,match='restricted'):
            run_query('SELECT * FROM orderdetail',url,schema)
        conn=psycopg2.connect(reader_url)
        try:
            with pytest.raises(psycopg2.Error):
                with conn.cursor() as cur:
                    cur.execute(sql.SQL('DELETE FROM {}.orderdetail').format(sql.Identifier(schema)))
        finally:conn.close()
    finally:
        with admin:
            with admin.cursor() as cur:
                cur.execute(sql.SQL('DROP OWNED BY {}; DROP ROLE {}').format(sql.Identifier(reader),sql.Identifier(reader)))
        admin.close()
