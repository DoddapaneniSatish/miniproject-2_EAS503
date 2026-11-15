"""Conservative PostgreSQL SELECT validation plus bounded read-only execution."""
import pandas as pd
import psycopg2
from psycopg2 import sql
from pglast import parse_sql

TABLES = {'region','country','customer','productcategory','product','orderdetail','etl_loads'}
FUNCTIONS = {'count','sum','avg','min','max','round','abs','ceil','ceiling','floor',
             'date_trunc','date_part','extract','to_char','lower','upper','trim','btrim',
             'ltrim','rtrim','length','concat','concat_ws','replace','substring',
             'row_number','rank','dense_rank','lag','lead','ntile','first_value','last_value',
             'percent_rank','cume_dist','now','age','make_date','greatest','least','coalesce','nullif'}
TYPES = {'numeric','decimal','int2','int4','int8','float4','float8','text','varchar',
         'bpchar','bool','date','timestamp','timestamptz','interval'}
BLOCKED = {'InsertStmt','UpdateStmt','DeleteStmt','MergeStmt','CopyStmt',
           'CallStmt','RangeFunction','RangeTableFunc','ParamRef','CollateClause'}

class QueryRejected(ValueError):
    pass


def validate_query(query, schema='public'):
    if not query.strip() or len(query) > 20000:
        raise QueryRejected('Enter one SELECT query of at most 20,000 characters.')
    try:
        statements = parse_sql(query)
    except Exception:
        raise QueryRejected('The query is not valid PostgreSQL SQL.') from None
    if len(statements) != 1 or type(statements[0].stmt).__name__ != 'SelectStmt':
        raise QueryRejected('Only a single SELECT query is allowed.')
    tree = statements[0].stmt()
    def walk(node):
        if isinstance(node,dict):
            yield node
            for value in node.values():
                yield from walk(value)
        elif isinstance(node,(tuple,list)):
            for value in node:
                yield from walk(value)
    nodes = list(walk(tree))
    ctes = {n['ctename'] for n in nodes if n.get('@') == 'CommonTableExpr'}
    for node in nodes:
        tag = node.get('@')
        if tag in BLOCKED:
            raise QueryRejected('This query contains an unsupported or non-read-only operation.')
        if tag == 'SelectStmt' and (node.get('intoClause') or node.get('lockingClause')):
            raise QueryRejected('SELECT INTO and row locking are not allowed.')
        if tag == 'RangeVar':
            if node.get('catalogname') or node.get('schemaname') not in (None,schema):
                raise QueryRejected('Queries may access only the configured sales schema.')
            if node['relname'] not in TABLES and not (node['relname'] in ctes and not node.get('schemaname')):
                raise QueryRejected('Queries may access only the documented sales tables.')
        if tag in ('FuncCall','TypeName'):
            names = [n['sval'] for n in node['funcname' if tag == 'FuncCall' else 'names']]
            allowed = FUNCTIONS if tag == 'FuncCall' else TYPES
            if names[-1] not in allowed or (len(names) > 1 and names[:-1] != ['pg_catalog']):
                raise QueryRejected('This function or type is not supported by the sales query tool.')
    return query.strip().rstrip(';').strip()


def run_query(query, database_url, schema='public', max_rows=500):
    query = validate_query(query,schema)
    conn = psycopg2.connect(database_url,connect_timeout=10)
    try:
        conn.set_session(readonly=True,autocommit=False)
        with conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = '5s'")
                cur.execute("SET LOCAL lock_timeout = '1s'")
                cur.execute(sql.SQL('SET LOCAL search_path TO pg_catalog, {}').format(sql.Identifier(schema)))
                cur.execute('SELECT rolsuper OR rolcreatedb OR rolcreaterole FROM pg_roles WHERE rolname=current_user')
                if cur.fetchone()[0]:
                    raise QueryRejected('Configure a restricted SELECT-only database login for the app.')
                cur.execute('SELECT tablename FROM pg_tables WHERE schemaname=%s AND tablename=ANY(%s)',(schema,list(TABLES)))
                tables = [r[0] for r in cur.fetchall()]
                if not tables:
                    raise QueryRejected('No sales tables found in the configured schema.')
                for table in tables:
                    cur.execute('SELECT has_table_privilege(current_user,%s,%s)',
                                (sql.Identifier(schema,table).as_string(conn),'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'))
                    if cur.fetchone()[0]:
                        raise QueryRejected('Configure a restricted SELECT-only database login for the app.')
                # Parse/serialize before embedding: removes a trailing SQL comment safely.
                from pglast.stream import RawStream
                normalized = RawStream()(parse_sql(query))
                cur.execute('SELECT * FROM (' + normalized + ') AS sales_result LIMIT %s',(max_rows+1,))
                rows = cur.fetchmany(max_rows+1)
                df = pd.DataFrame(rows[:max_rows],columns=[c.name for c in cur.description])
                df.attrs['truncated'] = len(rows) > max_rows
                return df
    finally:
        conn.close()
