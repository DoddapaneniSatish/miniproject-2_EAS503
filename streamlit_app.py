"""Password-gated sales SQL assistant with reviewed, bounded SELECT queries."""
import os
import re
import time
import bcrypt
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from query_service import QueryRejected, run_query
from utils import get_db_url

load_dotenv()
SCHEMA_CONTEXT = '''Sales tables (PostgreSQL, unquoted lowercase identifiers):
region(regionid, region)
country(countryid, country, regionid)
customer(customerid, firstname, lastname, address, city, countryid)
productcategory(productcategoryid, productcategory, productcategorydescription)
product(productid, productname, productunitprice, productcategoryid)
orderdetail(orderid, customerid, productid, orderdate, quantityordered)
Each orderdetail record is a LINE ITEM, not necessarily a distinct business order.
Revenue is SUM(productunitprice * quantityordered). No profit or margin fields exist.
Return one SELECT query, optionally with SELECT-only CTEs. Never modify data.
Use joins via foreign keys, LIMIT 100, and NUMERIC casts for ROUND expressions.
Use unqualified table names. Do not reference other tables, schemas or functions
with side effects. Output SQL only; do not include explanations.'''


def settings():
    keys = ('DATABASE_URL','POSTGRES_USERNAME','POSTGRES_PASSWORD','POSTGRES_SERVER',
            'POSTGRES_DATABASE','PGSSLMODE','APP_DB_SCHEMA','GOOGLE_API_KEY',
            'GEMINI_MODEL','HASHED_PASSWORD')
    config = {k:os.environ[k] for k in keys if k in os.environ}
    try:
        config.update({k:st.secrets[k] for k in keys if k in st.secrets})
    except FileNotFoundError:
        pass
    return config


def extract_sql(text):
    match = re.fullmatch(r'\s*```(?:sql)?\s*([\s\S]*?)```\s*',text,re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def login(config):
    if st.session_state.get('logged_in'):
        return
    st.title('Sales SQL Assistant')
    hashed = config.get('HASHED_PASSWORD')
    if not hashed:
        st.info('The site owner needs to configure HASHED_PASSWORD before enabling access.')
        st.stop()
    with st.form('login'):
        password = st.text_input('Password',type='password',max_chars=72)
        submitted = st.form_submit_button('Sign in')
    if submitted:
        attempts = [t for t in st.session_state.get('login_attempts',[]) if time.monotonic()-t < 60]
        if len(attempts) >= 5:
            st.warning('Please wait a minute before trying again.')
        else:
            st.session_state.login_attempts = attempts + [time.monotonic()]
            try:
                valid = bcrypt.checkpw(password.encode(),hashed.encode())
            except (ValueError,TypeError):
                st.error('Login is not configured correctly. Contact the site owner.')
            else:
                if valid:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error('Incorrect password.')
    st.stop()


def generate_sql(question,config):
    from google import genai
    client = genai.Client(api_key=config['GOOGLE_API_KEY'],http_options={'timeout':20000})
    try:
        response = client.models.generate_content(model=config.get('GEMINI_MODEL','gemini-2.5-flash'),
                    contents=SCHEMA_CONTEXT+'\nUser question: '+question)
        return extract_sql(response.text or '')
    finally:
        client.close()


def main():
    st.set_page_config(page_title='Sales SQL Assistant',page_icon='📊',layout='wide')
    config = settings()
    login(config)
    st.title('Sales SQL Assistant')
    st.caption('Explore sales data with reviewed, read-only PostgreSQL queries.')
    if st.sidebar.button('Sign out'):
        st.session_state.clear()
        st.rerun()
    st.sidebar.markdown('### Business questions\n- Monthly revenue and growth\n- Top product categories by revenue\n- Customer spending by country\n- Top products by units sold')
    st.sidebar.caption('Results are limited to 500 rows. Queries time out after 5 seconds.')
    st.session_state.setdefault('query_history',[])
    st.session_state.setdefault('sql_editor','SELECT COUNT(*) AS line_items FROM orderdetail;')
    question = st.text_area('Ask a sales question',max_chars=2000,
                            placeholder='Which product categories generated the most revenue?')
    if not config.get('GOOGLE_API_KEY'):
        st.info('AI generation is not configured. You can still write and run reviewed SQL below.')
    if st.button('Generate SQL',disabled=not bool(question.strip() and config.get('GOOGLE_API_KEY'))):
        try:
            with st.spinner('Preparing SQL for your review…'):
                generated = generate_sql(question,config)
            from query_service import validate_query
            validate_query(generated,config.get('APP_DB_SCHEMA','public'))
            st.session_state.sql_editor = generated
            st.session_state.pop('result',None)
        except QueryRejected as exc:
            st.warning(str(exc))
        except Exception:
            st.error('AI generation is unavailable. Try again later or enter SQL manually.')
    st.subheader('Review and run SQL')
    st.caption('Generating SQL does not execute it. Review the query, then choose Run reviewed query.')
    query = st.text_area('SQL query',key='sql_editor',height=220,max_chars=20000)
    if st.button('Run reviewed query',type='primary'):
        st.session_state.pop('result',None)
        try:
            with st.spinner('Running read-only query…'):
                df = run_query(query,get_db_url(config),config.get('APP_DB_SCHEMA','public'))
            st.session_state.result = df
            st.session_state.query_history = ([{'sql':query,'rows':len(df)}]+st.session_state.query_history)[:20]
        except (QueryRejected,ValueError) as exc:
            st.error(str(exc))
        except psycopg2.Error:
            st.error('The query could not complete. Check the query, database availability, and SELECT-only access. Long queries stop after 5 seconds.')
    if 'result' in st.session_state:
        df = st.session_state.result
        st.success(f'{len(df):,} rows returned.')
        if df.attrs.get('truncated'):
            st.warning('Showing the first 500 rows. Add filters or aggregation for a complete analysis.')
        st.dataframe(df,use_container_width=True)
    with st.expander('Database schema and metric definitions'):
        st.code(SCHEMA_CONTEXT)
    with st.expander('Session query history'):
        if st.button('Clear history'):
            st.session_state.query_history = []
        for item in st.session_state.query_history:
            st.code(item['sql'],language='sql')
            st.caption(f"{item['rows']} rows returned")

if __name__ == '__main__':
    main()
