import re
import streamlit as st
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import google.generativeai as genai
import bcrypt
from utils import get_db_url 

# Load environment variables
load_dotenv()

# Secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
HASHED_PASSWORD = st.secrets["HASHED_PASSWORD"].encode("utf-8")

# --- CONFIGURATION ---
MAX_ATTEMPTS = 3 # Maximum times Gemini will try to correct the query
POSTGRES_DIALECT_INSTRUCTIONS = """
Requirements for PostgreSQL Queries:
1. Generate ONLY the SQL query.
2. Use proper JOINs to get descriptive names.
3. Use SUM, AVG, COUNT when appropriate.
4. Use LIMIT 100 for queries returning many rows.
5. Format date columns correctly.
6. Add helpful column aliases using AS.
7. **For string concatenation (like names), use the PostgreSQL operator `||` (e.g., `FirstName || ' ' || LastName`).**
8. **CRITICAL: When using `ROUND()`, always explicitly cast the expression to NUMERIC using `::NUMERIC` to avoid type errors. Example: `ROUND((Price * Quantity)::NUMERIC, 2)`.**
"""

# Database schema for AI context
DATABASE_SCHEMA = f"""
Database Schema:

LOOKUP TABLES:
- ProductCategory(ProductCategoryID, ProductCategory, ProductCategoryDescription)

CORE TABLES:
- Region(RegionID, Region)
- Country(CountryID, Country, RegionID)
- Customer(CustomerID, FirstName, LastName, Address, City, CountryID)
- Product(ProductID, ProductName, ProductUnitPrice, ProductCategoryID)
- OrderDetail(OrderID, CustomerID, ProductID, OrderDate, QuantityOrdered)

IMPORTANT NOTES:
- OrderDate is DATE type
- QuantityOrdered is INTEGER

{POSTGRES_DIALECT_INSTRUCTIONS}
"""

# ----------------------
# Login functionality
# ----------------------
def login_screen():
    st.title("🔐 Secure Login")
    st.markdown("---")
    st.write("Enter your password to access the AI SQL Query Assistant.")
    
    password = st.text_input("Password", type="password", key="login_password")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        login_btn = st.button("🔓 Login", type="primary", use_container_width=True)
    
    if login_btn:
        if password:
            try:
                if bcrypt.checkpw(password.encode('utf-8'), HASHED_PASSWORD):
                    st.session_state.logged_in = True
                    st.success("✅ Authentication successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Incorrect password")
            except Exception as e:
                st.error(f"❌ Authentication error: {e}")
        else:
            st.warning("⚠️ Please enter a password")
    
    st.markdown("---")
    st.info("""
    **Security Notice:**
    - Passwords are protected using bcrypt hashing
    - Your session is secure and isolated
    - You will remain logged in until you close the browser or click logout
    """)

def require_login():
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        login_screen()
        st.stop()

# ----------------------
# Database connection
# ----------------------
@st.cache_resource
def get_db_connection():
    try:
        DATABASE_URL = get_db_url()
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return None

# ----------------------
# Run SQL query with error handling
# ----------------------
def run_query(sql):
    conn = get_db_connection()
    if conn is None:
        return None, "DB_CONNECTION_ERROR"
    
    try:
        df = pd.read_sql_query(sql, conn)
        return df, None
    except Exception as e:
        return None, str(e)

# ----------------------
# Gemini AI Client & SQL Generation
# ----------------------
@st.cache_resource
def get_openai_client():
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel("gemini-2.5-flash") 

def extract_sql_from_response(response_text):
    match = re.search(r"```sql\s*(.*?)\s*```", response_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()

def generate_sql_with_gpt(user_question, failing_sql=None, error_message=None):
    model = get_openai_client()
    
    if failing_sql and error_message:
        prompt = f"""You are a PostgreSQL expert assisting a developer. Your previous query failed.
        
        **DATABASE SCHEMA CONTEXT:**
        {DATABASE_SCHEMA}

        **ORIGINAL USER QUESTION:** {user_question}

        **FAILING QUERY:**
        ```sql
        {failing_sql}
        ```        
        **POSTGRESQL ERROR MESSAGE:**
        {error_message}

        **TASK:** Analyze the error message and generate a corrected PostgreSQL query.
        
        {POSTGRES_DIALECT_INSTRUCTIONS}
        
        Generate ONLY the corrected SQL query:"""
    else:
        prompt = f"""You are a PostgreSQL expert. Given the following database schema and a user's question, generate a valid PostgreSQL query.

        {DATABASE_SCHEMA}

        User Question: {user_question}
        
        Generate ONLY the SQL query:"""

    try:
        response = model.generate_content(prompt)
        sql_query = extract_sql_from_response(response.text)
        return sql_query
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return None

# ----------------------
# Self-correcting query execution
# ----------------------
def execute_self_correcting_query(user_question):
    current_sql = None
    attempt = 1
    
    st.info(f"Attempt {attempt}: Generating initial query...")
    current_sql = generate_sql_with_gpt(user_question)
    
    if not current_sql:
        return None, None
        
    while attempt <= MAX_ATTEMPTS:
        st.markdown(f"**Query Attempt {attempt}:**")
        st.code(current_sql, language="sql")
        
        df, error_message = run_query(current_sql)
        
        if error_message is None:
            st.success(f"✅ Success! Query executed on attempt {attempt}.")
            return df, current_sql
        
        st.error(f"❌ Execution failed on attempt {attempt}: {error_message}")
        
        if attempt == MAX_ATTEMPTS:
            st.warning(f"Maximum correction attempts ({MAX_ATTEMPTS}) reached. Displaying last failing query.")
            return None, current_sql

        attempt += 1
        with st.spinner(f"🧠 AI is self-correcting... (Attempt {attempt}/{MAX_ATTEMPTS})"):
            new_sql = generate_sql_with_gpt(
                user_question=user_question,
                failing_sql=current_sql,
                error_message=error_message
            )
            
            if new_sql and new_sql != current_sql:
                current_sql = new_sql
            else:
                st.warning("AI failed to generate a new, distinct query. Stopping corrections.")
                return None, current_sql

    return None, current_sql

# ----------------------
# Streamlit UI
# ----------------------
def main():
    require_login()

    # --- Initialize session state ---
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'generated_sql' not in st.session_state:
        st.session_state.generated_sql = None
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None

    # --- Page Layout ---
    st.set_page_config(
        page_title="AI SQL Query Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("<h1 style='text-align:center; color:#4B8BBE;'>🤖 AI-Powered SQL Query Assistant</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#6c757d;'>Ask questions in natural language and get PostgreSQL queries instantly. AI attempts to self-correct queries if they fail.</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # --- Sidebar ---
    st.sidebar.title("💡 Examples & Tips")
    st.sidebar.markdown("""
**Try asking questions like:**

**Demographics:**
- How many customers do we have by city?
- How many customers do we have by country?

**Products & Sales:**
- Top 10 products by quantity sold
- Total sales by product category
- Orders per customer

**Orders:**
- Average quantity ordered per order
- Total sales per month
- Total amount for each order, rounded to 2 decimal places
""")
    st.sidebar.markdown("---")
    st.sidebar.info(
        "🩼 **How it works:**\n"
        "1. Enter your question in plain English.\n"
        "2. AI generates SQL query.\n"
        "3. If the query fails, AI will attempt up to 3 self-corrections.\n"
        "4. Review the final query and results."
    )
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # --- User Input ---
    st.subheader("🔎 Ask Your Question")
    user_question = st.text_area(
        label="Your Question",
        placeholder="e.g., What is the total sales by product category?",
        height=100
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        generate_button = st.button("🟢 Generate & Run SQL", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.query_history = []
            st.session_state.generated_sql = None
            st.session_state.current_question = None

    # --- Execution Flow ---
    if generate_button and user_question:
        st.markdown("---")
        st.subheader("⚡ Execution Flow")
        df, final_sql = execute_self_correcting_query(user_question)
        st.session_state.generated_sql = final_sql
        st.session_state.current_question = user_question

        # --- Save query to history ---
        if final_sql:
            st.session_state.query_history.append({
                'question': user_question,
                'sql': final_sql,
                'rows': len(df) if df is not None else 0,
                'success': df is not None
            })

        if df is not None:
            st.success(f"✅ Query executed successfully, returned {len(df)} rows.")
            st.dataframe(df, use_container_width=True)
        else:
            st.error("❌ The AI failed to generate a successful query. Check the last generated query above.")

    # --- Manual SQL Editing ---
    if st.session_state.generated_sql and not generate_button:
        st.markdown("---")
        st.subheader("✏️ Review & Edit Last Generated SQL")
        st.info(f"**Question:** {st.session_state.current_question}")
        edited_sql = st.text_area(
            "Edit the SQL query if needed:",
            value=st.session_state.generated_sql,
            height=200
        )
        if st.button("Run Edited Query", type="secondary", use_container_width=True):
            with st.spinner("Executing query..."):
                df_manual, error_manual = run_query(edited_sql)
                if df_manual is not None:
                    st.success(f"✅ Query returned {len(df_manual)} rows")
                    st.dataframe(df_manual, use_container_width=True)
                else:
                    st.error(f"❌ Execution failed: {error_manual}")

    # --- Query History ---
    if st.session_state.query_history:
        st.markdown("---")
        st.subheader("📜 Query History")
        for idx, item in enumerate(reversed(st.session_state.query_history)):
            status_emoji = "✅" if item.get('success', False) else "❌"
            with st.expander(f"{status_emoji} Query {len(st.session_state.query_history)-idx}: {item['question'][:60]}..."):
                st.markdown(f"**Question:** {item['question']}")
                st.code(item["sql"], language="sql")
                st.caption(f"Rows returned: {item['rows']}" if item.get('success', False) else "Execution failed")
                if st.button(f"Re-run this query", key=f"rerun_{idx}"):
                    df_rerun, error_rerun = run_query(item["sql"])
                    if df_rerun is not None:
                        st.dataframe(df_rerun, use_container_width=True)
                    else:
                        st.error(f"❌ Rerun failed: {error_rerun}")

if __name__ == "__main__":
    main()
