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
# Login functionality (No changes needed here)
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

# --- MODIFIED run_query function to return error message ---
def run_query(sql):
    conn = get_db_connection()
    if conn is None:
        return None, "DB_CONNECTION_ERROR"
    
    try:
        df = pd.read_sql_query(sql, conn)
        return df, None # Success
    except Exception as e:
        # Return the error message string
        return None, str(e)

# ----------------------
# Gemini AI Client & SQL Generation
# ----------------------
@st.cache_resource
def get_openai_client():
    genai.configure(api_key=GOOGLE_API_KEY)
    # Using gemini-2.5-flash for responsiveness
    return genai.GenerativeModel("gemini-2.5-flash") 

def extract_sql_from_response(response_text):
    # Regex to extract content between ```sql ... ```
    match = re.search(r"```sql\s*(.*?)\s*```", response_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback to cleaning the whole response if code block markers are missed
    clean_sql = response_text.strip()
    return clean_sql

def generate_sql_with_gpt(user_question, failing_sql=None, error_message=None):
    model = get_openai_client()
    
    if failing_sql and error_message:
        # --- CORRECTION PROMPT ---
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

        **TASK:** Analyze the error message and the failing query. Generate a new, corrected PostgreSQL query that fixes the issue.
        
        {POSTGRES_DIALECT_INSTRUCTIONS}
        
        Generate ONLY the corrected SQL query:"""
    else:
        # --- INITIAL GENERATION PROMPT ---
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
# Core Execution Logic (MODIFIED)
# ----------------------
def execute_self_correcting_query(user_question):
    current_sql = None
    attempt = 1
    
    # Generate initial query
    st.info(f"Attempt {attempt}: Generating initial query...")
    current_sql = generate_sql_with_gpt(user_question)
    
    if not current_sql:
        return None, None
        
    while attempt <= MAX_ATTEMPTS:
        st.markdown(f"**Query Attempt {attempt}:**")
        st.code(current_sql, language="sql")
        
        # Run the query
        df, error_message = run_query(current_sql)
        
        if error_message is None:
            # SUCCESS
            st.success(f"✅ Success! Query executed on attempt {attempt}.")
            return df, current_sql
        
        # FAILURE
        st.error(f"❌ Execution failed on attempt {attempt}: {error_message}")
        
        if attempt == MAX_ATTEMPTS:
            st.warning(f"Maximum correction attempts ({MAX_ATTEMPTS}) reached. Displaying last failing query.")
            return None, current_sql

        # Prepare for next attempt (Self-Correction)
        attempt += 1
        with st.spinner(f"🧠 AI is self-correcting based on the error message... (Attempt {attempt} of {MAX_ATTEMPTS})"):
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

    return None, current_sql # Should be caught by the MAX_ATTEMPTS check, but here for safety


# ----------------------
# Streamlit App
# ----------------------
def main():
    require_login()
    st.title("🤖 AI-Powered SQL Query Assistant")
    st.markdown("Ask questions in natural language, and I will generate SQL queries for you to review and run!")
    st.markdown("---")
    
    # ... (Sidebar and Session State setup remains the same) ...

    st.sidebar.title("💡 Example Questions")
    st.sidebar.markdown("""
Try asking questions like:

**Demographics:**
- How many customers do we have by city?
- How many customers do we have by country?

**Products & Sales:**
- What are the top 10 products by quantity sold?
- What is the total sales by product category?
- How many orders per customer?

**Orders:**
- Average quantity ordered per order
- Total sales per month
- **Calculate the total amount for each order, rounded to 2 decimal places.**
""")
    st.sidebar.markdown("---")
    st.sidebar.info("""
    🩼**How it works:**
    1. Enter your question in plain English
    2. AI generates SQL query
    3. **If query fails, the AI automatically attempts to self-correct based on the database error (up to 3 times).**
    4. Review the final successful or failing query.
    """)

    if st.sidebar.button("🚪Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # ----------------------
    # Session state
    # ----------------------
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'generated_sql' not in st.session_state:
        st.session_state.generated_sql = None
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    
    # ----------------------
    # User input
    # ----------------------
    user_question = st.text_area(
        " What would you like to know?",
        height=100,
        placeholder="What is the total sales by product category?"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        generate_button = st.button(" Generate & Run SQL (Self-Correcting)", type="primary", use_container_width=True)
    with col2:
        if st.button(" Clear History", use_container_width=True):
            st.session_state.query_history = []
            st.session_state.generated_sql = None
            st.session_state.current_question = None

    # ----------------------
    # Run SQL (using new execution logic)
    # ----------------------
    if generate_button and user_question:
        user_question = user_question.strip()
        st.session_state.current_question = user_question
        st.session_state.generated_sql = None
        
        st.markdown("---")
        st.subheader("Execution Flow")
        
        df, final_sql = execute_self_correcting_query(user_question)
        st.session_state.generated_sql = final_sql # Store the final query (success or fail)

        if df is not None:
            # Success logic
            st.session_state.query_history.append(
                {'question': user_question, 'sql': final_sql, 'rows': len(df), 'success': True}
            )
            st.markdown("---")
            st.subheader("📊 Final Query Results")
            st.success(f"✅ Final query executed successfully, returned {len(df)} rows.")
            st.dataframe(df, width="stretch")
        else:
            # Failure logic
            st.session_state.query_history.append(
                {'question': user_question, 'sql': final_sql, 'rows': 0, 'success': False}
            )
            st.markdown("---")
            st.subheader("❌ Execution Failed")
            st.error("The AI failed to generate a successful query after all attempts. Review the last query above for manual correction.")


    # ----------------------
    # Show & Run SQL (for manual review/re-run)
    # ----------------------
    if st.session_state.generated_sql and not generate_button: # Only show this block if a query exists and we aren't mid-run
        st.markdown("---")
        st.subheader("Last Generated/Attempted SQL Query")
        st.info(f"**Question:** {st.session_state.current_question}")

        edited_sql = st.text_area(
            "Review and edit the SQL query if needed:",
            value=st.session_state.generated_sql,
            height=200
        )

        # Standard manual Run Query button
        if st.button("Manually Run Query", type="secondary", use_container_width=True):
             with st.spinner("Executing query ..."):
                df_manual, error_manual = run_query(edited_sql)
                if df_manual is not None:
                    st.markdown("---")
                    st.subheader("📊 Manual Query Results")
                    st.success(f"✅ Manual query returned {len(df_manual)} rows")
                    st.dataframe(df_manual, width="stretch")
                else:
                    st.error(f"❌ Manual execution failed: {error_manual}")
                    
    # ----------------------
    # Query history
    # ----------------------
    if st.session_state.query_history:
        st.markdown('---')
        st.subheader("📜 Query History")
        for idx, item in enumerate(reversed(st.session_state.query_history[-5:])):
            status_emoji = "✅" if item.get('success', False) else "❌"
            with st.expander(f"{status_emoji} Query {len(st.session_state.query_history)-idx}: {item['question'][:60]}..."):
                st.markdown(f"**Question:** {item['question']}")
                st.code(item["sql"], language="sql")
                if item.get('success', False):
                    st.caption(f"Returned {item['rows']} rows")
                else:
                    st.caption("Execution Failed (see run logs above)")
                    
                if st.button(f"Re-run this query", key=f"rerun_{idx}"):
                    df_rerun, error_rerun = run_query(item["sql"])
                    if df_rerun is not None:
                        st.dataframe(df_rerun, width="stretch")
                    else:
                        st.error(f"❌ Rerun failed: {error_rerun}")


if __name__ == "__main__":
    main()