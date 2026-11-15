# E-commerce Sales Analytics & SQL Assistant

A Python ETL pipeline converts a denormalized, tab-separated source into six related PostgreSQL sales tables. A password-gated Streamlit app supports manual SQL and optional Gemini-generated SQL, reviewed before execution.

## What this demonstrates

- Source validation, exact decimal prices, normalized data modeling and foreign keys.
- Atomic loading, snapshot accounting, repeatable reruns and explicit rejected-record reports.
- PostgreSQL joins, aggregates, CTEs and window functions through a restricted query interface.
- Gemini integration, bounded queries, automated PostgreSQL integration tests and Streamlit UI smoke tests.

The bundled `data.csv` is **tab-separated** despite its filename. Validation currently accounts for 91 source records containing 621,806 line items, with zero rejected records. These are dataset counts, not business performance claims. Source provenance and licensing should be established before redistributing it elsewhere.

## Local setup

Use Python 3.12. From the repository root:

```sh
pip install -r requirements.txt
cp .env.example .env
python generate_password.py
```

Edit `.env` locally. Put the bcrypt hash in `HASHED_PASSWORD`. Never commit real credentials. The app accepts settings from environment variables or private Streamlit secrets. Gemini is optional: `GOOGLE_API_KEY` enables generation; `GEMINI_MODEL` defaults to `gemini-2.5-flash` and can be changed to a model available to your account. The app uses the current `google-genai` SDK.

## Validate before loading

```sh
python populate_db.py --validate-only
```

Validation opens **no database connection**. It creates `etl_report.json`, recording source hash, accepted/rejected source records, line-item counts and rejection reasons. Reports contain source row numbers, not customer fields. Supported dates: `YYYYMMDD`, `YYYY-MM-DD`, and explicitly US `MM/DD/YYYY`.

A malformed item rejects its entire source record. List lengths must match; prices must be finite, nonnegative and fit four decimal places; quantities must be positive integers. Negative quantities are rejected because this dataset has no documented returns policy. Missing customer fields and conflicting geography/category descriptions are reported. The default load refuses any rejected records. `--allow-rejects` is an explicit opt-in after reviewing the report.

## Load an isolated schema

Use a **loader/owner** `DATABASE_URL` in your terminal, separate from the app login:

```sh
python populate_db.py --schema sales_analytics --dataset orders
```

The default schema is `sales_analytics` (or `APP_DB_SCHEMA` when set). The loader never drops existing tables. It refuses unmanaged legacy sales tables rather than migrating them implicitly. Keep the old `public` tables until you have verified the new schema and switched the app deliberately.

All schema creation, dimension inserts, facts and the `etl_loads` ledger commit in one transaction. A failed load rolls back. Repeating an identical source under the same dataset name inserts nothing. A changed snapshot with that name is refused. This version is an immutable snapshot loader, **not incremental change-data capture**. Do not assign a new dataset name to a revised or overlapping copy: that would count the same sales twice. New dataset names are for independently verified, non-overlapping batches only.

Customer identity includes full name, address, city and country rather than name alone. Products are distinguished by name, category and unit price. Facts retain source dataset/record/item positions; identical line items are preserved because the source supplies no independent order identifier.

## Configure SELECT-only app access

After loading, create a dedicated `sales_reader` LOGIN in your database provider with a generated password and no owner/admin/writer memberships. Run `setup_reader.sql` as the owner to grant access to `sales_analytics`. Configure the app's private settings with:

- `DATABASE_URL`: the **reader** connection, including the provider's required TLS settings.
- `APP_DB_SCHEMA`: `sales_analytics`.
- `HASHED_PASSWORD`: the bcrypt login hash.
- Optional `GOOGLE_API_KEY` and `GEMINI_MODEL`.

```sh
streamlit run streamlit_app.py
```

For legacy deployments, the app's default schema remains `public`; set it explicitly when switching to the new data. The app rejects privileged logins and logins with write privileges on the known sales tables. Database role configuration must be completed before enabling the updated app against existing owner credentials.

Queries must parse as a single PostgreSQL SELECT, including SELECT-only CTEs. Other relations/schemas, modifying CTEs, SELECT INTO, row locks, unsupported functions and custom casts are rejected. Approved queries run in a fresh read-only transaction with a 5-second statement timeout, 1-second lock timeout and 500-row result cap. Connection failures and raw provider/database errors are not displayed or sent to Gemini. Each query closes its connection, so a failed transaction cannot poison later requests or share transaction state across visitors.

This is defense in depth, not a general-purpose sandbox for arbitrary SQL. Restrict the reader to these sales tables and avoid custom functions/operators in the app schema. The password gate is a small demonstration login; use a managed authentication provider for a broader production application.

## Metric definitions and limitations

- Revenue: `SUM(productunitprice * quantityordered)` using the price attached to the source product record.
- `COUNT(*)` over `orderdetail` counts **line items**, not independently identified business orders.
- The source has no verified profit, margin, acquisition or returns fields. Do not infer them.
- A dedicated business dashboard and independent AI SQL evaluation are future work; neither is claimed as implemented.

## Verification

```sh
pip install -r requirements-dev.txt
python -m pytest -q
```

Unit tests require no keys or database. PostgreSQL integration tests run when `TEST_DATABASE_URL` points to a **disposable test database** whose login can create test schemas and roles. They create uniquely named schemas/readers and clean them up. Never set this variable to a production database. GitHub Actions provisions PostgreSQL 17, verifies repeatability, rollback, legacy-table preservation, reader permissions, query recovery, result limits, UI startup/login and full-source validation.

## Deployment and secret cleanup

The old README linked a Streamlit deployment; its current availability has not been verified. Deploy `streamlit_app.py` through your chosen host and configure secrets privately. No cloud resource is provisioned or database migrated by committing this code.

`.env` and `.streamlit/secrets.toml` were previously tracked. Their removal does not erase Git history. **Revoke/rotate any real database password, API key or other credential ever committed**, and update the hosting provider's private settings. No credential rotation is performed by this repository. `.env.example` contains placeholders only. Runtime packages are pinned; update them deliberately and run CI.
