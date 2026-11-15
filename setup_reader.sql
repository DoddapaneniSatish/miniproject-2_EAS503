-- Run as the database owner AFTER loading into sales_analytics.
-- Create a LOGIN role named sales_reader in your provider dashboard first,
-- with a generated password kept in the provider's secret settings.
-- This script intentionally contains no password and is not run automatically.
REVOKE ALL ON SCHEMA sales_analytics FROM sales_reader;
GRANT USAGE ON SCHEMA sales_analytics TO sales_reader;
REVOKE ALL ON ALL TABLES IN SCHEMA sales_analytics FROM sales_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA sales_analytics TO sales_reader;
ALTER ROLE sales_reader SET default_transaction_read_only = on;
-- Do not grant membership in an owner or writer role to sales_reader.
