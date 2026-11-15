"""Validate the source and atomically load a named, immutable sales snapshot."""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from utils import get_db_url

HEADERS = {
    'name': ('Name', 'CustomerName'), 'address': ('Address',), 'city': ('City',),
    'country': ('Country',), 'region': ('Region',),
    'products': ('ProductName', 'ProductNames'),
    'categories': ('ProductCategory', 'ProductCategories'),
    'descriptions': ('ProductCategoryDescription', 'ProductDescriptions'),
    'prices': ('ProductUnitPrice', 'ProductUnitPrices'),
    'quantities': ('QuantityOrderded', 'Quantities', 'QuantityOrdered'),
    'dates': ('OrderDate', 'Dates'),
}
DATE_FORMATS = ('%Y%m%d', '%Y-%m-%d', '%m/%d/%Y')
DDL = """
CREATE TABLE region (regionid SERIAL PRIMARY KEY, region TEXT UNIQUE NOT NULL);
CREATE TABLE country (countryid SERIAL PRIMARY KEY, country TEXT NOT NULL,
 regionid INTEGER NOT NULL REFERENCES region, UNIQUE(country,regionid));
CREATE TABLE customer (customerid SERIAL PRIMARY KEY, firstname TEXT NOT NULL,
 lastname TEXT NOT NULL, address TEXT NOT NULL, city TEXT NOT NULL,
 countryid INTEGER NOT NULL REFERENCES country,
 UNIQUE(firstname,lastname,address,city,countryid));
CREATE TABLE productcategory (productcategoryid SERIAL PRIMARY KEY,
 productcategory TEXT UNIQUE NOT NULL, productcategorydescription TEXT NOT NULL);
CREATE TABLE product (productid SERIAL PRIMARY KEY, productname TEXT NOT NULL,
 productunitprice NUMERIC(18,4) NOT NULL CHECK(productunitprice >= 0),
 productcategoryid INTEGER NOT NULL REFERENCES productcategory,
 UNIQUE(productname,productcategoryid,productunitprice));
CREATE TABLE etl_loads (dataset_name TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL,
 source_rows INTEGER NOT NULL, rejected_rows INTEGER NOT NULL,
 fact_rows INTEGER NOT NULL, loaded_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE orderdetail (orderid BIGSERIAL PRIMARY KEY,
 customerid INTEGER NOT NULL REFERENCES customer,
 productid INTEGER NOT NULL REFERENCES product, orderdate DATE NOT NULL,
 quantityordered INTEGER NOT NULL CHECK(quantityordered > 0),
 source_dataset TEXT NOT NULL REFERENCES etl_loads(dataset_name),
 source_row INTEGER NOT NULL, source_item INTEGER NOT NULL,
 UNIQUE(source_dataset,source_row,source_item));
CREATE INDEX ON orderdetail(orderdate);
CREATE INDEX ON orderdetail(customerid);
CREATE INDEX ON orderdetail(productid);
"""

@dataclass
class Source:
    sha256: str
    source_rows: int
    orders: list
    rejections: list

    def report(self):
        return {'source_sha256': self.sha256, 'source_rows': self.source_rows,
                'accepted_rows': self.source_rows - len(self.rejections),
                'rejected_rows': len(self.rejections), 'line_items': len(self.orders),
                'reasons': dict(Counter(r['reason'] for r in self.rejections)),
                'rejections': self.rejections}


def parse_date(value):
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError('invalid_date')


def parse_source(path):
    """Reject a whole source record on any invalid item; report reasons without PII."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    csv.field_size_limit(64 * 1024 * 1024)
    orders, rejected, categories, countries = [], [], {}, {}
    count = 0
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        fields = reader.fieldnames or []
        mapping = {key: next((h for h in aliases if h in fields), None)
                   for key, aliases in HEADERS.items()}
        missing = [key for key, value in mapping.items() if value is None]
        if missing:
            raise ValueError('Missing source columns: ' + ', '.join(missing))
        for row_number, raw in enumerate(reader, start=2):
            count += 1
            try:
                if None in raw or any(raw.get(h) is None for h in mapping.values()):
                    raise ValueError('invalid_column_count')
                row = {key: raw[h].strip() for key, h in mapping.items()}
                if not all(row[k] for k in ('name', 'address', 'city', 'country', 'region')):
                    raise ValueError('missing_customer_or_geography')
                full_name = row['name'].split(maxsplit=1)
                customer = (full_name[0], full_name[1] if len(full_name) > 1 else '',
                            row['address'], row['city'], row['country'], row['region'])
                lists = {key: [s.strip() for s in row[key].split(';')]
                         for key in ('products', 'categories', 'descriptions', 'prices', 'quantities', 'dates')}
                if len({len(v) for v in lists.values()}) != 1:
                    raise ValueError('mismatched_item_counts')
                pending, pending_categories = [], {}
                for item, values in enumerate(zip(*lists.values()), start=1):
                    name, category, description, price_text, qty_text, date_text = values
                    if not name or not category or not description:
                        raise ValueError('missing_product_fields')
                    try:
                        price = Decimal(price_text)
                        if not price.is_finite() or price < 0 or price >= Decimal('100000000000000') or price != price.quantize(Decimal('.0001')):
                            raise ValueError('invalid_price')
                        quantity = int(qty_text)
                    except (InvalidOperation, OverflowError):
                        raise ValueError('invalid_price') from None
                    except ValueError as exc:
                        if str(exc) == 'invalid_price':
                            raise
                        raise ValueError('invalid_quantity') from None
                    if quantity <= 0 or quantity > 2147483647:
                        raise ValueError('invalid_quantity')
                    date = parse_date(date_text)
                    previous = pending_categories.get(category, categories.get(category))
                    if previous is not None and previous != description:
                        raise ValueError('conflicting_category_description')
                    pending_categories[category] = description
                    pending.append((customer, (name, category, price), description, date, quantity, row_number, item))
                if row['country'] in countries and countries[row['country']] != row['region']:
                    raise ValueError('conflicting_country_region')
                countries[row['country']] = row['region']
                categories.update(pending_categories)
                orders.extend(pending)
            except ValueError as exc:
                rejected.append({'source_row': row_number, 'reason': str(exc)})
    return Source(digest, count, orders, rejected)


def load_source(source, database_url, schema='sales_analytics', dataset='orders', allow_rejects=False):
    if source.rejections and not allow_rejects:
        raise ValueError('Validation rejected source rows. Review the report; no database changes were made.')
    if not source.orders:
        raise ValueError('No valid line items; refusing an empty load.')
    if not dataset or len(dataset) > 120:
        raise ValueError('Use a dataset name between 1 and 120 characters.')
    conn = psycopg2.connect(database_url, connect_timeout=10)
    try:
        with conn:
            with conn.cursor() as cur:
                # Serialize loaders for this schema, including first-time initialization.
                cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ('sales-etl:' + schema,))
                cur.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(schema)))
                cur.execute(sql.SQL('SET LOCAL search_path TO {}, pg_catalog').format(sql.Identifier(schema)))
                cur.execute('SELECT tablename FROM pg_tables WHERE schemaname=%s', (schema,))
                existing = {r[0] for r in cur.fetchall()}
                managed = {'region','country','customer','productcategory','product','orderdetail','etl_loads'}
                if 'etl_loads' not in existing:
                    if existing & managed:
                        raise ValueError('Existing unmanaged sales tables found. Choose a new schema; the loader will not replace them.')
                    cur.execute(DDL)
                cur.execute('SELECT source_sha256 FROM etl_loads WHERE dataset_name=%s', (dataset,))
                previous = cur.fetchone()
                if previous:
                    if previous[0] == source.sha256:
                        return {'status':'unchanged', 'dataset':dataset, 'inserted_line_items':0}
                    raise ValueError('This dataset name already contains a different snapshot. No changes made. Use a new name only for genuinely independent data, not a revised copy.')
                def insert(table, columns, rows):
                    if rows:
                        query = sql.SQL('INSERT INTO {} ({}) VALUES %s ON CONFLICT DO NOTHING').format(
                            sql.Identifier(table), sql.SQL(',').join(map(sql.Identifier, columns)))
                        execute_values(cur, query.as_string(conn), rows, page_size=2000)
                customers = sorted({o[0] for o in source.orders})
                insert('region', ['region'], sorted({(c[5],) for c in customers}))
                cur.execute('SELECT region,regionid FROM region'); regions = dict(cur.fetchall())
                insert('country', ['country','regionid'], sorted({(c[4],regions[c[5]]) for c in customers}))
                cur.execute('SELECT country,regionid,countryid FROM country'); countries = {(a,b):i for a,b,i in cur.fetchall()}
                cust_rows = [(*c[:4],countries[(c[4],regions[c[5]])]) for c in customers]
                insert('customer', ['firstname','lastname','address','city','countryid'], cust_rows)
                cur.execute('SELECT firstname,lastname,address,city,countryid,customerid FROM customer')
                cust_map = {tuple(r[:-1]):r[-1] for r in cur.fetchall()}
                category_rows = sorted({(o[1][1],o[2]) for o in source.orders})
                cur.execute('SELECT productcategory,productcategorydescription FROM productcategory')
                old_categories = dict(cur.fetchall())
                if any(c in old_categories and old_categories[c] != d for c,d in category_rows):
                    raise ValueError('Category descriptions conflict with existing data.')
                insert('productcategory', ['productcategory','productcategorydescription'], category_rows)
                cur.execute('SELECT productcategory,productcategoryid FROM productcategory'); cats = dict(cur.fetchall())
                products = sorted({(o[1][0],o[1][2],cats[o[1][1]]) for o in source.orders})
                insert('product', ['productname','productunitprice','productcategoryid'], products)
                cur.execute('SELECT productname,productunitprice,productcategoryid,productid FROM product')
                product_map = {(n,p,c):i for n,p,c,i in cur.fetchall()}
                cur.execute('INSERT INTO etl_loads(dataset_name,source_sha256,source_rows,rejected_rows,fact_rows) VALUES(%s,%s,%s,%s,%s)',
                            (dataset,source.sha256,source.source_rows,len(source.rejections),len(source.orders)))
                batch = []
                for customer, product, _, date, qty, row, item in source.orders:
                    cust = (*customer[:4],countries[(customer[4],regions[customer[5]])])
                    prod = (product[0],product[2],cats[product[1]])
                    batch.append((cust_map[cust],product_map[prod],date,qty,dataset,row,item))
                    if len(batch) == 5000:
                        execute_values(cur,'INSERT INTO orderdetail(customerid,productid,orderdate,quantityordered,source_dataset,source_row,source_item) VALUES %s',batch)
                        batch.clear()
                if batch:
                    execute_values(cur,'INSERT INTO orderdetail(customerid,productid,orderdate,quantityordered,source_dataset,source_row,source_item) VALUES %s',batch)
        return {'status':'loaded', 'dataset':dataset, 'inserted_line_items':len(source.orders)}
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, default=Path(__file__).with_name('data.csv'))
    parser.add_argument('--schema', default=os.getenv('APP_DB_SCHEMA','sales_analytics'))
    parser.add_argument('--dataset', default='orders')
    parser.add_argument('--report', type=Path, default=Path('etl_report.json'))
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--allow-rejects', action='store_true')
    args = parser.parse_args()
    try:
        source = parse_source(args.file)
        args.report.write_text(json.dumps(source.report(), indent=2) + '\n')
        print(json.dumps({k:v for k,v in source.report().items() if k != 'rejections'}))
        if args.validate_only:
            if source.rejections and not args.allow_rejects:
                parser.exit(1, 'Validation failed. See the report; no database connection was opened.\n')
        else:
            print(json.dumps(load_source(source,get_db_url(),args.schema,args.dataset,args.allow_rejects)))
    except (ValueError, OSError, csv.Error) as exc:
        parser.exit(1, str(exc) + '\n')
    except psycopg2.Error:
        parser.exit(1, 'Database load failed and was rolled back. Check connectivity and loader permissions.\n')

if __name__ == '__main__':
    main()
