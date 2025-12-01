import psycopg2
from psycopg2 import extras
from datetime import datetime

from utils import get_db_url

DATA_FILE = "data.csv"   

DDL_SQL = """
DROP TABLE IF EXISTS orderdetail CASCADE;
DROP TABLE IF EXISTS product CASCADE;
DROP TABLE IF EXISTS productcategory CASCADE;
DROP TABLE IF EXISTS customer CASCADE;
DROP TABLE IF EXISTS country CASCADE;
DROP TABLE IF EXISTS region CASCADE;

CREATE TABLE region (
    regionid   SERIAL PRIMARY KEY,
    region     TEXT NOT NULL
);

CREATE TABLE country (
    countryid  SERIAL PRIMARY KEY,
    country    TEXT NOT NULL,
    regionid   INTEGER NOT NULL REFERENCES region(regionid)
);

CREATE TABLE customer (
    customerid SERIAL PRIMARY KEY,
    firstname  TEXT NOT NULL,
    lastname   TEXT NOT NULL,
    address    TEXT NOT NULL,
    city       TEXT NOT NULL,
    countryid  INTEGER NOT NULL REFERENCES country(countryid)
);

CREATE TABLE productcategory (
    productcategoryid SERIAL PRIMARY KEY,
    productcategory   TEXT NOT NULL,
    productcategorydescription TEXT NOT NULL
);

CREATE TABLE product (
    productid   SERIAL PRIMARY KEY,
    productname TEXT NOT NULL,
    productunitprice REAL NOT NULL,
    productcategoryid INTEGER NOT NULL REFERENCES productcategory(productcategoryid)
);

CREATE TABLE orderdetail (
    orderid        SERIAL PRIMARY KEY,
    customerid     INTEGER NOT NULL REFERENCES customer(customerid),
    productid      INTEGER NOT NULL REFERENCES product(productid),
    orderdate      DATE NOT NULL,
    quantityordered INTEGER NOT NULL
);
"""


def parse_regions(path):
    regions = set()
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 4 and parts[4].strip():
                regions.add(parts[4].strip())
    return sorted(regions)


def parse_countries(path):
    # returns set of (country, region)
    pairs = set()
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 4:
                country = parts[3].strip()
                region = parts[4].strip()
                if country and region:
                    pairs.add((country, region))
    # sort by country name
    return sorted(pairs, key=lambda x: x[0])


def parse_productcategories(path):
    # returns set of (category, description)
    cats = set()
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 7:
                cat_list = parts[6].split(";")
                desc_list = parts[7].split(";")
                for cat, desc in zip(cat_list, desc_list):
                    cat = cat.strip()
                    desc = desc.strip()
                    if cat:
                        cats.add((cat, desc))
    return sorted(cats, key=lambda x: x[0])


def parse_products(path):
    # returns set of (productname, category, unitprice)
    prods = set()
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 8:
                names = parts[5].split(";")
                cats = parts[6].split(";")
                prices = parts[8].split(";")
                for n, c, p in zip(names, cats, prices):
                    n = n.strip()
                    c = c.strip()
                    p = p.strip()
                    if n and c and p:
                        try:
                            price = float(p)
                        except ValueError:
                            continue
                        prods.add((n, c, price))
    return sorted(prods, key=lambda x: x[0])


def parse_customers(path, valid_countries):
    # returns set of (firstname, lastname, address, city, country)
    custs = set()
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 4:
                name = parts[0].strip()
                address = parts[1].strip()
                city = parts[2].strip()
                country = parts[3].strip()
                if not country or country not in valid_countries:
                    continue
                if not name:
                    continue
                name_parts = name.split()
                first = name_parts[0]
                last = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
                custs.add((first, last, address, city, country))
    # sort by "First Last"
    return sorted(custs, key=lambda x: (x[0] + " " + x[1]))


def parse_orders(path, customer_map, product_map):
    """
    customer_map: "First Last" -> customerid
    product_map: productname -> productid
    returns list of (customerid, productid, orderdate, quantity)
    """
    orders = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 10:
                name = " ".join(parts[0].split()).strip()
                if name not in customer_map:
                    continue
                cust_id = customer_map[name]

                prod_names = [x.strip() for x in parts[5].split(";")]
                qtys = [x.strip() for x in parts[9].split(";")]
                dates = [x.strip() for x in parts[10].split(";")]

                for pn, q, dt in zip(prod_names, qtys, dates):
                    if pn not in product_map:
                        continue
                    try:
                        qty = int(q)
                        orderdate = datetime.strptime(dt, "%Y%m%d").date()
                    except Exception:
                        continue
                    orders.append(
                        (cust_id, product_map[pn], orderdate, qty)
                    )
    return orders


def main():
    db_url = get_db_url()
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    print("Dropping and creating tables...")
    cur.execute(DDL_SQL)
    conn.commit()
    print("✅ Tables created")

    # ---------- REGION ----------
    print("Inserting regions...")
    regions = parse_regions(DATA_FILE)
    extras.execute_batch(
        cur,
        "INSERT INTO region (region) VALUES (%s)",
        [(r,) for r in regions],
        page_size=1000,
    )
    conn.commit()

    # build region map
    cur.execute("SELECT region, regionid FROM region")
    region_map = {r: rid for r, rid in cur.fetchall()}

    # ---------- COUNTRY ----------
    print("Inserting countries...")
    country_pairs = parse_countries(DATA_FILE)
    country_rows = [
        (country, region_map[region])
        for (country, region) in country_pairs
        if region in region_map
    ]
    extras.execute_batch(
        cur,
        "INSERT INTO country (country, regionid) VALUES (%s, %s)",
        country_rows,
        page_size=1000,
    )
    conn.commit()

    cur.execute("SELECT country, countryid FROM country")
    country_map = {c: cid for c, cid in cur.fetchall()}

    # ---------- PRODUCT CATEGORY ----------
    print("Inserting product categories...")
    categories = parse_productcategories(DATA_FILE)
    extras.execute_batch(
        cur,
        "INSERT INTO productcategory (productcategory, productcategorydescription) VALUES (%s, %s)",
        categories,
        page_size=1000,
    )
    conn.commit()

    cur.execute("SELECT productcategory, productcategoryid FROM productcategory")
    cat_map = {c: cid for c, cid in cur.fetchall()}

    # ---------- PRODUCT ----------
    print("Inserting products...")
    products_raw = parse_products(DATA_FILE)
    product_rows = [
        (name, price, cat_map[cat])
        for (name, cat, price) in products_raw
        if cat in cat_map
    ]
    extras.execute_batch(
        cur,
        "INSERT INTO product (productname, productunitprice, productcategoryid) VALUES (%s, %s, %s)",
        product_rows,
        page_size=1000,
    )
    conn.commit()

    cur.execute("SELECT productname, productid FROM product")
    product_map = {n: pid for n, pid in cur.fetchall()}

    # ---------- CUSTOMER ----------
    print("Inserting customers...")
    customers_raw = parse_customers(DATA_FILE, set(country_map.keys()))
    customer_rows = [
        (first, last, address, city, country_map[country])
        for (first, last, address, city, country) in customers_raw
    ]
    extras.execute_batch(
        cur,
        "INSERT INTO customer (firstname, lastname, address, city, countryid) VALUES (%s, %s, %s, %s, %s)",
        customer_rows,
        page_size=1000,
    )
    conn.commit()

    cur.execute("SELECT firstname, lastname, customerid FROM customer")
    cust_map = {f"{f} {l}".strip(): cid for f, l, cid in cur.fetchall()}

    # ---------- ORDERDETAIL ----------
    print("Inserting order details...")
    orders = parse_orders(DATA_FILE, cust_map, product_map)
    extras.execute_batch(
        cur,
        "INSERT INTO orderdetail (customerid, productid, orderdate, quantityordered) VALUES (%s, %s, %s, %s)",
        orders,
        page_size=5000,
    )
    conn.commit()

    cur.close()
    conn.close()
    print("✅ Finished populating mini-project2 sales database")


if __name__ == "__main__":
    main()