"""Build analysis-ready tables from the raw Olist CSVs with DuckDB.

Tables written to data/processed/olist.duckdb:
  orders            one row per non-cancelled order, keyed to the real customer (customer_unique_id)
  customers         one row per real customer with RFM metrics and segment
  cohort_retention  monthly acquisition cohorts x months since first order
  first_orders      one row per first-time buyer: first-order features + "bought again within 180 days" label
"""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "processed" / "olist.duckdb"
REPEAT_WINDOW_DAYS = 180


def raw(name: str) -> str:
    return f"read_csv_auto('{RAW / name}.csv', header = true)"


def build(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    # Orders: one row per order that was not cancelled/unavailable, with value, delivery and review facts.
    con.execute(f"""
    create or replace table orders as
    with items as (
        select order_id,
               sum(price) as items_value,
               sum(freight_value) as freight_value,
               count(*) as item_count,
               count(distinct seller_id) as seller_count,
               arg_max(product_id, price) as main_product_id
        from {raw('olist_order_items_dataset')}
        group by 1
    ),
    payments as (
        select order_id,
               arg_max(payment_type, payment_value) as payment_type,
               max(payment_installments) as installments,
               bool_or(payment_type = 'voucher') as used_voucher
        from {raw('olist_order_payments_dataset')}
        group by 1
    ),
    reviews as (
        -- raw reviews are not unique per order; keep the latest answered one
        select order_id, review_score
        from {raw('olist_order_reviews_dataset')}
        qualify row_number() over (partition by order_id order by review_answer_timestamp desc) = 1
    ),
    products as (
        select p.product_id,
               coalesce(t.product_category_name_english, p.product_category_name, 'unknown') as category,
               p.product_photos_qty as photo_count,
               p.product_weight_g as weight_g
        from {raw('olist_products_dataset')} p
        left join {raw('product_category_name_translation')} t using (product_category_name)
    )
    select
        o.order_id,
        c.customer_unique_id,
        upper(c.customer_state) as customer_state,
        o.order_purchase_timestamp as purchased_at,
        date_trunc('month', o.order_purchase_timestamp)::date as order_month,
        i.items_value + i.freight_value as order_value,
        i.items_value,
        i.freight_value,
        i.item_count,
        i.seller_count,
        pr.category,
        pr.photo_count,
        pr.weight_g,
        p.payment_type,
        p.installments,
        coalesce(p.used_voucher, false) as used_voucher,
        date_diff('day', o.order_purchase_timestamp, o.order_delivered_customer_date) as delivery_days,
        date_diff('day', o.order_estimated_delivery_date::date, o.order_delivered_customer_date::date) as days_vs_estimate,
        r.review_score
    from {raw('olist_orders_dataset')} o
    join {raw('olist_customers_dataset')} c using (customer_id)
    join items i using (order_id)
    left join payments p using (order_id)
    left join reviews r using (order_id)
    left join products pr on pr.product_id = i.main_product_id
    where o.order_status not in ('canceled', 'unavailable')
    """)

    # Customers: RFM snapshot as of the day after the last order in the data.
    con.execute("""
    create or replace table customers as
    with snapshot as (select max(purchased_at)::date + 1 as snapshot_date from orders),
    rfm as (
        select
            customer_unique_id,
            any_value(customer_state) as customer_state,
            min(purchased_at) as first_order_at,
            max(purchased_at) as last_order_at,
            count(*) as frequency,
            sum(order_value) as monetary,
            avg(review_score) as avg_review_score
        from orders
        group by 1
    ),
    scored as (
        select
            rfm.*,
            date_diff('day', last_order_at::date, snapshot.snapshot_date) as recency_days,
            date_trunc('month', first_order_at)::date as cohort_month,
            ntile(4) over (order by date_diff('day', last_order_at::date, snapshot.snapshot_date) desc) as r_score,
            ntile(4) over (order by monetary) as m_score
        from rfm, snapshot
    )
    select
        *,
        case
            when frequency > 1 and r_score >= 3 then 'Champions'
            when frequency > 1                  then 'Loyal, lapsing'
            when r_score >= 3 and m_score = 4   then 'New high-value'
            when r_score >= 3                   then 'New, low-value'
            when m_score = 4                    then 'Lost high-value'
            else                                     'Lost low-value'
        end as segment
    from scored
    """)

    # Cohort retention: share of each first-purchase cohort active N months later.
    con.execute("""
    create or replace table cohort_retention as
    with activity as (
        select c.cohort_month,
               date_diff('month', c.cohort_month, o.order_month) as months_since_first,
               count(distinct o.customer_unique_id) as active_customers,
               sum(o.order_value) as revenue
        from orders o join customers c using (customer_unique_id)
        group by 1, 2
    ),
    sizes as (select cohort_month, count(*) as cohort_size from customers group by 1)
    select a.*, s.cohort_size,
           a.active_customers / s.cohort_size as retention_rate,
           sum(a.revenue) over (partition by a.cohort_month order by a.months_since_first) / s.cohort_size
               as cumulative_revenue_per_customer
    from activity a join sizes s using (cohort_month)
    order by 1, 2
    """)

    # First orders + label: did the customer buy again within the window?
    # Only first orders with a complete observation window are labelled.
    con.execute(f"""
    create or replace table first_orders as
    with ranked as (
        select *,
               row_number() over (partition by customer_unique_id order by purchased_at, order_id) as n,
               lead(purchased_at) over (partition by customer_unique_id order by purchased_at, order_id) as next_order_at
        from orders
    ),
    data_end as (select max(purchased_at) as end_at from orders)
    select
        r.* exclude (n, next_order_at),
        dayofweek(purchased_at) as purchase_dow,
        hour(purchased_at) as purchase_hour,
        month(purchased_at) as purchase_month,
        freight_value / nullif(order_value, 0) as freight_share,
        days_vs_estimate > 0 as delivered_late,
        coalesce(date_diff('day', purchased_at, next_order_at) <= {REPEAT_WINDOW_DAYS}, false) as bought_again
    from ranked r, data_end
    where n = 1
      and purchased_at <= data_end.end_at - interval {REPEAT_WINDOW_DAYS} day
    """)

    for t in ["orders", "customers", "cohort_retention", "first_orders"]:
        print(f"{t:18s} {con.execute(f'select count(*) from {t}').fetchone()[0]:>8,} rows")
    con.close()


if __name__ == "__main__":
    build()
