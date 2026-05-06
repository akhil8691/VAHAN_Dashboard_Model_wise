import gdown
import hashlib
import re
import time
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import duckdb
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Vehicle CSV Dashboard", layout="wide")

# ---------------------------
# Config
# ---------------------------
# Cloud note:
# Do not rely on a Windows/Desktop path after deployment.
# Use CSV upload or a Google Drive / Google Sheets link instead.
DEFAULT_LOCAL_CSV = r"C:\Users\Akhil-EUR0750\OneDrive\OneDrive - Euler Motors Pvt Ltd\Desktop\tata-motors-2026-01-05.csv"

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s).strip())
    return s[:180] if s else "data"


def sql_quote_path(path: str) -> str:
    """Escape a file path safely for use inside DuckDB SQL string literals."""
    return str(path).replace("'", "''")


def google_link_to_download_url(url: str) -> str:
    """
    Converts common Google Drive / Google Sheets sharing links to direct CSV/download URLs.

    Supported:
    1. Google Drive file links:
       https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    2. Google Drive open links:
       https://drive.google.com/open?id=FILE_ID
    3. Google Drive uc links:
       https://drive.google.com/uc?id=FILE_ID
    4. Google Sheets links:
       https://docs.google.com/spreadsheets/d/SHEET_ID/edit#gid=0
    5. Direct CSV URLs.
    """
    url = str(url).strip()

    if not url:
        return url

    # Google Sheets -> export selected sheet as CSV
    if "docs.google.com/spreadsheets/d/" in url:
        sheet_id = url.split("/d/")[1].split("/")[0]

        parsed = urlparse(url)
        qs = parse_qs(parsed.query)

        gid = "0"
        if "gid" in qs:
            gid = qs["gid"][0]
        elif "#gid=" in url:
            gid = url.split("#gid=")[-1].split("&")[0]

        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

    # Google Drive file link -> direct file download
    if "drive.google.com/file/d/" in url:
        file_id = url.split("/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    # Google Drive open / uc links
    if "drive.google.com" in url:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        file_id = qs.get("id", [None])[0]
        if file_id:
            return f"https://drive.google.com/uc?export=download&id={file_id}"

    return url


def looks_like_html(content: bytes, content_type: str = "") -> bool:
    """Detect whether a response is an HTML page rather than a real CSV file."""
    content_type = (content_type or "").lower()
    start = content[:500].strip().lower()
    return (
        "text/html" in content_type
        or start.startswith(b"<!doctype html")
        or start.startswith(b"<html")
    )


def download_csv_from_link(url: str, refresh: bool = False) -> str:
    """
    Download CSV from Google Drive / Google Sheets / direct CSV URL.
    Uses gdown for Google Drive links because Drive often returns HTML pages.
    """
    url = str(url).strip()

    if not url:
        raise ValueError("Please paste a Google Drive, Google Sheets, or direct CSV link.")

    download_url = google_link_to_download_url(url)

    key = hashlib.md5(download_url.encode("utf-8")).hexdigest()
    local_path = CACHE_DIR / f"drive_data_{key}.csv"

    if local_path.exists() and not refresh:
        return str(local_path)

    if "drive.google.com" in url:
        output = str(local_path)
        result = gdown.download(url, output, quiet=False)

        if result is None or not Path(output).exists():
            raise ValueError(
                "Google Drive download failed. Make sure the file is shared as "
                "'Anyone with the link - Viewer'."
            )

        if Path(output).stat().st_size == 0:
            raise ValueError("Downloaded Google Drive file is empty.")

        return output

    response = requests.get(download_url, timeout=180)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")

    if looks_like_html(response.content, content_type):
        raise ValueError(
            "The link returned an HTML page instead of CSV. "
            "Check that the file is shared as 'Anyone with the link - Viewer', "
            "or use a Google Sheets link that can be exported as CSV."
        )

    local_path.write_bytes(response.content)
    return str(local_path)


def parse_mmyy_to_index(mmyy: str):
    """MM-YY -> (YYYY*12 + MM), assumes 20YY; returns None if invalid."""
    if mmyy is None:
        return None
    mmyy = str(mmyy).strip()
    m = re.match(r"^(\d{2})-(\d{2})$", mmyy)
    if not m:
        return None
    mm = int(m.group(1))
    yy = int(m.group(2))
    if not (1 <= mm <= 12):
        return None
    year = 2000 + yy
    return year * 12 + mm

def is_parquet_file(path: str) -> bool:
    """Parquet files usually start and end with PAR1."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PAR1"
    except Exception:
        return False


def ensure_data_file(data_path: str) -> str:
    """
    If file is already Parquet, use it directly.
    If file is CSV, convert it to cached Parquet.
    """
    if is_parquet_file(data_path):
        return data_path
    return ensure_parquet(data_path)

def index_to_mmyy(idx: int) -> str:
    year = idx // 12
    mm = idx % 12
    if mm == 0:
        year -= 1
        mm = 12
    return f"{mm:02d}-{str(year)[-2:]}"


@st.cache_resource(show_spinner=False)
def get_con():
    return duckdb.connect(database=":memory:")


def build_where_in(col, values):
    if not values:
        return None
    safe = [str(v).replace("'", "''") for v in values]
    quoted = ",".join([f"'{v}'" for v in safe])
    return f"{col} IN ({quoted})"


def month_index_expr(month_col: str) -> str:
    """DuckDB SQL expr: MM-YY -> YYYY*12+MM else NULL."""
    return f"""
    (
      CASE
        WHEN regexp_matches({month_col}, '^[0-9]{{2}}-[0-9]{{2}}$')
        THEN ( (2000 + CAST(substr({month_col}, 4, 2) AS INTEGER)) * 12
               + CAST(substr({month_col}, 1, 2) AS INTEGER) )
        ELSE NULL
      END
    )
    """


def normalized_model_expr(model_col: str) -> str:
    """
    DuckDB SQL expression to normalize model names to group variants:
    - UPPER
    - remove anything in (...)
    - remove suffix after '-'
    - collapse multiple spaces
    """
    return f"""
    regexp_replace(
      regexp_replace(
        regexp_replace(upper(coalesce({model_col}, '')),
          '\\\\s*\\\\(.*?\\\\)\\\\s*', ' ', 'g'
        ),
        '\\\\s*-\\\\s*.*$', '', 'g'
      ),
      '\\\\s+', ' ', 'g'
    )
    """


def ensure_parquet(csv_path: str) -> str:
    """CSV -> Parquet cached by filename+mtime+size."""
    p = Path(csv_path)
    stat = p.stat()
    key = f"{safe_filename(p.name)}__{stat.st_size}__{int(stat.st_mtime)}"
    parquet_path = CACHE_DIR / f"{key}.parquet"

    if parquet_path.exists():
        return str(parquet_path)

    st.info("First run: converting CSV to Parquet for faster reloads...")
    t0 = time.time()
    con = get_con()

    csv_sql = sql_quote_path(str(p))
    parquet_sql = sql_quote_path(str(parquet_path))

    con.execute(f"""
        COPY (
            SELECT * FROM read_csv_auto(
                '{csv_sql}',
                header=true,
                sample_size=200000,
                strict_mode=false,
                ignore_errors=true
            )
        )
        TO '{parquet_sql}'
        (FORMAT PARQUET);
    """)

    st.success(f"Parquet created in {time.time() - t0:.1f}s: {parquet_path.name}")
    return str(parquet_path)


@st.cache_data(show_spinner=False)
def distinct_vals(con_id: int, view_name: str, col: str, limit: int = 50000):
    con = get_con()
    q = f"SELECT DISTINCT {col} AS v FROM {view_name} WHERE {col} IS NOT NULL ORDER BY 1 LIMIT {limit}"
    return [r[0] for r in con.execute(q).fetchall()]


@st.cache_data(show_spinner=False)
def get_month_bounds(con_id: int, view_name: str, month_col: str):
    con = get_con()
    df = con.execute(f"SELECT DISTINCT {month_col} AS m FROM {view_name} WHERE {month_col} IS NOT NULL").df()
    idxs = [parse_mmyy_to_index(x) for x in df["m"].tolist()]
    idxs = [i for i in idxs if i is not None]
    if not idxs:
        return None, None
    return min(idxs), max(idxs)


@st.cache_data(show_spinner=False)
def search_normalized_models(
    con_id: int,
    view_name: str,
    maker_col: str,
    model_col: str,
    vehicle_count_col: str,
    maker_name: str,
    query: str,
    limit: int = 200,
):
    """
    Search normalized model buckets for a maker containing query,
    sorted by registrations desc then normalizedModel asc.
    Returns DataFrame: normalizedModel, registrations.
    """
    con = get_con()
    safe_maker = str(maker_name).replace("'", "''")
    q = str(query).strip().replace("'", "''")

    if not q:
        return pd.DataFrame(columns=["normalizedModel", "registrations"])

    n_expr = normalized_model_expr(model_col)

    sql = f"""
        SELECT
          NULLIF(trim({n_expr}), '') AS normalizedModel,
          SUM(CAST({vehicle_count_col} AS BIGINT)) AS registrations
        FROM {view_name}
        WHERE {maker_col} = '{safe_maker}'
          AND {model_col} IS NOT NULL
          AND {n_expr} ILIKE '%{q}%'
        GROUP BY 1
        HAVING normalizedModel IS NOT NULL
        ORDER BY registrations DESC, normalizedModel ASC
        LIMIT {int(limit)}
    """
    return con.execute(sql).df()


# ---------------------------
# Column mapping
# ---------------------------
COLS = {
    "registrationYear": "registrationYear",
    "financialYear": "financialYear",
    "registrationMonthMMYY": "registrationMonthMMYY",
    "makerName": "makerName",
    "stateName": "stateName",
    "rtoCode": "rtoCode",
    "rtoName": "rtoName",
    "vehicleCategoryName": "vehicleCategoryName",
    "vehicleModelName": "vehicleModelName",
    "fuelName": "fuelName",
    "vehicleClassName": "vehicleClassName",
    "grossVehicleWeight": "grossVehicleWeight",
    "pollutionNorm": "pollutionNorm",
    "saleType": "saleType",
    "vehicleCount": "vehicleCount",
}

# ---------------------------
# UI: Data source
# ---------------------------
st.title("Vehicle Registrations Dashboard")

with st.sidebar:
    st.header("Data Source")

    uploaded = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])

    drive_url = st.text_input(
        "Google Drive / Google Sheets / CSV / Parquet link",
        value="https://drive.google.com/file/d/1HfvP46RKrPe-RD2ccdXJxGgHD9y9rxND/view?usp=sharing",
        placeholder="Paste public Google Drive, Google Sheets, or CSV URL",
    )

    refresh_drive = st.button("Refresh Google Drive data")

    with st.expander("Local CSV path - for laptop/local use only"):
        use_local = st.checkbox("Use local CSV path", value=False)
        local_csv_path = st.text_input("Local CSV path", value=DEFAULT_LOCAL_CSV)

csv_path = None

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
        f.write(uploaded.read())
        csv_path = f.name
    st.sidebar.success("Uploaded CSV loaded.")

elif drive_url.strip():
    try:
        csv_path = download_csv_from_link(drive_url, refresh=refresh_drive)
        st.sidebar.success("Google/URL CSV loaded.")
    except Exception as e:
        st.error(f"Could not load the linked CSV: {e}")
        st.stop()

elif use_local and local_csv_path:
    csv_path = local_csv_path

else:
    st.info("Upload a CSV, paste a Google Drive/Google Sheets link, or enable local path.")
    st.stop()

if not Path(csv_path).exists():
    st.error(f"CSV file not found: {csv_path}")
    st.stop()

# ---------------------------
# Parquet + DuckDB view
# ---------------------------
parquet_path = ensure_data_file(csv_path)
con = get_con()
parquet_sql = sql_quote_path(parquet_path)
con.execute(f"CREATE OR REPLACE VIEW vdata AS SELECT * FROM read_parquet('{parquet_sql}')")

# ---------------------------
# Sidebar filters
# ---------------------------
st.sidebar.header("Filters")

con_id = id(con)
VIEW = "vdata"

years = distinct_vals(con_id, VIEW, COLS["registrationYear"])
fin_years = distinct_vals(con_id, VIEW, COLS["financialYear"])
states = distinct_vals(con_id, VIEW, COLS["stateName"])
makers = distinct_vals(con_id, VIEW, COLS["makerName"])
fuels = distinct_vals(con_id, VIEW, COLS["fuelName"])
categories = distinct_vals(con_id, VIEW, COLS["vehicleCategoryName"])
vehicle_classes = distinct_vals(con_id, VIEW, COLS["vehicleClassName"])
sale_types = distinct_vals(con_id, VIEW, COLS["saleType"])
poll_norms = distinct_vals(con_id, VIEW, COLS["pollutionNorm"])
months_list = distinct_vals(con_id, VIEW, COLS["registrationMonthMMYY"], limit=2000)

sel_years = st.sidebar.multiselect("Registration Year", years, default=years[-1:] if years else [])
sel_fin_years = st.sidebar.multiselect("Financial Year", fin_years, default=[])
sel_states = st.sidebar.multiselect("State", states, default=[])
sel_makers = st.sidebar.multiselect("Maker", makers, default=[])
sel_fuels = st.sidebar.multiselect("Fuel", fuels, default=[])
sel_categories = st.sidebar.multiselect("Category", categories, default=[])
sel_vehicle_classes = st.sidebar.multiselect("Vehicle Class", vehicle_classes, default=[])
sel_sale_types = st.sidebar.multiselect("Sale Type", sale_types, default=[])
sel_poll = st.sidebar.multiselect("Pollution Norm", poll_norms, default=[])

search_text = st.sidebar.text_input("Search (Maker/Model/RTO contains)", value="").strip()

# Month exact + range
sel_months = st.sidebar.multiselect("Month (MM-YY) exact", months_list, default=[])

min_idx, max_idx = get_month_bounds(con_id, VIEW, COLS["registrationMonthMMYY"])
month_range = None
if min_idx is not None and max_idx is not None:
    month_range = st.sidebar.slider(
        "Month Range (MM-YY)",
        min_value=min_idx,
        max_value=max_idx,
        value=(min_idx, max_idx),
        format="%d",
    )
    st.sidebar.caption(f"Range: **{index_to_mmyy(month_range[0])} -> {index_to_mmyy(month_range[1])}**")

# RTO drill-down filter
rto_filter_enabled = st.sidebar.checkbox("Enable RTO filter", value=False)
sel_rtos = []
if rto_filter_enabled:
    if sel_states:
        state_filter = build_where_in(COLS["stateName"], sel_states)
        rtos = con.execute(f"""
            SELECT DISTINCT {COLS['rtoName']} AS rto
            FROM {VIEW}
            WHERE {state_filter}
              AND {COLS['rtoName']} IS NOT NULL
            ORDER BY 1
            LIMIT 50000
        """).df()["rto"].tolist()
    else:
        rtos = distinct_vals(con_id, VIEW, COLS["rtoName"])
    sel_rtos = st.sidebar.multiselect("RTO Name", rtos, default=[])

row_limit = st.sidebar.slider("Max rows to show", 100, 50000, 500, step=100)

# ---------------------------
# WHERE clause builder
# ---------------------------
where_parts = []
for col, vals in [
    (COLS["registrationYear"], sel_years),
    (COLS["financialYear"], sel_fin_years),
    (COLS["stateName"], sel_states),
    (COLS["makerName"], sel_makers),
    (COLS["fuelName"], sel_fuels),
    (COLS["vehicleCategoryName"], sel_categories),
    (COLS["vehicleClassName"], sel_vehicle_classes),
    (COLS["saleType"], sel_sale_types),
    (COLS["pollutionNorm"], sel_poll),
]:
    clause = build_where_in(col, vals)
    if clause:
        where_parts.append(clause)

if rto_filter_enabled and sel_rtos:
    clause = build_where_in(COLS["rtoName"], sel_rtos)
    if clause:
        where_parts.append(clause)

# Month filter: exact overrides range
if sel_months:
    clause = build_where_in(COLS["registrationMonthMMYY"], sel_months)
    if clause:
        where_parts.append(clause)
elif month_range is not None:
    mcol = COLS["registrationMonthMMYY"]
    where_parts.append(f"{month_index_expr(mcol)} BETWEEN {int(month_range[0])} AND {int(month_range[1])}")

if search_text:
    s = search_text.replace("'", "''")
    where_parts.append(
        f"({COLS['makerName']} ILIKE '%{s}%' "
        f"OR {COLS['vehicleModelName']} ILIKE '%{s}%' "
        f"OR {COLS['rtoName']} ILIKE '%{s}%' "
        f"OR {COLS['rtoCode']} ILIKE '%{s}%')"
    )

where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

# ---------------------------
# KPIs
# ---------------------------
kpi_q = f"""
    SELECT
      SUM(CAST({COLS['vehicleCount']} AS BIGINT)) AS total_vehicles,
      COUNT(*) AS rows
    FROM {VIEW}
    {where_sql}
"""
total_vehicles, rows = con.execute(kpi_q).fetchone()
total_vehicles = total_vehicles or 0
rows = rows or 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Vehicle Count", f"{total_vehicles:,}")
k2.metric("Matching Rows", f"{rows:,}")
k3.metric("Parquet Cache", Path(parquet_path).name)
k4.metric("Search", search_text if search_text else "-")

# ---------------------------
# Top-level charts
# ---------------------------
st.divider()
left, right = st.columns(2)

state_q = f"""
    SELECT {COLS['stateName']} AS state,
           SUM(CAST({COLS['vehicleCount']} AS BIGINT)) AS vehicle_count
    FROM {VIEW}
    {where_sql}
    GROUP BY 1
    ORDER BY vehicle_count DESC
    LIMIT 20
"""
maker_q = f"""
    SELECT {COLS['makerName']} AS maker,
           SUM(CAST({COLS['vehicleCount']} AS BIGINT)) AS vehicle_count
    FROM {VIEW}
    {where_sql}
    GROUP BY 1
    ORDER BY vehicle_count DESC
    LIMIT 20
"""
state_df = con.execute(state_q).df()
maker_df = con.execute(maker_q).df()

with left:
    st.subheader("Top States")
    st.dataframe(state_df, use_container_width=True, height=260)
    if not state_df.empty:
        st.plotly_chart(px.bar(state_df, x="state", y="vehicle_count"), use_container_width=True)

with right:
    st.subheader("Top Makers")
    st.dataframe(maker_df, use_container_width=True, height=260)
    if not maker_df.empty:
        st.plotly_chart(px.bar(maker_df, x="maker", y="vehicle_count"), use_container_width=True)

# ---------------------------
# Fuel + Category pie charts
# ---------------------------
st.divider()
c1, c2 = st.columns(2)

fuel_q = f"""
    SELECT COALESCE({COLS['fuelName']}, 'Unknown') AS fuel,
           SUM(CAST({COLS['vehicleCount']} AS BIGINT)) AS vehicle_count
    FROM {VIEW}
    {where_sql}
    GROUP BY 1
    ORDER BY vehicle_count DESC
    LIMIT 12
"""
cat_q = f"""
    SELECT COALESCE({COLS['vehicleCategoryName']}, 'Unknown') AS category,
           SUM(CAST({COLS['vehicleCount']} AS BIGINT)) AS vehicle_count
    FROM {VIEW}
    {where_sql}
    GROUP BY 1
    ORDER BY vehicle_count DESC
    LIMIT 12
"""
fuel_df = con.execute(fuel_q).df()
cat_df = con.execute(cat_q).df()

with c1:
    st.subheader("Fuel Mix")
    if not fuel_df.empty:
        st.plotly_chart(px.pie(fuel_df, names="fuel", values="vehicle_count"), use_container_width=True)
    st.dataframe(fuel_df, use_container_width=True, height=220)

with c2:
    st.subheader("Category Mix")
    if not cat_df.empty:
        st.plotly_chart(px.pie(cat_df, names="category", values="vehicle_count"), use_container_width=True)
    st.dataframe(cat_df, use_container_width=True, height=220)

# ---------------------------
# RTO drill-down
# ---------------------------
st.divider()
st.subheader("RTO Drill-down")

drill_cols = st.columns([1, 2])
with drill_cols[0]:
    drill_state = st.selectbox("Pick a State for RTO breakdown", options=["(All)"] + states, index=0)
with drill_cols[1]:
    st.caption("Select a state to see top RTOs under current filters.")

drill_where = where_parts.copy()
if drill_state != "(All)":
    drill_where = [w for w in drill_where if not w.startswith(COLS["stateName"])]
    safe_state = str(drill_state).replace("'", "''")
    drill_where.append(f"{COLS['stateName']} = '{safe_state}'")

drill_where_sql = ("WHERE " + " AND ".join(drill_where)) if drill_where else ""

rto_q = f"""
    SELECT
      COALESCE({COLS['rtoCode']}, '') AS rto_code,
      COALESCE({COLS['rtoName']}, 'Unknown') AS rto_name,
      SUM(CAST({COLS['vehicleCount']} AS BIGINT)) AS vehicle_count
    FROM {VIEW}
    {drill_where_sql}
    GROUP BY 1,2
    ORDER BY vehicle_count DESC
    LIMIT 30
"""
rto_df = con.execute(rto_q).df()

r1, r2 = st.columns(2)
with r1:
    st.dataframe(rto_df, use_container_width=True, height=420)
with r2:
    if not rto_df.empty:
        st.plotly_chart(px.bar(rto_df, x="rto_name", y="vehicle_count"), use_container_width=True)

# ---------------------------
# Sample table: grouped only by normalizedModel
# ---------------------------
st.divider()
st.subheader("Filtered Data (Normalized Model buckets)")

n_expr = normalized_model_expr(COLS["vehicleModelName"])

sample_q = f"""
    SELECT
      NULLIF(trim({n_expr}), '') AS normalizedModel,
      SUM(CAST({COLS['vehicleCount']} AS BIGINT)) AS registrations,
      COUNT(*) AS rows_in_bucket
    FROM {VIEW}
    {where_sql}
    GROUP BY 1
    HAVING normalizedModel IS NOT NULL
    ORDER BY registrations DESC
    LIMIT {int(row_limit)}
"""
sample_df = con.execute(sample_q).df()
st.dataframe(sample_df, use_container_width=True, height=520)

st.download_button(
    "Download normalized-model table (CSV)",
    data=sample_df.to_csv(index=False).encode("utf-8"),
    file_name="normalized_model_buckets.csv",
    mime="text/csv",
)

# ---------------------------
# Detail table: normalized model aggregated view
# ---------------------------
st.divider()
st.subheader("Filtered Data (Normalized Model - aggregated view)")

norm_expr = normalized_model_expr(COLS["vehicleModelName"])

detail_q = f"""
SELECT
  NULLIF(trim({norm_expr}), '') AS normalizedModel,

  ANY_VALUE({COLS['makerName']}) AS makerName,
  ANY_VALUE({COLS['vehicleCategoryName']}) AS vehicleCategoryName,
  ANY_VALUE({COLS['vehicleClassName']}) AS vehicleClassName,
  ANY_VALUE({COLS['fuelName']}) AS fuelName,
  ANY_VALUE({COLS['saleType']}) AS saleType,
  ANY_VALUE({COLS['pollutionNorm']}) AS pollutionNorm,

  SUM(CAST({COLS['vehicleCount']} AS BIGINT)) AS totalRegistrations,
  COUNT(*) AS rowCount,

  MIN({COLS['registrationMonthMMYY']}) AS firstMonth,
  MAX({COLS['registrationMonthMMYY']}) AS lastMonth

FROM {VIEW}
{where_sql}
GROUP BY 1
HAVING normalizedModel IS NOT NULL
ORDER BY totalRegistrations DESC
LIMIT {int(row_limit)}
"""
detail_df = con.execute(detail_q).df()

st.dataframe(detail_df, use_container_width=True, height=520)

st.download_button(
    "Download normalized model (aggregated) CSV",
    data=detail_df.to_csv(index=False).encode("utf-8"),
    file_name="normalized_model_aggregated.csv",
    mime="text/csv",
)

# ==========================================================
# Maker + Category + Class + optional Normalized Model -> Monthly + Pivot + YoY
# ==========================================================
st.divider()
st.subheader("Monthly Registrations (Maker / Category / Class / Optional Normalized Model)")

r1, r2, r3, r4 = st.columns([1.2, 1.4, 1.6, 0.8])
with r1:
    mm_maker = st.selectbox("Maker", options=["(Select)"] + makers, index=0)

with r2:
    mm_category = st.multiselect("Vehicle Category (optional)", options=categories, default=[])

with r3:
    mm_class = st.multiselect("Vehicle Class (optional)", options=vehicle_classes, default=[])

with r4:
    show_yoy = st.checkbox("YoY", value=True)

s1, s2 = st.columns([2.4, 1])
with s1:
    model_query = st.text_input(
        "Search Normalized Model (optional; contains)",
        value="",
        placeholder="Type model text e.g. NEXON, ACE, PRIMA...",
    )
with s2:
    max_results = st.selectbox("Results limit", options=[50, 100, 200, 500], index=1)

if mm_maker == "(Select)":
    st.info("Select a Maker to see monthly registrations.")
else:
    mm_where = where_parts.copy()

    safe_maker = str(mm_maker).replace("'", "''")
    mm_where.append(f"{COLS['makerName']} = '{safe_maker}'")

    if mm_category:
        mm_where.append(build_where_in(COLS["vehicleCategoryName"], mm_category))
    if mm_class:
        mm_where.append(build_where_in(COLS["vehicleClassName"], mm_class))

    n_expr = normalized_model_expr(COLS["vehicleModelName"])

    selected_norm = None
    if model_query.strip():
        qtext = model_query.strip().replace("'", "''")
        extra_where_sql = ("WHERE " + " AND ".join([w for w in mm_where if w])) if mm_where else ""
        search_sql = f"""
            SELECT
              NULLIF(trim({n_expr}), '') AS normalizedModel,
              SUM(CAST({COLS['vehicleCount']} AS BIGINT)) AS registrations
            FROM {VIEW}
            {extra_where_sql}
              AND {n_expr} ILIKE '%{qtext}%'
            GROUP BY 1
            HAVING normalizedModel IS NOT NULL
            ORDER BY registrations DESC, normalizedModel ASC
            LIMIT {int(max_results)}
        """
        matches_df = con.execute(search_sql).df()

        if matches_df.empty:
            st.warning("No normalized model buckets found for current Maker/Category/Class + search text.")
        else:
            st.write("### Matching Normalized Models (sorted by registrations)")
            st.dataframe(matches_df, use_container_width=True, height=240)

            selected_norm = st.selectbox(
                "Select a normalized model bucket (optional)",
                options=["(All matching data)"] + matches_df["normalizedModel"].tolist(),
                index=0,
            )
            if selected_norm == "(All matching data)":
                selected_norm = None

    if selected_norm:
        safe_norm = str(selected_norm).replace("'", "''")
        mm_where.append(f"{n_expr} = '{safe_norm}'")

    mm_where_sql = ("WHERE " + " AND ".join([w for w in mm_where if w])) if mm_where else ""

    month_col = COLS["registrationMonthMMYY"]
    midx = month_index_expr(month_col)

    monthly_q = f"""
        SELECT
          {COLS['makerName']} AS maker,
          COALESCE({COLS['vehicleCategoryName']}, 'Unknown') AS vehicleCategory,
          COALESCE({COLS['vehicleClassName']}, 'Unknown') AS vehicleClass,
          {month_col} AS month,
          {midx} AS month_idx,
          SUM(CAST({COLS['vehicleCount']} AS BIGINT)) AS registrations
        FROM {VIEW}
        {mm_where_sql}
        GROUP BY 1,2,3,4,5
        HAVING month_idx IS NOT NULL
        ORDER BY month_idx
    """
    monthly_df = con.execute(monthly_q).df()

    if monthly_df.empty:
        st.info("No monthly data for this selection under current filters.")
    else:
        st.write("### Monthly Registrations (table)")
        st.dataframe(monthly_df.drop(columns=["month_idx"]), use_container_width=True, height=300)

        monthly_df["month_label"] = monthly_df["month"].astype(str)
        month_order = (
            monthly_df[["month_label", "month_idx"]]
            .dropna()
            .drop_duplicates()
            .sort_values("month_idx")
        )["month_label"].tolist()

        pivot = monthly_df.pivot_table(
            index=["maker", "vehicleCategory", "vehicleClass"],
            columns="month_label",
            values="registrations",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()

        ordered_cols = ["maker", "vehicleCategory", "vehicleClass"] + [m for m in month_order if m in pivot.columns]
        pivot = pivot[ordered_cols]

        st.write("### Monthly Pivot (months as columns)")
        st.dataframe(pivot, use_container_width=True, height=260)

        if show_yoy:
            yoy_base = (
                monthly_df.groupby(["month_label", "month_idx"], as_index=False)["registrations"].sum()
                .sort_values("month_idx")
            )
            yoy_base["prev_year_regs"] = yoy_base["registrations"].shift(12)

            yoy_base["yoy_pct"] = None
            mask = yoy_base["prev_year_regs"].fillna(0) != 0
            yoy_base.loc[mask, "yoy_pct"] = (
                (yoy_base.loc[mask, "registrations"] - yoy_base.loc[mask, "prev_year_regs"])
                / yoy_base.loc[mask, "prev_year_regs"]
                * 100.0
            )

            st.write("### YoY (total across selected filters)")
            st.dataframe(
                yoy_base.rename(columns={"month_label": "month"})[
                    ["month", "registrations", "prev_year_regs", "yoy_pct"]
                ],
                use_container_width=True,
                height=240,
            )

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download Monthly Table (CSV)",
                data=monthly_df.drop(columns=["month_idx"]).to_csv(index=False).encode("utf-8"),
                file_name="monthly_regs_table.csv",
                mime="text/csv",
            )
        with dl2:
            st.download_button(
                "Download Pivot (CSV)",
                data=pivot.to_csv(index=False).encode("utf-8"),
                file_name="monthly_regs_pivot.csv",
                mime="text/csv",
            )

# ---------------------------
# Detail table
# ---------------------------
st.subheader("Filtered Data (sample)")
detail_q = f"""
    SELECT *
    FROM vdata
    {where_sql}
    LIMIT {int(row_limit)}
"""
detail_df = con.execute(detail_q).df()
st.dataframe(detail_df, use_container_width=True, height=500)

# ---------------------------
# Download filtered output
# ---------------------------
st.download_button(
    "Download filtered CSV",
    data=detail_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_data.csv",
    mime="text/csv",
)
