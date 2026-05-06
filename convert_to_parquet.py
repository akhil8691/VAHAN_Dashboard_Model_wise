from pathlib import Path
import duckdb

csv_path = r"C:\Users\Akhil-EUR0750\OneDrive\OneDrive - Euler Motors Pvt Ltd\Desktop\tata-motors-2026-01-05.csv"
parquet_path = "vahan_data.parquet"

csv_sql = csv_path.replace("'", "''")
parquet_sql = parquet_path.replace("'", "''")

duckdb.sql(f"""
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
    (FORMAT PARQUET, COMPRESSION ZSTD);
""")

print(f"Created: {Path(parquet_path).resolve()}")