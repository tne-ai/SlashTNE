from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Generator, List
import pandas as pd
import sqlalchemy
from sqlalchemy import MetaData, Table, Column, String, text

from prompt_formatters import TableColumn, Table as FormatterTable

@dataclass
class PostgresConnector:
    """Postgres connection."""

    user: str
    password: str
    dbname: str
    host: str
    port: int

    @cached_property
    def pg_uri(self) -> str:
        """Get Postgres URI."""
        uri = (
            f"postgresql://"
            f"{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
        )
        engine = sqlalchemy.create_engine(uri)
        conn = engine.connect()

        # assuming the above connection is successful, we can now close the connection
        conn.close()
        engine.dispose()

        return uri

    @contextmanager
    def connect(self) -> Generator[sqlalchemy.engine.base.Connection, None, None]:
        """Yield a connection to a Postgres db."""
        try:
            engine = sqlalchemy.create_engine(self.pg_uri)
            conn = engine.connect()
            yield conn
        finally:
            conn.close()
            engine.dispose()

    def run_sql_as_df(self, sql: str) -> pd.DataFrame:
        """Run SQL statement."""
        with self.connect() as conn:
            return pd.read_sql(sql, conn)

    def get_tables(self) -> List[str]:
        """Get all tables in the database."""
        engine = sqlalchemy.create_engine(self.pg_uri)
        metadata = MetaData()
        metadata.reflect(bind=engine)
        table_names = metadata.tables.keys()
        engine.dispose()
        return table_names

    def select_three(self, table: str) -> FormatterTable:
        with self.connect() as conn:
            rows = []
            sql = f"""
                SELECT * FROM {table}
                ORDER BY RANDOM()
                LIMIT 3;
            """
            db_rows = conn.execute(text(sql)).fetchall()
            for row in db_rows:
                rows.append(' '.join(row))

        return rows

    def get_distinct_values(self, table: str) -> dict:
        """Get up to 25 distinct values for each column of a table."""
        distinct_values = {}

        with self.connect() as conn:
            for column in self.get_schema(table).columns:
                sql = f"""
                    SELECT DISTINCT {column.name} FROM {table}
                    LIMIT 25;
                """
                values = conn.execute(text(sql)).fetchall()
                distinct_values[column.name] = [value[0] for value in values]

        return distinct_values

    def get_schema(self, table: str) -> FormatterTable:
        """Return Table."""
        with self.connect() as conn:
            columns = []
            sql = f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table}';
            """
            schema = conn.execute(text(sql)).fetchall()
            for col, type_ in schema:
                columns.append(TableColumn(name=col, dtype=type_))
            return FormatterTable(name=table, columns=columns)

    def insert_sql_logs(self, table_name: str, data: dict):
        """Insert a key/value pair."""

        # Check for expected keys
        expected_keys = ['input_text', 'output_sql', 'db_name']
        if not all(key in expected_keys for key in data.keys()):
            raise ValueError(f"Invalid keys in data. Expected {', '.join(expected_keys)}.")

        columns_str = ', '.join(data.keys())
        placeholders = ', '.join([f":{key}" for key in data.keys()])

        # Check if table exists
        with self.connect() as conn:
            result = conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = '{table_name}'
                );
            """)).fetchone()
            table_exists = result[0]
            if not table_exists:
                conn.execute(text(f"""
                    CREATE TABLE {table_name} (
                        input_text TEXT NOT NULL,
                        output_sql TEXT NOT NULL,
                        db_name TEXT NOT NULL
                    );
                """))

            query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            conn.execute(text(query), data)
