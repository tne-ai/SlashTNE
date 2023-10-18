import re
from typing import Union, List, Dict
from pydantic import BaseModel

class TableColumn(BaseModel):
    """Table column."""
    name: str
    dtype: Union[str, None]

class ForeignKey(BaseModel):
    """Foreign key."""
    # Referenced column
    column: TableColumn
    # References table name
    references_name: str
    # References column
    references_column: TableColumn

class Table(BaseModel):
    """Table."""
    name: str
    columns: Union[List[TableColumn], None]
    pks: Union[List[TableColumn], None]
    # FK from this table to another column in another table
    fks: Union[List[ForeignKey], None]

class RajkumarFormatter:
    """RajkumarFormatter class.
    From https://arxiv.org/pdf/2204.00498.pdf.
    """
    table_sep: str = "\n\n"

    def __init__(self, tables: List[Table], data: List[str], unique_vals: List[Dict]) -> None:
        self.tables = tables
        self.table_str = self.format_tables(tables)
        self.data_str = self.format_data(data)
        self.unique_vals_str = self.format_unique_values(unique_vals)

    def format_unique_values(self, unique_values_dict: dict) -> str:
        """Format the distinct values for each table in a readable manner."""
        formatted_tables = []

        for table_name, unique_values in unique_values_dict.items():
            formatted_values = []
            for column, values in unique_values.items():
                values_str = ", ".join(map(str, values))
                formatted_values.append(f"{column}: {values_str}")

            formatted_table = "\n".join(formatted_values)
            formatted_tables.append(formatted_table)

        return "\n\n".join(formatted_tables)

    def format_data(self, data: str) -> str:
        data_str = ""
        for i, table_data in enumerate(data):
            table_str = ""
            for n, row in enumerate(table_data):
                if n < len(table_data)-1:
                    table_str += row + '\n'
                else:
                    table_str += row

            if i < len(data)-1:
                data_str += table_str + '\n\n'
            else:
                data_str += table_str

        return data_str

    def format_table(self, table: Table) -> str:
        """Get table format."""
        table_fmt = []
        table_name = table.name
        for col in table.columns or []:
            # This is technically an incorrect type, but it should be a catchall word
            table_fmt.append(f"    {col.name} {col.dtype or 'any'}")
        if table.pks:
            table_fmt.append(
                f"    primary key ({', '.join(pk.name for pk in table.pks)})"
            )
        for fk in table.fks or []:
            table_fmt.append(
                f"    foreign key ({fk.column.name}) references {fk.references_name}({fk.references_column.name})"  # noqa: E501
            )
        if table_fmt:
            all_cols = ",\n".join(table_fmt)
            create_tbl = f"CREATE TABLE {table_name} (\n{all_cols}\n)"
        else:
            create_tbl = f"CREATE TABLE {table_name}"
        return create_tbl

    def format_tables(self, tables: List[Table]) -> str:
        """Get tables format."""
        return self.table_sep.join(self.format_table(table) for table in tables)

    def pull_schema(self) -> dict:
        """Get formatted schema string."""

        db_data = f"SCHEMA\n{self.table_str}\nSAMPLE CONTENTS\n{self.data_str}\nUNIQUE VALUES\n{self.unique_vals_str}"

        return db_data