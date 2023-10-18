import ibm_db
import json
import os
from typing import List, Dict, Union, Generator
import logging

logger = logging.getLogger(__name__)


def _create_db2_conn() -> ibm_db.IBM_DBConnection:
    conn_str = (
        "DRIVER={IBM DB2 ODBC DRIVER};"
        f"DATABASE=BLUDB;HOSTNAME={os.environ['db2_host']};PORT={os.environ['db2_port']};PROTOCOL=TCPIP;UID={os.environ['db2_user']};Pwd={os.environ['db2_pwd']};SECURITY=SSL;"
    )

    conn = ibm_db.pconnect(conn_str, "", "")

    return conn


class DB2DataBaseConnection:
    """
    Connection to a DB2 Database
    """

    def __init__(self, client_info_table_name: str = "CLIENT_DATA"):
        self.conn: ibm_db.IBM_DBConnection = _create_db2_conn()
        self.client_info_table_name = client_info_table_name
        self.column_names = self._get_column_names()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        del exception_type
        del exception_value
        del exception_traceback

        ibm_db.close(self.conn)

    def close(self):
        self.__exit__()

    @property
    def sql_safe_clint_info_table_name(self) -> str:
        return self.client_info_table_name.replace('"', "")

    def insert_account_from_row_dict(
        self,
        row_dict: Dict[str, Union[str, int]],
    ) -> int:
        row_cols = set(row_dict.keys())
        iStmtColsSql = ",".join([f'"{col}"' for col in row_cols])
        iValues = ",".join(["?" for _ in range(len(row_cols))])

        iSql = f'SELECT CLIENT_ID FROM FINAL TABLE (INSERT INTO "{self.sql_safe_clint_info_table_name}" ({iStmtColsSql}) VALUES({iValues}))'

        print(f"preparing statement {iSql}")
        stmt = ibm_db.prepare(self.conn, iSql)

        for idx, col in enumerate(row_cols):
            ibm_db.bind_param(stmt, idx + 1, row_dict[col])
        ibm_db.execute(stmt)

        client_id = ibm_db.fetch_tuple(stmt)[0]
        logging.debug(f"Created client with ID {client_id}")
        return client_id

    def update_account_from_row_change_dict(
        self,
        client_id: int,
        changes: Dict[str, Union[str, int]],
    ):
        stmt = None

        current_data = self.get_client_info(client_id)
        changes = {
            col: value
            for col, value in changes.items()
            if col in current_data and current_data[col] != value
        }

        assert "CLIENT_ID" not in changes

        cols = list(changes.keys())
        col_assign = ", ".join([f'"{col}" = ?' for col in cols])
        iSql = f'UPDATE "{self.client_info_table_name}" SET {col_assign} WHERE CLIENT_ID = ?'

        logging.debug(f"preparing statement {iSql}")
        stmt = ibm_db.prepare(self.conn, iSql)

        for idx, col in enumerate(cols):
            ibm_db.bind_param(stmt, idx + 1, changes[col])
        ibm_db.bind_param(stmt, len(cols) + 1, client_id)
        ibm_db.execute(stmt)

    def _get_column_names(self) -> List[str]:
        get_column_names_sql = f"SELECT COLUMN_NAME FROM SYSIBM.COLUMNS WHERE TABLE_NAME = ? ORDER BY ORDINAL_POSITION"
        stmt = ibm_db.prepare(self.conn, get_column_names_sql)
        ibm_db.execute(stmt, tuple([self.client_info_table_name]))

        col_names = []
        row = ibm_db.fetch_tuple(stmt)
        while row:
            col_names.append(row[0])
            row = ibm_db.fetch_tuple(stmt)

        return col_names

    def get_number_of_clients(self) -> int:
        query = f"SELECT COUNT(*) FROM {self.client_info_table_name}"
        logging.debug(f"preparing statement {query}")
        stmt = ibm_db.prepare(self.conn, query)
        ibm_db.execute(stmt)

        row = ibm_db.fetch_tuple(stmt)
        return row[0]

    def get_clients(
        self, offset: int = 0, limit: int = 25
    ) -> Generator[int, None, None]:
        query = f"SELECT CLIENT_ID FROM {self.client_info_table_name} ORDER BY CLIENT_ID ASC LIMIT ? OFFSET ?"
        logging.debug(f"preparing statement {query}")
        stmt = ibm_db.prepare(self.conn, query)
        ibm_db.execute(stmt, (limit, offset))

        row = ibm_db.fetch_tuple(stmt)
        while row:
            yield row[0]
            row = ibm_db.fetch_tuple(stmt)

    def get_client_info(self, client_id: int) -> Dict[str, Union[str, int]]:
        json_obj_kv = lambda col_name: f"'{col_name}' VALUE " + f'"{col_name}"'
        key_values = ",".join([json_obj_kv(col_name) for col_name in self.column_names])
        sql_stmt_txt = f'SELECT JSON_OBJECT({key_values}) FROM "{self.sql_safe_clint_info_table_name}" WHERE CLIENT_ID = ?'
        logger.debug(f"Preparing statement {sql_stmt_txt}")
        stmt = ibm_db.prepare(self.conn, sql_stmt_txt)

        ibm_db.execute(stmt, tuple([client_id]))

        row = ibm_db.fetch_tuple(stmt)
        return json.loads(row[0])
