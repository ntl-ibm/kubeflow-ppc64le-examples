import ibm_db
import json
import os
from typing import List, Dict, Union, Generator, Any
import logging
import psycopg

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

        iSql = f'SELECT ACCOUNT_ID FROM FINAL TABLE (INSERT INTO "{self.sql_safe_clint_info_table_name}" ({iStmtColsSql}) VALUES({iValues}))'

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

        if "Risk" in changes and changes["Risk"] == "Unknown":
            changes["Risk"] = None

        current_data = self.get_account_info(client_id)

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

    def get_number_of_accounts(self) -> int:
        query = f"SELECT COUNT(*) FROM {self.client_info_table_name}"
        logging.debug(f"preparing statement {query}")
        stmt = ibm_db.prepare(self.conn, query)
        ibm_db.execute(stmt)

        row = ibm_db.fetch_tuple(stmt)
        return row[0]

    def get_accounts(
        self, offset: int = 0, limit: int = 25
    ) -> Generator[Dict[str, Any], None, None]:
        query = f'SELECT ACCOUNT_ID, "Risk", "PredictedRisk" FROM {self.client_info_table_name} ORDER BY CLIENT_ID ASC LIMIT ? OFFSET ?'
        logging.debug(f"preparing statement {query}")
        stmt = ibm_db.prepare(self.conn, query)
        ibm_db.execute(stmt, (limit, offset))

        row = ibm_db.fetch_assoc(stmt)
        while row:
            yield row
            row = ibm_db.fetch_assoc(stmt)

    def get_account_info(self, account_id: int) -> Dict[str, Union[str, int]]:
        json_obj_kv = lambda col_name: f"'{col_name}' VALUE " + f'"{col_name}"'
        key_values = ",".join([json_obj_kv(col_name) for col_name in self.column_names])
        sql_stmt_txt = f'SELECT JSON_OBJECT({key_values}) FROM "{self.sql_safe_clint_info_table_name}" WHERE ACCOUNT_ID = ?'
        logger.debug(f"Preparing statement {sql_stmt_txt}")
        stmt = ibm_db.prepare(self.conn, sql_stmt_txt)

        ibm_db.execute(stmt, tuple([account_id]))

        row = ibm_db.fetch_tuple(stmt)
        return json.loads(row[0])


def _create_postgresql_conn() -> psycopg.Connection:
    host, dbname, username, password, port = (
        os.environ.get("PG_HOST"),
        os.environ.get("PG_DB_NAME"),
        os.environ.get("PG_USER"),
        os.environ.get("PG_PWD"),
        int(os.environ.get("PG_PORT")),
    )

    conn_str = f"postgresql://{username}:{password}@{host}:{port}/{dbname}?application_name=credit_risk_app"
    conn = psycopg.connect(conn_str)

    return conn


class PostgreSQLConnection:
    """
    Connection to a PostgreSQL Database
    """

    def __init__(self, client_info_table_name: str = "CLIENT_DATA"):
        self.conn: psycopg.Connection = _create_postgresql_conn()
        self.client_info_table_name = client_info_table_name
        self.column_names = self._get_column_names()

    def __enter__(self):
        return self

    def __exit__(
        self, exception_type=None, exception_value=None, exception_traceback=None
    ):
        del exception_value
        del exception_traceback

        if exception_type:
            self.conn.rollback()
        else:
            self.conn.commit()

        self.conn.close()

    def close(self):
        self.__exit__()

    def insert_account_from_row_dict(
        self,
        row_dict: Dict[str, Union[str, int]],
    ) -> int:
        row_cols = list(row_dict.keys())

        insert = psycopg.sql.SQL(
            'INSERT INTO {0} ({1}) VALUES({2}) RETURNING "ACCOUNT_ID"'
        ).format(
            psycopg.sql.Identifier("CLIENT_DATA"),
            psycopg.sql.SQL(", ").join(
                [psycopg.sql.Identifier(col) for col in row_cols]
            ),
            psycopg.sql.SQL(", ").join([psycopg.sql.SQL("%s")] * len(row_cols)),
        )

        with self.conn.cursor() as cur:
            print(insert.as_string(cur))
            cur.execute(insert, tuple([row_dict[col] for col in row_cols]))
            return cur.fetchone()[0]

    def update_account_from_row_change_dict(
        self,
        account_id: int,
        changes: Dict[str, Union[str, int]],
    ):
        if "Risk" in changes and changes["Risk"] == "Unknown":
            changes["Risk"] = None

        current_data = self.get_account_info(account_id)

        changes = {
            col: value
            for col, value in changes.items()
            if col in current_data and current_data[col] != value
        }

        assert "ACCOUNT_ID" not in changes

        cols = list(changes.keys())
        update = psycopg.sql.SQL('UPDATE {0} SET {1} WHERE "ACCOUNT_ID" = %s').format(
            psycopg.sql.Identifier(self.client_info_table_name),
            psycopg.sql.SQL(", ").join(
                [
                    psycopg.sql.SQL("{0} = %s").format(psycopg.sql.Identifier(col))
                    for col in cols
                ]
            ),
        )

        with self.conn.cursor() as cur:
            cur.execute(update, tuple([changes[col] for col in cols] + [account_id]))

    def _get_column_names(
        self,
    ):  # (table_name: str, db: psycopg.Connection) -> List[str]:
        with self.conn.cursor() as cur:
            cur.execute(
                psycopg.sql.SQL(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = {} AND table_schema = CURRENT_SCHEMA ORDER BY ORDINAL_POSITION ASC"
                ).format(psycopg.sql.Literal(self.client_info_table_name))
            )
            return [row[0] for row in cur.fetchall()]

    def get_number_of_accounts(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                psycopg.sql.SQL("SELECT COUNT(*) FROM {} ").format(
                    psycopg.sql.Identifier(self.client_info_table_name)
                )
            )
            return cur.fetchone()[0]

    def get_accounts(
        self, offset: int = 0, limit: int = 25
    ) -> Generator[Dict[str, Any], None, None]:
        query = psycopg.sql.SQL(
            'SELECT "ACCOUNT_ID", "Risk", "PredictedRisk" FROM {} ORDER BY "ACCOUNT_ID" ASC LIMIT %s OFFSET %s'
        ).format(psycopg.sql.Identifier(self.client_info_table_name))
        with self.conn.cursor() as cur:
            cur.execute(query, (limit, offset))

            columns = [c[0] for c in cur.description]
            row = cur.fetchone()
            while row:
                yield {columns[i]: row[i] for i in range(len(columns))}
                row = cur.fetchone()

    def get_account_info(self, account_id: int) -> Dict[str, Union[str, int]]:
        with self.conn.cursor() as cur:
            cur.execute(
                psycopg.sql.SQL(
                    'SELECT json_agg(t) FROM {} AS t WHERE "ACCOUNT_ID" = %s'
                ).format(psycopg.sql.Identifier(self.client_info_table_name)),
                (account_id,),
            )
            r = cur.fetchone()
        return r[0][0] if r and r[0] else {}


if os.environ.get("PG_HOST"):
    DBConnection = PostgreSQLConnection
else:
    DBConnection = DB2DataBaseConnection
