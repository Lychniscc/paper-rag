
# database.py
# SQLite问答历史记录模块

import json
import sqlite3
from pathlib import Path

# 数据库存储位置
DB_PATH = Path("data") / "rag_history.db"


def get_connection():
    """建立SQLite数据库连接。"""

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(
        DB_PATH,
        timeout=10
    )

    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """首次启动时自动创建问答记录表。"""

    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS qa_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                model TEXT,
                sources TEXT,
                total_seconds REAL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                created_at TEXT DEFAULT
                    (datetime('now'))
            )
        """)

        conn.commit()


def save_qa(
    question,
    answer,
    model,
    sources,
    total_seconds,
    input_tokens,
    output_tokens
):
    """将一次成功的RAG回答保存到数据库。"""

    with get_connection() as conn:

        cursor = conn.execute(
            """
            INSERT INTO qa_history (
                question,
                answer,
                model,
                sources,
                total_seconds,
                input_tokens,
                output_tokens
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                question,
                answer,
                model,
                json.dumps(
                    sources,
                    ensure_ascii=False
                ),
                total_seconds,
                input_tokens,
                output_tokens
            )
        )

        conn.commit()

        return cursor.lastrowid


def get_history(limit=20):
    """查询最近的问答历史。"""

    with get_connection() as conn:

        rows = conn.execute(
            """
            SELECT
                id,
                question,
                answer,
                model,
                total_seconds,
                created_at
            FROM qa_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]


def get_history_detail(record_id):
    """根据ID查询一条完整问答记录。"""

    with get_connection() as conn:

        row = conn.execute(
            """
            SELECT *
            FROM qa_history
            WHERE id = ?
            """,
            (record_id,)
        ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["sources"] = json.loads(
            result["sources"] or "[]"
        )

        return result


def delete_history(record_id):
    """删除指定的问答记录。"""

    with get_connection() as conn:

        cursor = conn.execute(
            """
            DELETE FROM qa_history
            WHERE id = ?
            """,
            (record_id,)
        )

        conn.commit()
        return cursor.rowcount > 0


if __name__ == "__main__":

    init_db()

    print("SQLite数据库初始化成功！")
    print("数据库位置：", DB_PATH.resolve())
