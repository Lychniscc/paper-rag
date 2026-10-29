
# tests/test_database.py
# SQLite数据库自动化单元测试

import pytest
import database


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """
    每项测试使用独立的临时数据库。
    不会修改真正的rag_history.db。
    """

    db_path = tmp_path / "test_history.db"

    monkeypatch.setattr(
        database,
        "DB_PATH",
        db_path
    )

    database.init_db()

    return db_path


def test_save_qa(test_db):
    """测试1：保存问答记录。"""

    record_id = database.save_qa(
        question="什么是DOA？",
        answer="DOA是波达方向。",
        model="test-model",
        sources=[{"chunk_id": 1, "page": 2}],
        total_seconds=1.5,
        input_tokens=100,
        output_tokens=50
    )

    assert record_id > 0

    record = database.get_history_detail(record_id)

    assert record is not None
    assert record["question"] == "什么是DOA？"
    assert record["answer"] == "DOA是波达方向。"


def test_get_history(test_db):
    """测试2：查询历史列表。"""

    database.save_qa(
        "问题1",
        "答案1",
        "test-model",
        [],
        1.0,
        10,
        10
    )

    records = database.get_history(limit=10)

    assert len(records) == 1
    assert records[0]["question"] == "问题1"


def test_sources_json(test_db):
    """测试3：来源信息能否正确保存与恢复。"""

    record_id = database.save_qa(
        "论文问题",
        "论文答案",
        "test-model",
        [{"chunk_id": 60, "page": 7}],
        2.5,
        200,
        100
    )

    record = database.get_history_detail(record_id)

    assert record["sources"][0]["chunk_id"] == 60
    assert record["sources"][0]["page"] == 7


def test_delete_history(test_db):
    """测试4：删除记录。"""

    record_id = database.save_qa(
        "测试问题",
        "测试答案",
        "test-model",
        [],
        1.0,
        10,
        10
    )

    deleted = database.delete_history(record_id)

    assert deleted is True

    record = database.get_history_detail(record_id)

    assert record is None


def test_missing_record(test_db):
    """测试5：查询不存在的记录。"""

    record = database.get_history_detail(999999)

    assert record is None


def test_delete_missing_record(test_db):
    """测试6：删除不存在的记录。"""

    deleted = database.delete_history(999999)

    assert deleted is False
