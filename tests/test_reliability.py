
# tests/test_reliability.py
# 测试RAG回答可靠性检查

from rag_reliability_v522 import validate_answer


# 模拟检索到的论文片段
SOURCES = [
    {"chunk_id": 60, "page": 7},
    {"chunk_id": 105, "page": 13}
]


def test_valid_answer():
    """正常回答应通过检查。"""

    answer = "本文采用PARAFAC方法。[资料60]"

    result = validate_answer(answer, SOURCES)

    assert result["passed"] is True


def test_empty_answer():
    """空回答应被拒绝。"""

    result = validate_answer("", SOURCES)

    assert result["passed"] is False


def test_invalid_citation():
    """不存在的引用编号应被识别。"""

    answer = "本文采用PARAFAC方法。[资料999]"

    result = validate_answer(answer, SOURCES)

    assert result["passed"] is False
    assert "999" in result["invalid_ids"]


def test_missing_citation():
    """缺少资料引用应被拒绝。"""

    answer = "本文采用PARAFAC方法。"

    result = validate_answer(answer, SOURCES)

    assert result["passed"] is False


def test_internal_analysis():
    """模拟Q9曾经出现的英文分析文本。"""

    answer = (
        "The user is asking about this paper. "
        "本文采用PARAFAC方法。[资料60]"
    )

    result = validate_answer(answer, SOURCES)

    assert result["passed"] is False
