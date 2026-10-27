
import csv
from pathlib import Path

# 项目评测文件夹
folder = Path("evaluation_results")

# 原始未评分文件
source = folder / "retrieval_labels.csv"

# 生成新的已评分文件，不覆盖原件
output = folder / "retrieval_labels_scored.csv"

# 已完成的人工相关性评分
# 每个问题编号下：文本块编号 -> 相关性分数
SCORES = {
    "1": {"17": 1, "105": 2, "96": 0, "15": 1, "13": 1, "18": 1, "46": 1, "72": 2},
    "2": {"62": 0, "104": 2, "21": 1, "82": 1, "25": 2, "105": 2, "35": 0, "2": 1, "15": 1},
    "3": {"44": 2, "43": 2, "82": 1, "60": 2, "24": 1, "105": 1},
    "4": {"68": 0, "95": 0, "46": 1, "64": 1, "17": 0, "18": 0, "86": 0, "60": 2, "66": 0, "61": 2, "63": 1},
    "5": {"72": 1, "60": 1, "87": 0, "85": 0, "46": 1, "53": 1, "64": 2, "49": 1, "61": 1},
    "6": {"67": 2, "25": 1, "23": 1, "2": 1, "1": 1, "8": 1, "105": 1},
    "7": {"39": 2, "105": 1, "17": 0, "67": 0, "46": 0, "20": 2, "36": 2, "11": 0},
    "8": {"13": 1, "11": 1, "23": 1, "25": 2, "86": 1, "106": 2, "22": 1, "14": 1, "26": 2},
    "9": {"87": 2, "100": 2, "94": 2, "105": 0, "72": 0, "77": 1, "101": 2, "23": 0},
    "10": {"87": 0, "100": 0, "4": 0, "94": 0, "96": 0, "104": 2, "36": 0, "105": 1},
}

# 检查原始文件
if not source.exists():
    raise FileNotFoundError(
        f"找不到原始文件：{source.resolve()}"
    )

# 读取全部检索结果
with source.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

# 检查必要字段
required = {"question_id", "chunk_id", "relevance"}

if not required.issubset(set(fieldnames or [])):
    raise ValueError("CSV缺少必要字段")

# 逐行填写评分
missing = []

for row in rows:
    question_id = row["question_id"].strip()
    chunk_id = row["chunk_id"].strip()

    score = SCORES.get(question_id, {}).get(chunk_id)

    if score is None:
        missing.append((question_id, chunk_id))
    else:
        row["relevance"] = str(score)

# 如果存在无法匹配的文本块，停止保存
if missing:
    raise ValueError(
        f"有{len(missing)}条评分无法匹配：{missing[:10]}"
    )

# 保存已经评分的CSV
with output.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("评分填写完成！")
print(f"记录总数：{len(rows)}")
print(f"输出文件：{output.resolve()}")
print("原始CSV未被覆盖")
