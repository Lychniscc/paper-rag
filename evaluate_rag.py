
# =====================================================
# evaluate_rag.py
# 阶段5.1.5：自动检索评测与结果统计
#
# 使用方法：
# python evaluate_rag.py collect
# python evaluate_rag.py score
#
# 不调用Claude API
# =====================================================

import csv
import json
import sys
import statistics
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

# FastAPI检索诊断接口
API_URL = "http://127.0.0.1:8000/debug/retrieval"

# 保存评测文件的目录
OUTPUT_DIR = Path("evaluation_results")

# 需要测试的问题
QUESTIONS = [
    "论文提出了什么DOA估计方法？其基本原理是什么？",
    "论文为什么采用二阶延迟互相关？",
    "PARAFAC模型是如何构造的？",
    "Algorithm 1如何获得角度和距离的粗估计？",
    "论文如何实现精细且无模糊的参数估计？",
    "论文如何区分近场信源和远场信源？",
    "本文为什么使用统一的精确传播模型？",
    "本文如何避免传统的多维谱峰搜索？",
    "论文使用什么仿真参数和指标评价性能？",
    "论文的运行时间比较得出了什么结论？"
]

# 三种对比方法
METHODS = {
    "vector": "baseline_vector_top3",
    "mmr": "mmr_top5",
    "reranker": "reranker_top5"
}


# ================== 调用FastAPI ==================

def request_retrieval(question):

    # 将问题转换成JSON格式
    payload = json.dumps(
        {"query": question},
        ensure_ascii=False
    ).encode("utf-8")

    # 构造HTTP POST请求
    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    # 发送请求
    # CPU运行Reranker时可能较慢
    with urllib.request.urlopen(
        request,
        timeout=180
    ) as response:

        # 解析返回的JSON
        return json.loads(
            response.read().decode("utf-8")
        )


# ================== 收集检索结果 ==================

def collect():

    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 用于保存人工评分表
    rows = []

    # 保存性能统计
    performance = []

    print("开始进行RAG检索评测...")

    # 先预热Reranker，避免模型加载影响后续计时
    print("正在预热模型...")

    request_retrieval(QUESTIONS[0])

    print("模型预热完成。")

    # 遍历全部测试问题
    for question_id, question in enumerate(
        QUESTIONS,
        start=1
    ):

        print(
            f"正在评测 Q{question_id}/{len(QUESTIONS)}"
        )

        # 向FastAPI发送问题
        result = request_retrieval(question)

        # 获取检索性能数据
        timings = result.get(
            "timings_seconds",
            {}
        )

        # 保存当前问题的时间记录
        performance.append({
            "question_id": question_id,
            "question": question,
            "embedding_seconds": timings.get(
                "embedding", ""
            ),
            "reranker_seconds": timings.get(
                "reranker", ""
            ),
            "total_seconds": timings.get(
                "total", ""
            ),
            "candidate_count": result.get(
                "candidate_counts", {}
            ).get("reranked", 0)
        })

        # 分别整理三种检索结果
        for method, field in METHODS.items():

            results = result.get(field, [])

            # 遍历当前方法的所有结果
            for rank, item in enumerate(
                results,
                start=1
            ):

                rows.append({
                    "question_id": question_id,
                    "question": question,
                    "method": method,
                    "rank": rank,
                    "chunk_id": item.get(
                        "chunk_id", ""
                    ),
                    "page": item.get(
                        "page", ""
                    ),
                    "text": item.get(
                        "text", ""
                    ),
                    # 人工填写0、1、2
                    "relevance": ""
                })

    # 保存人工标注表
    labels_file = OUTPUT_DIR / "retrieval_labels.csv"

    with labels_file.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question_id",
                "question",
                "method",
                "rank",
                "chunk_id",
                "page",
                "text",
                "relevance"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    # 保存性能记录
    performance_file = (
        OUTPUT_DIR / "performance.json"
    )

    performance_file.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(),
                "question_count": len(QUESTIONS),
                "records": performance
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("检索收集完成！")
    print("人工评分表：", labels_file)
    print("性能统计：", performance_file)
    print("下一步：填写relevance列，然后运行score")


# ================== 计算检索指标 ==================

def score():

    labels_file = OUTPUT_DIR / "retrieval_labels.csv"

    if not labels_file.exists():
        print("未找到评分表，请先运行collect")
        return

    # 读取人工评分
    with labels_file.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        rows = list(csv.DictReader(file))

    # 保存每种方法、每道题的结果
    grouped = {}

    # 检查是否全部完成评分
    for row in rows:

        label = row["relevance"].strip()

        if label not in {"0", "1", "2"}:
            print(
                "存在未评分的文本块：",
                row["question_id"],
                row["method"],
                row["rank"]
            )
            return

        # 分组：方法+问题编号
        key = (
            row["method"],
            int(row["question_id"])
        )

        grouped.setdefault(key, []).append({
            "rank": int(row["rank"]),
            "relevance": int(label)
        })

    # 保存评测结果
    summary = {}

    for method in METHODS:

        p3_values = []
        hit5_values = []
        mrr5_values = []

        for question_id in range(
            1,
            len(QUESTIONS) + 1
        ):

            key = (method, question_id)

            # 读取当前问题的全部结果
            items = sorted(
                grouped.get(key, []),
                key=lambda x: x["rank"]
            )

            # 高度相关定义为2分
            relevant_ranks = [
                item["rank"]
                for item in items
                if item["relevance"] == 2
            ]

            # Precision@3
            p3 = sum(
                1 for rank in relevant_ranks
                if rank <= 3
            ) / 3

            # Hit@5
            hit5 = int(
                any(rank <= 5 for rank in relevant_ranks)
            )

            # MRR@5
            first_rank = min(
                [
                    rank for rank in relevant_ranks
                    if rank <= 5
                ],
                default=None
            )

            mrr5 = (
                1.0 / first_rank
                if first_rank is not None
                else 0.0
            )

            p3_values.append(p3)
            hit5_values.append(hit5)
            mrr5_values.append(mrr5)

        summary[method] = {
            "P@3": round(
                statistics.mean(p3_values), 4
            ),
            "Hit@5": round(
                statistics.mean(hit5_values), 4
            ),
            "MRR@5": round(
                statistics.mean(mrr5_values), 4
            )
        }

    # 纯向量方法仅返回Top-3
    # 所以其Hit@5/MRR@5实际只有Top-3可评
    # 为避免与Top-5方法不公平比较：
    # 只展示三种方法共同的P@3
    # 其他指标仅对Top-5方法进行比较

    print("\n===== 检索评测结果 =====")

    for method, metrics in summary.items():

        print(f"\n方法：{method}")
        print("P@3：", metrics["P@3"])

        if method != "vector":
            print("Hit@5：", metrics["Hit@5"])
            print("MRR@5：", metrics["MRR@5"])

    # 保存结果到JSON
    output_file = OUTPUT_DIR / "metrics.json"

    output_file.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n评测报告已保存：", output_file)


# ================== 程序入口 ==================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "请输入 collect 或 score"
        )

    elif sys.argv[1] == "collect":
        collect()

    elif sys.argv[1] == "score":
        score()

    else:
        print("未知操作，请使用collect或score")
