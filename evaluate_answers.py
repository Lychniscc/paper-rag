
# ============================================================
# evaluate_answers.py
#
# 阶段5.2.1：RAG端到端问答自动化评测
#
# 功能：
# 1. 自动发送10个论文问题
# 2. 保存模型完整回答和来源证据
# 3. 记录成功率、耗时和Token数量
# 4. 生成可人工评分的CSV
# 5. 人工评分后计算回答质量指标
#
# 使用方法：
# python evaluate_answers.py collect
# python evaluate_answers.py score
#
# 注意：
# collect会调用实际的Claude API，可能产生费用
# score只读取本地CSV，不调用Claude
# ============================================================

import csv
import json
import sys
import time
import statistics
import urllib.request
import urllib.error

from pathlib import Path
from datetime import datetime

import numpy as np


# ==================== 1. 基本配置 ====================

# 新版RAG问答接口
API_URL = "http://127.0.0.1:8000/rag-ask-v3"

# 输出文件夹
OUTPUT_DIR = Path("answer_evaluation_v3")

# 每道问题仅调用一次
# 失败不自动重试，避免重复消耗Token

# 10个固定测试问题
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


# ==================== 2. 调用RAG接口 ====================

def call_rag(question):

    # 将问题编码为JSON
    payload = json.dumps(
        {"question": question},
        ensure_ascii=False
    ).encode("utf-8")

    # 构造HTTP请求
    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    # 记录客户端发送请求前的时间
    start = time.perf_counter()

    try:

        # 调用FastAPI新版RAG接口
        with urllib.request.urlopen(
            request,
            timeout=180
        ) as response:

            status_code = response.status

            data = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as exc:

        elapsed = time.perf_counter() - start

        error_body = exc.read().decode(
            "utf-8",
            errors="replace"
        )

        return {
            "success": False,
            "http_status": exc.code,
            "elapsed_seconds": round(elapsed, 3),
            "error": error_body
        }

    except Exception as exc:

        elapsed = time.perf_counter() - start

        return {
            "success": False,
            "http_status": None,
            "elapsed_seconds": round(elapsed, 3),
            "error": str(exc)
        }

    # 客户端实际观测的请求耗时
    elapsed = time.perf_counter() - start

    # HTTP 200且具有非空答案才计为成功
    success = (
        status_code == 200
        and isinstance(data.get("answer"), str)
        and bool(data.get("answer", "").strip())
    )

    return {
        "success": success,
        "http_status": status_code,
        "elapsed_seconds": round(elapsed, 3),
        "data": data
    }


# ==================== 3. 安全读取数字 ====================

def to_float(value):

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value):

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ==================== 4. 自动采集10题 ====================

def collect():

    # 创建保存文件夹
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 保存本轮所有原始响应
    raw_results = []

    # 方便Excel人工评分的表格
    label_rows = []

    # 统计开始时间
    created_at = datetime.now().isoformat()

    print("开始阶段5.2.1端到端问答评测")
    print(f"问题数量：{len(QUESTIONS)}")
    print("每道题调用Claude API一次")
    print("=" * 50)

    # 逐题调用新版问答接口
    for index, question in enumerate(
        QUESTIONS,
        start=1
    ):

        print(
            f"\n[{index}/{len(QUESTIONS)}] {question}",
            flush=True
        )

        # 调用后端
        result = call_rag(question)

        # 保存完整原始响应
        raw_results.append({
            "question_id": index,
            "question": question,
            **result
        })

        # 请求失败时，保留错误信息
        if not result["success"]:

            print(
                f"失败：{result.get('error', '')}",
                flush=True
            )

            label_rows.append({
                "question_id": index,
                "question": question,
                "http_success": 0,
                "answer": "",
                "source_ids": "",
                "server_retrieval_seconds": "",
                "server_llm_seconds": "",
                "server_total_seconds": "",
                "client_elapsed_seconds": (
                    result["elapsed_seconds"]
                ),
                "input_tokens": 0,
                "output_tokens": 0,
                "citation_id_valid": "",
                "correctness": "",
                "completeness": "",
                "citation_support": "",
                "notes": result.get("error", "")
            })

            continue

        # 解析成功响应
        data = result["data"]

        timings = data.get(
            "timings_seconds", {}
        )

        usage = data.get(
            "usage", {}
        )

        citations = data.get(
            "citation_check", {}
        )

        sources = data.get(
            "sources", []
        )

        # 记录所引用的资料编号
        source_ids = ",".join(
            str(item.get("chunk_id", ""))
            for item in sources
        )

        # 引用编号合法性
        citation_valid = (
            citations.get(
                "reference_ids_valid", False
            )
            and citations.get(
                "has_citations", False
            )
        )

        # 保存人工评分记录
        label_rows.append({
            "question_id": index,
            "question": question,
            "http_success": 1,
            "answer": data.get("answer", ""),
            "source_ids": source_ids,
            "server_retrieval_seconds": timings.get(
                "retrieval", ""
            ),
            "server_llm_seconds": timings.get(
                "llm", ""
            ),
            "server_total_seconds": timings.get(
                "total", ""
            ),
            "client_elapsed_seconds": (
                result["elapsed_seconds"]
            ),
            "input_tokens": usage.get(
                "input_tokens", 0
            ),
            "output_tokens": usage.get(
                "output_tokens", 0
            ),
            "citation_id_valid": (
                int(citation_valid)
            ),
            # 以下3项由人工评分
            "correctness": "",
            "completeness": "",
            "citation_support": "",
            "notes": ""
        })

        # 打印当前进度
        print(
            f"成功！检索耗时："
            f"{timings.get('retrieval', 0)}秒"
        )

        print(
            f"Claude耗时："
            f"{timings.get('llm', 0)}秒"
        )

        print(
            f"总耗时："
            f"{timings.get('total', 0)}秒"
        )

        print(
            f"Token：输入"
            f"{usage.get('input_tokens', 0)}，"
            f"输出{usage.get('output_tokens', 0)}"
        )

    # ==================== 保存原始JSON ====================

    raw_file = (
        OUTPUT_DIR / "answers_raw.json"
    )

    raw_file.write_text(
        json.dumps(
            {
                "created_at": created_at,
                "question_count": len(QUESTIONS),
                "results": raw_results
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # ==================== 保存评分CSV ====================

    csv_file = (
        OUTPUT_DIR / "answer_labels.csv"
    )

    fieldnames = [
        "question_id",
        "question",
        "http_success",
        "answer",
        "source_ids",
        "server_retrieval_seconds",
        "server_llm_seconds",
        "server_total_seconds",
        "client_elapsed_seconds",
        "input_tokens",
        "output_tokens",
        "citation_id_valid",
        "correctness",
        "completeness",
        "citation_support",
        "notes"
    ]

    with csv_file.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(label_rows)

    # ==================== 自动统计基础数据 ====================

    successful = [
        row for row in label_rows
        if row["http_success"] == 1
    ]

    success_count = len(successful)

    success_rate = (
        success_count / len(QUESTIONS)
    )

    total_input = sum(
        to_int(row["input_tokens"])
        for row in label_rows
    )

    total_output = sum(
        to_int(row["output_tokens"])
        for row in label_rows
    )

    # 保存基础统计
    basic_metrics = {
        "question_count": len(QUESTIONS),
        "successful_requests": success_count,
        "request_success_rate": round(
            success_rate, 4
        ),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output
    }

    metrics_file = (
        OUTPUT_DIR / "basic_metrics.json"
    )

    metrics_file.write_text(
        json.dumps(
            basic_metrics,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # 输出提示
    print("\n" + "=" * 50)
    print("端到端问答采集完成！")

    print(
        f"请求成功：{success_count}/{len(QUESTIONS)}"
    )

    print(
        f"请求成功率：{success_rate * 100:.1f}%"
    )

    print("总输入Token：", total_input)
    print("总输出Token：", total_output)

    print("\n生成的文件：")
    print(raw_file.resolve())
    print(csv_file.resolve())
    print(metrics_file.resolve())

    print("\n下一步：人工检查答案并评分")


# ==================== 5. 计算人工评测指标 ====================

def score():

    csv_file = (
        OUTPUT_DIR / "answer_labels.csv"
    )

    if not csv_file.exists():

        print("找不到评分文件，请先运行collect")
        return

    # 读取人工评分表
    with csv_file.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        rows = list(csv.DictReader(file))

    if not rows:
        print("评分表为空")
        return

    # 检查评分是否完成
    for row in rows:

        # 请求失败的记录不要求人工评分
        if row["http_success"] != "1":
            continue

        # 成功回答必须具有3项人工评分
        for field in [
            "correctness",
            "completeness",
            "citation_support"
        ]:

            value = row[field].strip()

            if value not in {"0", "1", "2"}:

                print(
                    f"问题{row['question_id']} "
                    f"的{field}尚未评分"
                )

                return

    # ==================== 提取成功回答 ====================

    successful = [
        row for row in rows
        if row["http_success"] == "1"
    ]

    question_count = len(rows)

    successful_count = len(successful)

    # 请求成功率
    request_success_rate = (
        successful_count / question_count
    )

    # ==================== 回答正确率 ====================

    # 预先规定的合格标准：
    # 正确性=2
    # 完整性>=1
    # 引用支持>=1
    #
    # 这里表示整体回答合格率
    # 不等于逐条事实的完全正确率

    qualified_count = 0

    for row in successful:

        correctness = int(
            row["correctness"]
        )

        completeness = int(
            row["completeness"]
        )

        citation_support = int(
            row["citation_support"]
        )

        if (
            correctness == 2
            and completeness >= 1
            and citation_support >= 1
        ):
            qualified_count += 1

    # 以全部提交的问题为分母
    answer_pass_rate = (
        qualified_count / question_count
    )

    # ==================== 人工评分平均值 ====================

    def average_label(field):

        if not successful:
            return None

        return round(
            statistics.mean(
                int(row[field])
                for row in successful
            ),
            4
        )

    # ==================== 引用编号合法率 ====================

    citation_valid_count = sum(
        row["citation_id_valid"] == "1"
        for row in successful
    )

    citation_id_valid_rate = (
        citation_valid_count / successful_count
        if successful_count
        else 0.0
    )

    # ==================== 耗时统计 ====================

    def timing_stats(field):

        values = [
            to_float(row[field])
            for row in successful
        ]

        values = [
            value for value in values
            if value is not None
        ]

        if not values:
            return None

        return {
            "mean": round(
                statistics.mean(values), 3
            ),
            "P50": round(
                float(np.percentile(values, 50)),
                3
            ),
            "P95": round(
                float(np.percentile(values, 95)),
                3
            )
        }

    # ==================== Token统计 ====================

    total_input = sum(
        to_int(row["input_tokens"])
        for row in rows
    )

    total_output = sum(
        to_int(row["output_tokens"])
        for row in rows
    )

    # ==================== 汇总最终结果 ====================

    metrics = {
        "question_count": question_count,
        "successful_requests": successful_count,
        "request_success_rate": round(
            request_success_rate, 4
        ),
        "qualified_answers": qualified_count,
        "answer_pass_rate": round(
            answer_pass_rate, 4
        ),
        "mean_correctness_0_to_2": average_label(
            "correctness"
        ),
        "mean_completeness_0_to_2": average_label(
            "completeness"
        ),
        "mean_citation_support_0_to_2": average_label(
            "citation_support"
        ),
        "citation_id_valid_rate": round(
            citation_id_valid_rate, 4
        ),
        "server_retrieval_seconds": timing_stats(
            "server_retrieval_seconds"
        ),
        "server_llm_seconds": timing_stats(
            "server_llm_seconds"
        ),
        "server_total_seconds": timing_stats(
            "server_total_seconds"
        ),
        "client_elapsed_seconds": timing_stats(
            "client_elapsed_seconds"
        ),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output
    }

    # 保存最终评测JSON
    output_file = (
        OUTPUT_DIR / "answer_metrics.json"
    )

    output_file.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # 输出最终结果
    print("\n===== RAG端到端评测结果 =====")

    print(
        "请求成功率：",
        f"{request_success_rate * 100:.2f}%"
    )

    print(
        "回答合格率：",
        f"{answer_pass_rate * 100:.2f}%"
    )

    print(
        "平均正确性评分：",
        metrics["mean_correctness_0_to_2"]
    )

    print(
        "平均完整性评分：",
        metrics["mean_completeness_0_to_2"]
    )

    print(
        "平均引用支持评分：",
        metrics["mean_citation_support_0_to_2"]
    )

    print(
        "引用编号合法率：",
        f"{citation_id_valid_rate * 100:.2f}%"
    )

    print(
        "完整问答服务端耗时：",
        metrics["server_total_seconds"]
    )

    print(
        "完整问答客户端耗时：",
        metrics["client_elapsed_seconds"]
    )

    print(
        "Token总使用量：",
        total_input + total_output
    )

    print("\n结果已保存至：")
    print(output_file.resolve())


# ==================== 6. 命令行入口 ====================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "用法："
            "python evaluate_answers.py collect "
            "或 score"
        )

    elif sys.argv[1] == "collect":

        collect()

    elif sys.argv[1] == "score":

        score()

    else:

        print("未知操作：请使用collect或score")
