
# ==========================================================
# retrieval_v514.py
# 阶段5.1.4：RAG检索诊断与Cross-Encoder重排序
#
# 使用已有：
# 1. Embedding向量及问题向量
# 2. 关键词检索函数
# 3. PDF文本块与页码
#
# 新增：
# 1. 扩大候选召回数量
# 2. RRF融合
# 3. MMR去重
# 4. Cross-Encoder精细排序
# 5. 详细检索诊断
# ==========================================================

# ==================== 1. 导入依赖 ====================

# 统计各阶段运行时间
import time

# 缓存Reranker模型
from functools import lru_cache

# 数组计算
import numpy as np

# 载入Cross-Encoder模型
from sentence_transformers import CrossEncoder

# 将同步计算移到工作线程
from starlette.concurrency import run_in_threadpool


# ==================== 2. 检索参数 ====================

# 向量检索召回20个
VECTOR_TOP_K = 20

# 关键词检索召回20个
KEYWORD_TOP_K = 20

# RRF平滑常数
RRF_K = 60

# 关键词融合权重
KEYWORD_WEIGHT = 0.55

# 最终返回5个结果
FINAL_TOP_K = 5

# Reranker最多处理的候选数
RERANK_CANDIDATES = 20

# MMR参数
MMR_LAMBDA = 0.85

# 多语言Reranker模型
RERANK_MODEL = (
    "cross-encoder/"
    "mmarco-mMiniLMv2-L12-H384-v1"
)


# ==================== 3. 加载Reranker模型 ====================

@lru_cache(maxsize=1)
def get_reranker():

    # 第一次调用会下载模型
    # 之后从缓存读取
    model = CrossEncoder(
        RERANK_MODEL,
        device="cpu",
        max_length=512
    )

    return model


# ==================== 4. 获取排名列表 ====================

def top_indices(scores, limit, positive_only=False):

    # 转换为NumPy数组
    values = np.asarray(scores)

    # 从高到低排列
    order = np.argsort(values)[::-1]

    # 关键词检索只保留正分结果
    if positive_only:

        order = [
            int(i)
            for i in order
            if values[i] > 0
        ]

    # 返回指定数量
    return [int(i) for i in order[:limit]]


# ==================== 5. RRF融合 ====================

def rrf_fusion(vector_indices, keyword_indices):

    # 保存候选的融合得分
    scores = {}

    # 向量召回贡献
    for rank, index in enumerate(
        vector_indices,
        start=1
    ):

        scores[index] = (
            scores.get(index, 0.0)
            + 1.0 / (RRF_K + rank)
        )

    # 关键词召回贡献
    for rank, index in enumerate(
        keyword_indices,
        start=1
    ):

        scores[index] = (
            scores.get(index, 0.0)
            + KEYWORD_WEIGHT / (RRF_K + rank)
        )

    # 返回每个候选的融合分数
    return scores


# ==================== 6. MMR选择 ====================

def mmr_select(fusion_scores, embeddings, limit):

    # 无候选时返回空列表
    if not fusion_scores:
        return []

    # 记录尚未选中的候选
    remaining = set(fusion_scores)

    # 保存已选结果
    selected = []

    # 获取最大融合分数
    max_fusion = max(fusion_scores.values())

    # 逐个挑选候选
    while remaining and len(selected) < limit:

        best_index = None
        best_score = float("-inf")

        # 遍历剩余候选
        for index in sorted(remaining):

            # 将融合分数归一化
            relevance = (
                fusion_scores[index] / max_fusion
            )

            # 计算和已选候选的重复程度
            redundancy = 0.0

            if selected:

                redundancy = max(
                    float(
                        embeddings[index] @ embeddings[j]
                    )
                    for j in selected
                )

                # 不使用负相似度惩罚
                redundancy = max(redundancy, 0.0)

            # MMR公式
            score = (
                MMR_LAMBDA * relevance
                - (1 - MMR_LAMBDA) * redundancy
            )

            # 保存最优候选
            if score > best_score:

                best_score = score
                best_index = index

        # 加入结果
        selected.append(best_index)

        # 移出剩余集合
        remaining.remove(best_index)

    return selected


# ==================== 7. Reranker打分 ====================

def cross_encoder_scores(query, candidates):

    # 加载模型
    model = get_reranker()

    # 构造问题-候选文本对
    pairs = [
        (query, item["text"])
        for item in candidates
    ]

    # 无候选直接返回
    if not pairs:
        return []

    # 批量计算交叉编码器分数
    scores = model.predict(
        pairs,
        batch_size=8,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    # 转换为Python列表
    return (
        np.asarray(scores)
        .reshape(-1)
        .astype(float)
        .tolist()
    )


# ==================== 8. 整理输出格式 ====================

def format_results(
    indices,
    chunks,
    similarities,
    fusion_scores=None,
    reranker_scores=None
):

    results = []

    # 按输入顺序依次整理
    for index in indices:

        chunk = chunks[index]

        item = {
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "similarity": round(
                float(similarities[index]), 4
            ),
            "text": chunk["text"]
        }

        # 添加RRF分数
        if fusion_scores is not None:

            item["rrf_score"] = round(
                float(fusion_scores.get(index, 0)),
                6
            )

        # 添加Cross-Encoder分数
        if (
            reranker_scores is not None
            and index in reranker_scores
        ):

            item["reranker_score"] = round(
                float(reranker_scores[index]),
                4
            )

        results.append(item)

    return results


# ==================== 9. 完整检索流水线 ====================

async def run_retrieval(
    query,
    chunks,
    embeddings,
    encode_texts,
    lexical_scores,
    debug=True
):

    # 总计时开始
    total_start = time.perf_counter()

    # 转为NumPy数组
    embeddings = np.asarray(embeddings)

    # 检查索引
    if len(chunks) == 0:
        raise ValueError("论文索引为空")

    if len(chunks) != len(embeddings):
        raise ValueError("文本块数量和向量数量不一致")

    # ---------- 阶段A：生成问题向量 ----------

    start = time.perf_counter()

    # 沿用main.py中已安装的Embedding模型
    question_vectors = await run_in_threadpool(
        encode_texts,
        [query]
    )

    # 问题向量
    question_vector = question_vectors[0]

    # 计算归一化向量内积
    similarities = embeddings @ question_vector

    vector_time = time.perf_counter() - start

    # ---------- 阶段B：关键词得分 ----------

    start = time.perf_counter()

    # 使用main.py中的术语映射和关键词算法
    keyword_scores = lexical_scores(
        query,
        chunks
    )

    lexical_time = time.perf_counter() - start

    # ---------- 阶段C：分别召回 ----------

    # 纯向量Top-20
    vector_rank = top_indices(
        similarities,
        VECTOR_TOP_K
    )

    # 关键词Top-20
    keyword_rank = top_indices(
        keyword_scores,
        KEYWORD_TOP_K,
        positive_only=True
    )

    # ---------- 阶段D：RRF融合 ----------

    start = time.perf_counter()

    fusion_scores = rrf_fusion(
        vector_rank,
        keyword_rank
    )

    # 按融合分数降序排列
    fused_rank = sorted(
        fusion_scores,
        key=lambda i: (
            -fusion_scores[i],
            i
        )
    )

    fusion_time = time.perf_counter() - start

    # ---------- 阶段E：旧方案MMR ----------

    start = time.perf_counter()

    mmr_rank = mmr_select(
        fusion_scores,
        embeddings,
        FINAL_TOP_K
    )

    mmr_time = time.perf_counter() - start

    # ---------- 阶段F：Cross-Encoder重排序 ----------

    start = time.perf_counter()

    # 仅对融合排序靠前的候选执行精细排序
    candidate_indices = fused_rank[
        :RERANK_CANDIDATES
    ]

    # 形成问题-段落列表
    candidates = [
        {
            "index": i,
            "text": chunks[i]["text"]
        }
        for i in candidate_indices
    ]

    # Cross-Encoder属于同步计算
    # 放入工作线程，避免阻塞FastAPI事件循环
    scores = await run_in_threadpool(
        cross_encoder_scores,
        query,
        candidates
    )

    # 保存候选的Cross-Encoder分数
    reranker_scores = {
        index: float(score)
        for index, score in zip(
            candidate_indices,
            scores
        )
    }

    # 根据Cross-Encoder分数重新排序
    reranked_indices = sorted(
        candidate_indices,
        key=lambda i: (
            -reranker_scores[i],
            i
        )
    )

    # 最终Top-5
    final_indices = reranked_indices[:FINAL_TOP_K]

    reranker_time = time.perf_counter() - start

    # ---------- 阶段G：整理检索结果 ----------

    # 旧版纯向量Top-3
    baseline = format_results(
        vector_rank[:3],
        chunks,
        similarities
    )

    # RRF融合后的Top-5
    rrf_results = format_results(
        fused_rank[:5],
        chunks,
        similarities,
        fusion_scores
    )

    # MMR选出的Top-5
    mmr_results = format_results(
        mmr_rank,
        chunks,
        similarities,
        fusion_scores
    )

    # Cross-Encoder最终Top-5
    final_results = format_results(
        final_indices,
        chunks,
        similarities,
        fusion_scores,
        reranker_scores
    )

    # 统计总运行时间
    total_time = time.perf_counter() - total_start

    # 非调试模式，仅返回最终结果
    if not debug:
        return final_results

    # 调试模式，返回各阶段的排名
    return {
        "query": query,

        # 纯向量原始结果
        "baseline_vector_top3": baseline,

        # 关键词召回的候选
        "keyword_top10": [
            {
                **format_results(
                    [i],
                    chunks,
                    similarities
                )[0],
                "keyword_score": round(
                    float(keyword_scores[i]), 4
                )
            }
            for i in keyword_rank[:10]
        ],

        # 向量召回前10
        "vector_top10": format_results(
            vector_rank[:10],
            chunks,
            similarities
        ),

        # RRF融合后排名
        "rrf_top5": rrf_results,

        # MMR排名
        "mmr_top5": mmr_results,

        # Cross-Encoder最终结果
        "reranker_top5": final_results,

        # 本次候选规模
        "candidate_counts": {
            "vector": len(vector_rank),
            "keyword": len(keyword_rank),
            "unique_fused": len(fused_rank),
            "reranked": len(candidate_indices)
        },

        # 各阶段执行耗时，单位秒
        "timings_seconds": {
            "embedding": round(vector_time, 3),
            "keyword": round(lexical_time, 3),
            "rrf": round(fusion_time, 3),
            "mmr": round(mmr_time, 3),
            "reranker": round(reranker_time, 3),
            "total": round(total_time, 3)
        },

        "reranker_model": RERANK_MODEL,

        "message": "检索流水线诊断完成"
    }
