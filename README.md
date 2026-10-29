# Academic Paper RAG — 学术论文智能问答系统

基于Python、FastAPI和大语言模型构建的学术论文检索增强生成（RAG）系统，支持IEEE双栏PDF解析、混合检索、Cross-Encoder重排序、中文论文问答及SQLite历史记录持久化。

## 1. 项目简介

针对科研论文篇幅较长、双栏排版复杂、关键词检索难以理解语义等问题，开发学术论文智能问答后端。

系统通过PDF解析和文本分块建立论文知识库，结合Embedding语义召回、关键词检索、RRF排序融合与Cross-Encoder重排序，选取相关原文片段供大语言模型生成中文回答，并返回原文页码和引用信息。

## 2. 技术栈

- **后端框架：** Python、FastAPI、Uvicorn
- **PDF解析：** PyMuPDF、PyMuPDF4LLM
- **语义检索：** Sentence-Transformers、Embedding
- **检索优化：** Keyword Search、RRF、MMR、Cross-Encoder
- **大模型接口：** Anthropic Messages API兼容接口
- **数据持久化：** SQLite
- **测试与版本管理：** Pytest、Git

## 3. 系统架构

```text
上传PDF
   |
PDF解析与双栏阅读顺序处理
   |
文本分块
   |
Embedding向量化
   |
用户提问
   |
   +---- 向量语义召回
   |
   +---- 关键词召回
   |
RRF排序融合
   |
Cross-Encoder重排序
   |
Top-5相关文本片段
   |
Claude生成中文回答
   |
引用编号与异常输出检查
   |
返回答案、原文出处与耗时
   |
SQLite保存问答历史
```

## 4. 核心功能

### PDF解析

支持PDF上传、正文提取和文本分块，并针对IEEE双栏论文实现按栏读取与参考文献过滤，减少正文阅读顺序混乱对检索结果的影响。

### 混合检索

结合Embedding语义相似度与关键词匹配，通过RRF融合不同检索结果，使用MMR控制候选冗余，并引入Cross-Encoder对候选片段进行精细排序。

### RAG问答

通过Anthropic Messages兼容接口调用大语言模型，基于检索到的论文片段生成中文回答，返回原文页码、引用编号和Token使用量。

### 可靠性处理

实现临时API错误有限重试、指数退避、异常回答格式检查、引用编号有效性检查以及响应耗时统计。

### SQLite问答历史

使用SQLite保存成功生成的问答记录，支持历史列表查询、详情查询及指定记录删除。服务器重启后，已保存的问答历史仍可访问。

## 5. 主要API接口

| 接口 | 方法 | 功能 |
|---|---|---|
| `/upload-pdf` | POST | 上传并解析PDF |
| `/rag-ask-v3` | POST | RAG问答与可靠性检查 |
| `/history` | GET | 查询问答历史 |
| `/history/{record_id}` | GET | 查询单条记录 |
| `/history/{record_id}` | DELETE | 删除指定记录 |
| `/debug/retrieval` | POST | 查看检索过程及结果 |

可通过FastAPI提供的Swagger页面进行接口调试。

## 6. 实验评测

### 检索质量

基于单篇14页IEEE论文，构建10个检索测试问题，对130条检索结果进行人工相关性标注。

| 检索方案 | Hit@5 | MRR@5 |
|---|---:|---:|
| RRF + MMR | 70% | 0.428 |
| RRF + Cross-Encoder | 100% | 0.642 |

上述结果仅代表当前测试集，不代表对所有论文的泛化性能。

### 端到端问答

在10题端到端评测中：

- V2请求成功率：90%
- V2人工回答合格率：70%
- V3请求成功率：90%
- V3严格人工回答合格率：60%

V3增加了API错误重试和异常输出拦截，但尚未在当前小样本上证明整体回答质量优于V2。

### 性能分析

在20候选配置的初步检索实验中，预热后平均检索耗时约2.33秒。

系统分别记录检索耗时、模型生成耗时和完整请求耗时，用于定位性能瓶颈。

## 7. 本地运行

### 创建虚拟环境

```bash
python -m venv .venv
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

在项目根目录创建`.env`文件，填写所使用的模型平台配置：

```env
ANTHROPIC_API_KEY=your_api_key
ANTHROPIC_BASE_URL=https://your-api-provider.example
ANTHROPIC_MODEL=your_model_id
```

环境变量名称应与项目实际代码保持一致。

### 启动服务

```bash
python -m uvicorn main:app
```

启动后访问：

`http://127.0.0.1:8000/docs`

先上传PDF，再调用RAG问答接口。

## 8. 自动化测试

使用Pytest测试SQLite数据库CRUD与RAG回答有效性检查。

```bash
python -m pytest tests -v
```

当前共有11项基础自动化测试，已全部通过。

## 9. 当前局限与后续方向

- PDF复杂数学公式仍可能存在解析缺失。
- 目前使用进程内存保存PDF文本和Embedding索引，服务器重启后需要重新上传PDF。
- 引用编号合法性检查不等同于完整的事实一致性验证。
- 现有评测主要基于单篇论文和10个问题，仍需扩大测试集。
- 当前为本地开发与学习项目，尚未实现多用户认证与生产环境部署。

## 10. 项目总结

本项目完成从PDF解析、混合检索、模型生成到问答历史持久化的RAG后端流程，并通过人工标注、性能评测及Pytest测试验证关键模块，实现了检索算法实验与后端工程实践的结合。