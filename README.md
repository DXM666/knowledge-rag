# RAG知识库项目

这是一个基于RAG (Retrieval-Augmented Generation) 的知识库项目，使用 LangChain 框架开发。

## 功能特点

- 支持PDF文档处理和分块
- 使用Faiss作为向量搜索引擎
- 使用PostgreSQL作为向量数据存储
- 使用DeepSeek-R1-Distill-Qwen-1.5B作为大模型
- 采用4bit量化技术加速推理
- 使用Transformers库进行模型加载和推理
- 交互式问答界面，支持基于知识库的问答
- 模块化设计，便于扩展和维护
- 完善的日志系统，支持文件日志和控制台错误提示
- 健壮的错误处理机制，提供详细的错误信息
- 支持大规模文档处理和索引优化

## 项目结构

```
.
├── data/                   # PDF文档目录
├── logs/                  # 日志文件目录
├── vector_store/          # 向量数据库存储目录
├── knowledgerag/          # 源代码目录
│   ├── core/              # 核心功能模块
│   │   ├── model_loader.py      # 模型加载和配置
│   │   ├── rag_chain.py         # RAG链构建器
│   │   └── interactive_qa.py    # 交互式问答界面
│   ├── db/                # 数据库相关模块
│   │   └── postgres.py           # PostgreSQL数据库连接和操作
│   ├── document_processing/  # 文档处理模块
│   │   ├── document_processor.py  # 文档处理器类
│   │   └── processor_runner.py    # 文档处理运行器
│   ├── embeddings/        # 嵌入模型模块
│   │   └── embedding_model.py     # 嵌入模型加载和使用
│   ├── retrieval/         # 检索模块
│   │   └── vector_store.py        # 向量存储和检索
│   ├── utils/             # 工具模块
│   │   └── logging_utils.py       # 日志工具
│   ├── config/            # 配置模块
│   │   └── settings.py            # 配置文件
│   └── cli/               # 命令行接口
│       └── commands.py            # 命令行命令
├── run.py                # 新版入口脚本
├── main.py               # 原版入口脚本
└── requirements.txt      # 项目依赖
```

## 安装依赖

```bash
pip install -r requirements.txt
```

或使用uv（推荐）：

```bash
uv pip install -r requirements.txt
```

## 数据库设置

本项目使用PostgreSQL作为向量数据存储，需要先安装PostgreSQL数据库，并安装pgvector扩展：

```sql
CREATE EXTENSION vector;
```

默认配置使用以下参数连接数据库：
- 主机：localhost
- 端口：5432
- 数据库：knowledge_base
- 用户：postgres
- 密码：postgres
- 表名：document_embeddings

可以在`knowledgerag/config/settings.py`中修改这些配置。

## 使用方法

### 处理文档

将需要处理的PDF文档放入 `data` 目录，然后运行：

```bash
python run.py --process
```

如需强制重建向量数据库：

```bash
python run.py --process --force-rebuild
```

### 启动交互式问答

```bash
python run.py
```

### 直接查询（不进入交互模式）

```bash
python run.py --query "你的问题"
```

### 自定义参数

```bash
python run.py --data-dir 自定义数据目录 --vector-store-path 自定义向量存储路径
```

## 工作流程

1. **文档处理**：系统读取PDF文档，并使用递归字符分割器将文档分割成小块
2. **向量化与索引**：
   - 使用嵌入模型将文本块转换为向量并存储到Faiss向量数据库
   - 将文本块和向量存储到PostgreSQL数据库，支持向量检索
3. **检索**：使用Faiss进行高效的向量相似度检索
4. **检索增强生成**：用户提问时，系统检索最相关的文本块，并结合大模型生成回答
5. **交互式界面**：提供简单的命令行交互界面，支持连续问答

## 配置说明

项目配置集中在 `knowledgerag/config/settings.py` 文件中，包括：

- 模型配置：模型名称、加载选项等
- PostgreSQL配置：连接参数、表名等
- 检索器配置：检索数量等
- 文本分割配置：分块大小、重叠大小等
- 提示模板：用于指导大模型生成回答的提示模板

## 技术实现细节

### 向量存储与检索

- **FAISS索引**：使用Facebook AI开发的FAISS库进行高效的向量相似度搜索
  - 默认使用`IndexFlatL2`进行精确检索
  - 对于大规模文档，可配置使用`IndexIVFFlat`或其他高效索引类型
  - 支持索引持久化到PostgreSQL数据库

- **文档存储**：使用`InMemoryDocstore`管理文档对象
  - 建立索引位置到文档ID的映射关系
  - 支持文档存储的重建和更新

### 模型加载与优化

- **4bit量化**：使用bitsandbytes库实现模型的4bit量化，大幅降低内存占用
- **自动设备映射**：根据可用硬件自动选择CPU或GPU进行推理
- **高效推理**：使用HuggingFace Transformers的pipeline机制进行高效推理

### 错误处理与日志

- **分层日志系统**：
  - 详细日志记录到时间戳命名的日志文件
  - 仅警告和错误信息输出到控制台
  - 第三方库日志级别控制

- **健壮的错误处理**：
  - 针对文档处理、向量转换、索引构建等关键环节的专门错误处理
  - 详细的错误堆栈信息记录
  - 用户友好的错误提示

## 性能优化

### 大规模文档处理

对于大规模文档集合（数十万文档以上），系统提供以下优化策略：

1. **索引类型自适应**：根据文档数量自动选择合适的索引类型
   - 小规模文档：使用`IndexFlatL2`提供精确检索
   - 大规模文档：使用`IndexIVFFlat`提供高效近似检索

2. **向量数据类型处理**：
   - 自动处理和转换不同格式的向量数据
   - 支持字符串格式和浮点数列表格式的向量

3. **内存优化**：
   - 批量处理文档，避免内存溢出
   - 使用临时文件进行大型索引的序列化和反序列化

## 最近改进

- **修复了FAISS索引映射问题**：确保索引位置到文档ID的映射正确更新
- **修复了文档存储类型问题**：使用正确的`InMemoryDocstore`类型而非普通字典
- **改进了日志配置**：将日志输出重定向到文件，避免干扰查询结果
- **优化了查询结果输出格式**：使用分隔线和格式化输出，提高可读性
- **增强了错误处理**：添加了更详细的错误信息和异常捕获

## 未来计划

- 支持更多文档格式（Word、HTML等）
- 实现增量更新机制，只处理新增或修改的文档
- 添加Web界面，提供更友好的用户交互
- 实现分布式索引，支持超大规模文档集合
- 集成更多大模型选项，支持模型切换
