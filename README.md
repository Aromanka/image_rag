# Image_RAG

Image_RAG 是面向工业与施工安全图像的本地多模态 RAG 系统。服务使用 SigLIP2 编码查询图像，从 ChromaDB 检索相似历史案例，再将查询图像和检索上下文交给本地视觉语言模型完成安全判断。

当前仓库只保留两条主要工作流：

1. 使用 `image_server.py` 启动原始图片字节推理服务；
2. 使用 `evaluate_image_server.py` 直接评估服务后端，不经过 HTTP。

项目面向 Linux GPU 主机运行，模型只从本地路径加载，不会在线下载。

## 1. 环境准备

要求：

- Python 3.10+
- 支持当前 PyTorch 版本的 NVIDIA GPU
- 本地 SigLIP2、VLM、LoRA 和可选 SBERT 模型
- 已构建的 ChromaDB RAG 索引

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --no-index \
  --find-links /path/to/local/wheelhouse \
  -r requirements.txt
```

运行前检查 `config.py` 中的路径：

```python
INSPECSAFE_DATA_ROOT = "/root/autodl-tmp/data/inspecsafe/DATA_PATH"
EMBED_MODEL_PATH = "/root/autodl-tmp/model/siglip2"
VLM_MODEL_PATH = "/root/autodl-tmp/model/gemma3_4b"
SBERT_MODEL_PATH = "/root/autodl-tmp/model/all-MiniLM-L6-v2/sentence-transformers/all-MiniLM-L6-v2"
```

默认 LoRA 目录为：

```text
lora_weights/gemma3_4b_lora_v2
lora_weights/gemma3_4b_lora_inspecsafe
lora_weights/gemma3_4b_lora_lab_v1
```

## 2. 构建 RAG 索引

如果对应的 `chroma_db/<dataset>/` 已经存在，可以跳过本节。建议只使用训练集构建索引，避免测试数据泄漏。

InspecSafe：

```bash
python build_index.py \
  --dataset-input data/inspecsafe/train.csv
```

ConstructionSite-10K：

```bash
python build_index.py \
  --constructionsite-json constructionsite_10k/train.json
```

LabSafety-Gen：

```bash
python build_index.py \
  --lab-safety-gen-jsonl data/lab_safety_gen/annotations.jsonl \
  --split train
```

索引分别写入：

```text
chroma_db/inspecsafe/
chroma_db/constructionsite10k/
chroma_db/lab_safety_gen/
```

ChromaDB 中保存的 `image_path` 必须在运行服务器的机器上可访问，因为生成 RAG prompt 时仍需读取检索到的原始图片。

## 3. 启动 image server

启动服务：

```bash
python image_server.py --host 0.0.0.0 --port 8000
```

服务默认预加载 SigLIP2 和 VLM。看到以下输出后再发送请求：

```text
Model loading complete. Server is ready.
```

调试时可延迟模型加载：

```bash
python image_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --no-preload
```

指定初始 LoRA：

```bash
python image_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --lora-weights lora_weights/gemma3_4b_lora_inspecsafe
```

常用启动参数：

```text
--top-k N
--max-new-tokens N
--stage-one-max-new-tokens N
--stage-two-max-new-tokens N
--max-upload-mb N
--lora-weights PATH
--no-preload
--local-test
--local-test-dataset {inspecsafe_safety_level,labsafety_gen}
--local-test-history-size N
```

服务只启动一个 worker，并且同一时间只执行一个 GPU 推理请求。模型繁忙时会立即返回 `BUSY`，客户端应稍后重试。

### 3.1 健康检查

```bash
curl "http://127.0.0.1:8000/health"
```

健康检查会返回支持的推理模式、当前 LoRA、RAG 参数和本地测试通道状态。

### 3.2 切换 LoRA

基础 VLM 保持加载，通过以下接口切换适配器：

```bash
curl -X POST "http://127.0.0.1:8000/model/switch?model=inspecsafe"
curl -X POST "http://127.0.0.1:8000/model/switch?model=constructionsite"
curl -X POST "http://127.0.0.1:8000/model/switch?model=labsafety"
```

`/model/switch` 和 `/infer` 共用推理锁。切换或推理期间收到 `BUSY` 时应稍后重试。

### 3.3 发送图片

请求体直接发送图片字节，不使用 multipart form。

Accuracy：安全等级结构化 RAG 推理。

```bash
curl --data-binary @query.jpg \
  -H "Content-Type: image/jpeg" \
  "http://127.0.0.1:8000/infer?mode=accuracy"
```

Latency：不使用 RAG 的两阶段安全判断。

```bash
curl --data-binary @query.jpg \
  -H "Content-Type: image/jpeg" \
  "http://127.0.0.1:8000/infer?mode=latency"
```

Energy：当前使用与 Accuracy 相同的安全等级 RAG 路径，但保留独立模式名称。

```bash
curl --data-binary @query.jpg \
  -H "Content-Type: image/jpeg" \
  "http://127.0.0.1:8000/infer?mode=energy"
```

Balanced：先共享一次 top-3 RAG 检索，再执行两阶段安全判断。

```bash
curl --data-binary @query.jpg \
  -H "Content-Type: image/jpeg" \
  "http://127.0.0.1:8000/infer?mode=balanced"
```

Accuracy 和 Energy 可按请求覆盖检索数量；Balanced 固定使用 top-3：

```bash
curl --data-binary @query.jpg \
  -H "Content-Type: image/jpeg" \
  "http://127.0.0.1:8000/infer?mode=accuracy&top_k=8"
```

成功响应的核心字段为：

```json
{
  "status": "success",
  "mode": "accuracy",
  "safe": "safe",
  "annotation": "No visible hazards.",
  "response": "original model output",
  "response_time_seconds": 2.731
}
```

RAG 模式还会返回 `rag_dataset`、`top_k`、`gated_rag`、`retrieved_count_before_gate` 和 `retrieved_count`。

### 3.4 SSH 端口转发

不建议把服务端口直接暴露到不可信网络。可在本地建立隧道：

```bash
ssh -N \
  -L 18000:127.0.0.1:8000 \
  -p SSH_PORT \
  USER@SERVER
```

随后通过本地端口访问：

```bash
curl --data-binary @query.jpg \
  -H "Content-Type: image/jpeg" \
  "http://127.0.0.1:18000/infer?mode=accuracy"
```

### 3.5 响应转发

在 `config.py` 中设置 `RESPONSE_FORWARD_URL` 后，成功的模型原始输出会以后台任务形式发送到目标 HTTP 服务。转发时间不计入 `response_time_seconds`；留空表示关闭。

## 4. RAG evaluation

统一评估入口为：

```bash
python evaluate_image_server.py --dataset DATASET [OPTIONS]
```

支持的数据集：

```text
inspecsafe
constructionsite10k
labsafety_gen
```

支持的模式：

```text
all
accuracy
latency
energy
balanced
```

默认 `--mode all` 会依次评估四个服务模式。评估器直接调用 `image_server.py` 的后端函数，不启动 HTTP 服务，也不发送响应转发或 WebSocket 通知。

### 4.1 快速冒烟评估

先用少量样本和 `--skip-sbert` 验证完整推理链路：

```bash
python evaluate_image_server.py \
  --dataset inspecsafe \
  --mode accuracy \
  --limit 5 \
  --skip-sbert \
  --output save/smoke_inspecsafe.json
```

### 4.2 InspecSafe

默认标注文件为 `data/inspecsafe/test.csv`：

```bash
python evaluate_image_server.py \
  --dataset inspecsafe \
  --mode all \
  --lora-weights lora_weights/gemma3_4b_lora_inspecsafe \
  --output save/eval_inspecsafe.json
```

CSV 中的图片路径必须在评估机器上存在。也可以显式指定标注文件：

```bash
python evaluate_image_server.py \
  --dataset inspecsafe \
  --dataset-path /path/to/test.csv \
  --mode accuracy \
  --skip-sbert
```

### 4.3 ConstructionSite-10K

默认标注文件为 `constructionsite_10k/test.json`。如果 JSON 中的相对图片路径不位于该目录下，需要传入 `--image-root`：

```bash
python evaluate_image_server.py \
  --dataset constructionsite10k \
  --dataset-path constructionsite_10k/test.json \
  --image-root /path/to/constructionsite_10k/images \
  --mode all \
  --lora-weights lora_weights/gemma3_4b_lora_v2 \
  --output save/eval_constructionsite10k.json
```

### 4.4 LabSafety-Gen

默认标注文件为 `data/lab_safety_gen/annotations.jsonl`，默认只评估 test split：

```bash
python evaluate_image_server.py \
  --dataset labsafety_gen \
  --split test \
  --mode all \
  --lora-weights lora_weights/gemma3_4b_lora_lab_v1 \
  --output save/eval_labsafety_gen.json
```

如果图片不在标注文件旁的相对路径下：

```bash
python evaluate_image_server.py \
  --dataset labsafety_gen \
  --dataset-path /path/to/annotations.jsonl \
  --image-root /path/to/labsafety_gen \
  --split test \
  --mode accuracy
```

### 4.5 SBERT 与常用参数

默认情况下评估器会加载 `SBERT_MODEL_PATH`，计算预测 annotation 与参考描述的语义相似度。没有本地 SBERT 模型时必须传入：

```text
--skip-sbert
```

常用参数：

```text
--dataset-path PATH
--mode {all,accuracy,latency,energy,balanced}
--split {train,test,all}
--image-root PATH
--limit N
--offset N
--top-k N
--max-new-tokens N
--stage-one-max-new-tokens N
--stage-two-max-new-tokens N
--checkpoint-every N
--sbert-path PATH
--sbert-device DEVICE
--sbert-batch-size N
--skip-sbert
--no-preload
--lora-weights PATH
--output PATH
```

若未提供 `--output`，结果默认写入：

```text
save/image_server_eval_<dataset>_<timestamp>.json
```

评估报告包含：

- 每种模式的 accuracy、precision、recall 和 F1；
- TP、FP、TN 和 FN；
- coverage 与 end-to-end accuracy；
- 平均延迟和 P95 延迟；
- 可选 SBERT annotation 相似度；
- 每条样本的模型原始输出、标准化结果和错误信息。

评估过程默认每 10 张图片写入一次 checkpoint，可通过 `--checkpoint-every` 调整，设为 `0` 表示关闭中途保存。

## 5. 静态检查

无需加载 GPU 模型即可检查当前核心 Python 文件：

```bash
python -m compileall \
  image_server.py build_index.py config.py embedding.py \
  rag_answer.py response_forwarding.py retrieval_gating.py \
  retriever.py two_stage_inference.py vlm_inference.py \
  evaluate_image_server.py evaluate_constructionsite10k.py \
  evaluate_labsafety_gen.py utils
```

查看完整命令参数：

```bash
python image_server.py --help
python evaluate_image_server.py --help
python build_index.py --help
```

## 6. 当前核心文件

```text
image_server.py                 原始图片 HTTP 推理服务
evaluate_image_server.py        统一服务后端评估入口
vlm_inference.py                VLM、RAG 和两阶段推理
embedding.py                    SigLIP2 图像/文本编码
retriever.py                    ChromaDB 检索
retrieval_gating.py             相似度门控
rag_answer.py                   多图 RAG prompt 构造
two_stage_inference.py          两阶段安全判断策略
build_index.py                  RAG 索引构建
config.py                       模型、数据、LoRA 和服务配置
response_forwarding.py          可选结果转发
utils/evaluate_utils.py         输出解析与评估指标
utils/local_test_channel.py     可选 WebSocket 完成通知
utils/local_test_data.py        本地测试数据集名称适配
utils/inspecsafe_paths.py       InspecSafe 路径解析
```
