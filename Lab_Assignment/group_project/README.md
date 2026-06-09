# Bài Tập Nhóm — RAG Chatbot & Evaluation

## Mục Tiêu

Xây dựng chatbot trả lời câu hỏi về pháp luật Việt Nam liên quan tới ma túy và các bài báo về nghệ sĩ liên quan tới ma túy. Hệ thống dùng pipeline cá nhân Task 1-10, có citation, hiển thị source chunks, hỗ trợ follow-up cơ bản và có evaluation report.

## Deliverables

- `group_project/app.py` — Streamlit chatbot demo.
- `group_project/rag_service.py` — wrapper dùng chung cho UI và evaluation.
- `group_project/evaluation/golden_dataset.json` — 15 Q&A pairs.
- `group_project/evaluation/eval_pipeline.py` — script evaluation với 4 metrics và A/B comparison.
- `group_project/evaluation/results.md` — báo cáo điểm, worst performers, recommendations.
- `group_project/evaluation/results.json` — raw evaluation output để debug.

## Kiến Trúc Hệ Thống

```text
User
  |
  v
Streamlit Chat UI (group_project/app.py)
  |
  v
RAG Service (group_project/rag_service.py)
  |
  v
Task 10 Generation with Citation
  |
  v
Task 9 Retrieval Pipeline
  |
  +--> Semantic Search (Task 5)
  +--> Lexical BM25 Search (Task 6)
  +--> RRF Merge + Rerank (Task 7)
  +--> PageIndex fallback (Task 8)
  |
  v
Markdown corpus from Task 1-3 and index from Task 4
```

## Chatbot

### Tính năng

- Giao diện chat bằng Streamlit.
- Gọi trực tiếp pipeline Task 9/10.
- Trả lời có citation dạng `[source.md]`.
- Hiển thị source chunks trong expander.
- Có conversation memory đơn giản cho câu hỏi follow-up ngắn.

### Chạy app

```bash
pip install -r requirements.txt
streamlit run group_project/app.py
```

## Evaluation Pipeline

### Golden Dataset

Dataset có 15 câu bao phủ:

- Luật Phòng, chống ma túy 2021.
- Nghị định 105/2021/NĐ-CP.
- Nghị định 57/2022/NĐ-CP.
- Các bài báo về Nhikolai Đinh, Hữu Tín, G-Dragon, Lee Sun Kyun, Oh Kwang Rok.
- Một câu kiểm tra hành vi khi thiếu evidence.

### Metrics

Script dùng custom offline evaluator lấy cảm hứng từ RAGAS/DeepEval:

- **Faithfulness:** answer có được hỗ trợ bởi retrieved context không.
- **Answer Relevance:** answer có overlap với question không.
- **Context Recall:** retrieved context có chứa expected evidence không.
- **Context Precision:** tỷ lệ chunks có ích trong top contexts.

### A/B Comparison

- **Config A:** hybrid search + RRF + reranking.
- **Config B:** hybrid search + RRF, không reranking.

### Chạy evaluation

```bash
python group_project/evaluation/eval_pipeline.py
```

Kết quả hiện tại:

| Metric | Config A (hybrid + rerank) | Config B (hybrid no rerank) | Delta |
|--------|-----------------------------|------------------------------|-------|
| Faithfulness | 0.936 | 0.938 | -0.002 |
| Answer Relevance | 0.743 | 0.715 | +0.028 |
| Context Recall | 0.933 | 0.933 | +0.000 |
| Context Precision | 1.000 | 1.000 | +0.000 |
| Average | 0.903 | 0.897 | +0.006 |

Chi tiết nằm trong `group_project/evaluation/results.md`.

## Phân Công Công Việc

| Thành viên | MSSV | File phụ trách | Nhiệm vụ | Trạng thái |
|-----------|------|----------------|----------|------------|
| Người 1 | | `group_project/app.py` | Làm Streamlit chatbot, chat history, hiển thị sources | Hoàn thành |
| Người 2 | | `group_project/rag_service.py`, `src/task9_retrieval_pipeline.py`, `src/task10_generation.py` | Tích hợp RAG pipeline, citation, follow-up query wrapper | Hoàn thành |
| Người 3 | | `group_project/evaluation/golden_dataset.json`, `group_project/evaluation/eval_pipeline.py` | Tạo 15 Q&A, chạy 4 metrics, A/B comparison | Hoàn thành |
| Người 4 | | `group_project/evaluation/results.md`, `group_project/README.md` | Viết report, phân tích worst performers, hướng dẫn chạy | Hoàn thành |

## Hướng Dẫn Chuẩn Bị

```bash
pip install -r requirements.txt
python src/task3_convert_markdown.py
python src/task4_chunking_indexing.py
```

Nếu có PageIndex key, đặt trong `.env`:

```env
PAGEINDEX_API_KEY=...
```

Không commit `.env` vì chứa API key.

## Known Limitations

- Legal docs hiện là DOCX fallback/tóm tắt, nên câu hỏi pháp luật chi tiết như Điều 249 Bộ luật Hình sự cần bổ sung văn bản đầy đủ hơn.
- News crawl còn nhiễu menu/footer, ảnh hưởng precision và generation.
- Evaluator hiện là offline token-overlap, nên chưa thay thế hoàn toàn LLM-as-judge như DeepEval/RAGAS/TruLens.

## Đề Xuất Cải Tiến

1. Bổ sung full text Bộ luật Hình sự 2015 sửa đổi 2017, đặc biệt các điều về tội phạm ma túy.
2. Làm sạch HTML/news content trước khi convert markdown.
3. Chạy DeepEval hoặc RAGAS thật khi có OpenAI/Judge model API key.
