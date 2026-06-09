"""
Supervisor-Workers Multi-Agent RAG System
=========================================
Cải tiến Day08 RAG Pipeline bằng pattern Supervisor-Workers (LangGraph).

Kiến trúc:
    User Query
        │
        ▼
    Supervisor Agent  ──→ phân loại intent, quyết định routing
        │
        ├──→ Worker 1: LegalRetriever   — chuyên văn bản pháp luật
        ├──→ Worker 2: NewsRetriever    — chuyên tin tức nghệ sĩ
        └──→ Worker 3: Generator Agent — tổng hợp + sinh câu trả lời có citation

Chạy:
    python supervisor_agent.py
    python supervisor_agent.py --query "Hình phạt tội tàng trữ ma tuý?"
"""

import argparse
import asyncio
import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

# Load .env từ Lab_Assignment trước, rồi fallback lên thư mục cha (nếu key hết hạn)
_here = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_here, ".env"))
# Nếu không có API key hợp lệ ở local → thử load từ project cha (Batch02-Day9)
_parent_env = os.path.join(_here, "..", ".env")
if os.path.exists(_parent_env):
    load_dotenv(_parent_env, override=False)  # override=False: local key ưu tiên

# Thêm src vào path để import các task modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


# =============================================================================
# LLM FACTORY
# =============================================================================

def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=temperature,
        max_tokens=1024,
    )


# =============================================================================
# SHARED STATE
# =============================================================================

def _concat(left: str | None, right: str | None) -> str:
    """Reducer: nối chuỗi (dùng cho kết quả từ nhiều workers)."""
    parts = [p for p in [left, right] if p]
    return "\n\n".join(parts)


def _last(left, right):
    """Reducer: giá trị mới ghi đè."""
    return right if right is not None else left


class RAGState(TypedDict):
    # Input
    query: str

    # Supervisor analysis
    intent: Annotated[str, _last]            # "legal" | "news" | "mixed"
    needs_legal: Annotated[bool, _last]
    needs_news: Annotated[bool, _last]
    supervisor_analysis: Annotated[str, _last]

    # Worker outputs (có thể từ nhiều workers — concat)
    legal_context: Annotated[str, _concat]
    news_context: Annotated[str, _concat]
    legal_chunks: Annotated[list, _last]
    news_chunks: Annotated[list, _last]

    # Final
    final_answer: Annotated[str, _last]
    sources_used: Annotated[list, _last]


# =============================================================================
# WORKER 1: Legal Retriever Agent
# =============================================================================

def legal_retriever_worker(state: RAGState) -> dict:
    """
    Worker chuyên về văn bản pháp luật Việt Nam về ma tuý.
    Tìm kiếm trong vector store (Weaviate) và BM25 index.
    Ưu tiên các nguồn: Luật Phòng chống ma tuý 2021, BLHS 2015, Nghị định...
    """
    print("  [Worker 1] LegalRetriever đang tìm kiếm văn bản pháp luật...")

    query = state["query"]

    # Tăng cường query với ngữ cảnh pháp luật
    legal_query = f"pháp luật Việt Nam quy định {query}"

    try:
        from task9_retrieval_pipeline import retrieve

        # Lấy nhiều hơn để lọc theo type=legal
        all_chunks = retrieve(legal_query, top_k=10, use_reranking=True)
        legal_chunks = [
            c for c in all_chunks
            if c.get("metadata", {}).get("type") == "legal"
            or str(c.get("metadata", {}).get("source", "")).startswith(
                ("luat-", "nghi-dinh", "bo-luat", "thong-tu")
            )
        ]
        # Fallback: nếu lọc ra ít hơn 2, dùng tất cả
        if len(legal_chunks) < 2:
            legal_chunks = all_chunks[:5]

    except Exception as exc:
        print(f"    ⚠ Retrieval error: {exc}. Dùng mock data.")
        legal_chunks = _mock_legal_chunks()

    # Reorder để tránh lost in the middle
    reordered = _reorder_chunks(legal_chunks[:5])

    # Format context
    context_parts = []
    for i, chunk in enumerate(reordered, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source") or meta.get("title") or f"Nguồn {i}"
        section = meta.get("section_title") or meta.get("physical_index") or ""
        label = f"{source}" + (f", {section}" if section else "")
        context_parts.append(
            f"[Văn bản pháp luật {i} | {label} | Score: {chunk.get('score', 0):.3f}]\n"
            f"{chunk.get('content', '')}"
        )

    context = "\n\n---\n\n".join(context_parts)
    print(f"    ✓ Tìm được {len(reordered)} chunks pháp luật")

    return {
        "legal_context": context,
        "legal_chunks": legal_chunks[:5],
    }


# =============================================================================
# WORKER 2: News Retriever Agent
# =============================================================================

def news_retriever_worker(state: RAGState) -> dict:
    """
    Worker chuyên về tin tức nghệ sĩ Việt Nam liên quan ma tuý.
    Tìm kiếm trong news corpus đã crawl từ Task 2.
    """
    print("  [Worker 2] NewsRetriever đang tìm kiếm tin tức...")

    query = state["query"]
    news_query = f"nghệ sĩ diễn viên {query} ma tuý bị bắt"

    try:
        from task9_retrieval_pipeline import retrieve

        all_chunks = retrieve(news_query, top_k=10, use_reranking=True)
        news_chunks = [
            c for c in all_chunks
            if c.get("metadata", {}).get("type") == "news"
            or str(c.get("metadata", {}).get("source", "")).startswith("article_")
        ]
        if len(news_chunks) < 2:
            news_chunks = all_chunks[:5]

    except Exception as exc:
        print(f"    ⚠ Retrieval error: {exc}. Dùng mock data.")
        news_chunks = _mock_news_chunks()

    reordered = _reorder_chunks(news_chunks[:5])

    context_parts = []
    for i, chunk in enumerate(reordered, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source") or meta.get("title") or f"Bài báo {i}"
        date = meta.get("date") or meta.get("crawl_date") or ""
        label = f"{source}" + (f", {date}" if date else "")
        context_parts.append(
            f"[Tin tức {i} | {label} | Score: {chunk.get('score', 0):.3f}]\n"
            f"{chunk.get('content', '')}"
        )

    context = "\n\n---\n\n".join(context_parts)
    print(f"    ✓ Tìm được {len(reordered)} chunks tin tức")

    return {
        "news_context": context,
        "news_chunks": news_chunks[:5],
    }


# =============================================================================
# WORKER 3: Generator Agent
# =============================================================================

def generator_worker(state: RAGState) -> dict:
    """
    Worker tổng hợp context từ các workers khác và sinh câu trả lời có citation.
    Áp dụng:
    - Lost in the middle mitigation (chunks đã được reorder bởi workers)
    - Citation injection
    - Fallback extractive nếu không có API key
    """
    print("  [Worker 3] Generator tổng hợp và sinh câu trả lời...")

    query = state["query"]
    intent = state.get("intent", "mixed")

    # Gộp context theo intent
    context_parts = []
    if state.get("legal_context"):
        context_parts.append("=== NGUỒN PHÁP LUẬT ===\n" + state["legal_context"])
    if state.get("news_context"):
        context_parts.append("=== NGUỒN TIN TỨC ===\n" + state["news_context"])

    if not context_parts:
        return {
            "final_answer": "Không tìm thấy thông tin liên quan trong knowledge base.",
            "sources_used": [],
        }

    combined_context = "\n\n".join(context_parts)

    # Hệ thống prompt với citation requirement
    system_prompt = """Bạn là trợ lý pháp lý chuyên về pháp luật Việt Nam về ma tuý và tin tức nghệ sĩ.

Hãy trả lời câu hỏi dựa HOÀN TOÀN vào context được cung cấp.

QUY TẮC BẮT BUỘC:
1. Mỗi tuyên bố sự thật PHẢI có citation trong ngoặc vuông, ví dụ: [Luật Phòng chống ma tuý 2021, Điều 3] hoặc [VnExpress, 2024]
2. Nếu thông tin không có trong context → trả lời "Tôi không thể xác minh thông tin này từ nguồn hiện có"
3. Không suy đoán hay tổng hợp kiến thức ngoài context
4. Cấu trúc câu trả lời rõ ràng với các đề mục
5. Trả lời bằng tiếng Việt"""

    user_message = f"""Context:
{combined_context}

---

Câu hỏi: {query}

Hãy trả lời dựa vào context trên, có citation đầy đủ."""

    try:
        llm = get_llm(temperature=0.3)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])
        answer = response.content
        print(f"    ✓ Sinh được câu trả lời ({len(answer)} ký tự)")
    except Exception as exc:
        print(f"    ⚠ LLM error: {exc}. Dùng extractive fallback.")
        answer = _extractive_fallback(query, state)

    # Gộp sources
    all_chunks = (state.get("legal_chunks") or []) + (state.get("news_chunks") or [])

    return {
        "final_answer": answer,
        "sources_used": all_chunks,
    }


# =============================================================================
# SUPERVISOR AGENT
# =============================================================================

def supervisor_agent(state: RAGState) -> dict:
    """
    Supervisor phân tích câu hỏi và quyết định:
    - Intent: legal / news / mixed
    - Cần gọi workers nào
    """
    print(f"\n[Supervisor] Phân tích câu hỏi: '{state['query'][:80]}'")

    query = state["query"].lower()

    # Keyword-based intent classification (nhanh, không cần LLM call)
    legal_keywords = {
        "luật", "nghị định", "pháp luật", "hình phạt", "điều", "khoản",
        "tội", "phạt tù", "bộ luật", "quy định", "cai nghiện", "bắt buộc",
        "tiền chất", "danh mục", "chất ma tuý", "tàng trữ", "mua bán",
        "vận chuyển", "sản xuất", "hình sự", "dân sự",
    }
    news_keywords = {
        "nghệ sĩ", "diễn viên", "ca sĩ", "người mẫu", "nổi tiếng",
        "bị bắt", "bị khởi tố", "sử dụng ma tuý", "liên quan", "vụ án",
        "g-dragon", "hữu tín", "châu việt cường", "bắt giữ", "xét xử",
        "năm 2024", "năm 2023", "tin tức",
    }

    query_tokens = set(query.replace(",", " ").replace(".", " ").split())
    legal_score = len(query_tokens & legal_keywords)
    news_score = len(query_tokens & news_keywords)

    # Dùng LLM để phân loại nếu keyword không rõ ràng
    if legal_score == 0 and news_score == 0:
        try:
            llm = get_llm(temperature=0.0)
            response = llm.invoke([
                SystemMessage(content=(
                    "Bạn là classifier. Phân loại câu hỏi về ma tuý Việt Nam "
                    "vào một trong 3 nhóm: 'legal' (pháp luật), 'news' (tin tức nghệ sĩ), "
                    "hoặc 'mixed' (cả hai). Chỉ trả về đúng 1 từ."
                )),
                HumanMessage(content=state["query"]),
            ])
            intent = response.content.strip().lower()
            if intent not in {"legal", "news", "mixed"}:
                intent = "mixed"
        except Exception:
            intent = "mixed"
    elif legal_score >= news_score * 2:
        intent = "legal"
    elif news_score >= legal_score * 2:
        intent = "news"
    else:
        intent = "mixed"

    needs_legal = intent in {"legal", "mixed"}
    needs_news = intent in {"news", "mixed"}

    print(f"  → Intent: {intent} | needs_legal={needs_legal} | needs_news={needs_news}")

    analysis = (
        f"Câu hỏi liên quan đến: {'pháp luật' if needs_legal else ''}"
        f"{'+ ' if needs_legal and needs_news else ''}"
        f"{'tin tức nghệ sĩ' if needs_news else ''}. "
        f"Routing đến {('LegalRetriever, ' if needs_legal else '')}"
        f"{'NewsRetriever' if needs_news else ''}."
    )

    return {
        "intent": intent,
        "needs_legal": needs_legal,
        "needs_news": needs_news,
        "supervisor_analysis": analysis,
        "legal_context": "",
        "news_context": "",
        "legal_chunks": [],
        "news_chunks": [],
    }


def route_to_workers(state: RAGState) -> list[Send]:
    """Conditional edge: gửi tasks đến workers song song dựa trên intent."""
    tasks = []

    if state.get("needs_legal", True):
        tasks.append(Send("legal_retriever", state))

    if state.get("needs_news", True):
        tasks.append(Send("news_retriever", state))

    # Luôn luôn cần Generator
    # (Generator sẽ chạy sau khi cả hai workers xong, qua edge riêng)
    return tasks if tasks else [Send("legal_retriever", state)]


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_supervisor_graph() -> StateGraph:
    """
    Xây dựng LangGraph với Supervisor-Workers pattern.

    Topology:
        START → supervisor → [legal_retriever ∥ news_retriever] → generator → END
    """
    graph = StateGraph(RAGState)

    # Nodes
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("legal_retriever", legal_retriever_worker)
    graph.add_node("news_retriever", news_retriever_worker)
    graph.add_node("generator", generator_worker)

    # Edges
    graph.add_edge(START, "supervisor")

    # Supervisor → Workers (parallel via Send API)
    graph.add_conditional_edges("supervisor", route_to_workers)

    # Workers → Generator (sau khi tất cả workers hoàn thành)
    graph.add_edge("legal_retriever", "generator")
    graph.add_edge("news_retriever", "generator")

    graph.add_edge("generator", END)

    return graph.compile()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _reorder_chunks(chunks: list[dict]) -> list[dict]:
    """Lost-in-the-middle mitigation: quan trọng nhất ở đầu và cuối."""
    if len(chunks) <= 2:
        return chunks
    front = chunks[::2]
    back = chunks[1::2][::-1]
    return front + back


def _extractive_fallback(query: str, state: RAGState) -> str:
    """Fallback đơn giản: trích xuất câu liên quan nhất."""
    import re
    all_context = " ".join(filter(None, [
        state.get("legal_context", ""),
        state.get("news_context", ""),
    ]))
    if not all_context:
        return "Không tìm thấy thông tin liên quan."

    query_tokens = set(re.findall(r"[\wÀ-ỹ]+", query.lower()))
    sentences = re.split(r"(?<=[.!?])\s+", all_context)

    scored = []
    for sentence in sentences:
        if len(sentence) < 30:
            continue
        s_tokens = set(re.findall(r"[\wÀ-ỹ]+", sentence.lower()))
        score = len(query_tokens & s_tokens)
        if score > 0:
            scored.append((score, sentence))

    scored.sort(reverse=True)
    top = [s for _, s in scored[:3]]
    return "\n- ".join(["Thông tin trích xuất:"] + top) if top else "Không đủ thông tin."


def _mock_legal_chunks() -> list[dict]:
    """Mock data khi vector store chưa được setup."""
    return [
        {
            "content": (
                "Điều 249 BLHS 2015: Tội tàng trữ trái phép chất ma tuý. "
                "Người nào tàng trữ trái phép chất ma tuý mà không nhằm mục đích mua bán, "
                "thì bị phạt tù từ 01 năm đến 05 năm. Khoản 2: phạt tù từ 05 năm đến 10 năm "
                "nếu số lượng lớn hơn mức quy định."
            ),
            "score": 0.87,
            "metadata": {
                "source": "bo-luat-hinh-su-2015",
                "type": "legal",
                "section_title": "Điều 249",
            },
        },
        {
            "content": (
                "Điều 3 Luật Phòng, chống ma tuý 2021: Ma tuý là các chất được quy định "
                "trong danh mục các chất ma tuý do Chính phủ ban hành. "
                "Nghiêm cấm sản xuất, tàng trữ, vận chuyển, mua bán, sử dụng trái phép."
            ),
            "score": 0.82,
            "metadata": {
                "source": "luat-phong-chong-ma-tuy-2021",
                "type": "legal",
                "section_title": "Điều 3",
            },
        },
        {
            "content": (
                "Nghị định 105/2021/NĐ-CP: Quy định về biện pháp cai nghiện ma tuý tự nguyện. "
                "Người nghiện ma tuý tự nguyện đăng ký cai nghiện tại cơ sở cai nghiện "
                "được hưởng các quyền lợi theo quy định."
            ),
            "score": 0.75,
            "metadata": {
                "source": "nghi-dinh-105-2021",
                "type": "legal",
                "section_title": "Điều 15",
            },
        },
    ]


def _mock_news_chunks() -> list[dict]:
    """Mock data tin tức."""
    return [
        {
            "content": (
                "Hữu Tín bị bắt vào tháng 9/2018 vì sử dụng ma tuý tại TP.HCM. "
                "TAND TP.HCM xử phạt 7 năm 6 tháng tù giam."
            ),
            "score": 0.85,
            "metadata": {
                "source": "VnExpress",
                "type": "news",
                "date": "2019",
            },
        },
        {
            "content": (
                "Châu Việt Cường bị bắt năm 2019, liên quan đến vụ nhét tỏi vào miệng nạn nhân "
                "tử vong do dùng ma tuý cùng nhau. Bị phạt 13 năm tù về tội giết người."
            ),
            "score": 0.80,
            "metadata": {
                "source": "Thanh Niên",
                "type": "news",
                "date": "2020",
            },
        },
    ]


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_result(result: dict, query: str) -> None:
    print("\n" + "=" * 70)
    print("KẾT QUẢ CUỐI CÙNG")
    print("=" * 70)
    print(f"\n📌 Câu hỏi: {query}")
    print(f"🔍 Intent:  {result.get('intent', 'unknown')}")
    print(f"📊 Phân tích: {result.get('supervisor_analysis', '')}")
    print("\n" + "-" * 70)
    print("💬 Trả lời:\n")
    print(result.get("final_answer", "Không có câu trả lời."))
    print("\n" + "-" * 70)

    sources = result.get("sources_used", [])
    if sources:
        print(f"\n📚 Nguồn tham khảo ({len(sources)} chunks):")
        seen = set()
        for chunk in sources:
            src = chunk.get("metadata", {}).get("source", "Unknown")
            if src not in seen:
                seen.add(src)
                doc_type = chunk.get("metadata", {}).get("type", "?")
                print(f"  • [{doc_type}] {src}")

    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

async def run_query(query: str) -> dict:
    """Chạy một query qua Supervisor-Workers pipeline."""
    graph = build_supervisor_graph()

    initial_state: RAGState = {
        "query": query,
        "intent": "",
        "needs_legal": True,
        "needs_news": False,
        "supervisor_analysis": "",
        "legal_context": "",
        "news_context": "",
        "legal_chunks": [],
        "news_chunks": [],
        "final_answer": "",
        "sources_used": [],
    }

    result = await graph.ainvoke(initial_state)
    return result


async def main():
    parser = argparse.ArgumentParser(description="Supervisor-Workers RAG Agent")
    parser.add_argument("--query", type=str, help="Câu hỏi cần trả lời")
    args = parser.parse_args()

    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Nghệ sĩ nào đã bị bắt vì sử dụng ma tuý?",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện và hình phạt?",
    ]

    if args.query:
        queries = [args.query]
    else:
        queries = test_queries

    print("\n" + "=" * 70)
    print("SUPERVISOR-WORKERS MULTI-AGENT RAG SYSTEM")
    print("Day08 RAG Pipeline — Cải tiến với LangGraph")
    print("=" * 70)
    print("\nKiến trúc:")
    print("  Supervisor → [LegalRetriever ∥ NewsRetriever] → Generator")
    print()

    for query in queries:
        print(f"\n{'─' * 70}")
        result = await run_query(query)
        print_result(result, query)


if __name__ == "__main__":
    asyncio.run(main())
