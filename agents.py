"""
agents.py
-----------
Multi-Agent system using LangGraph (open source, part of LangChain).

4 Agents working together:

  1. ROUTER AGENT      - reads the question, decides: is this a GENERAL
                          astrology knowledge question, or a PERSONAL
                          prediction question that needs the user's birth chart?

  2. CHART AGENT        - if personal: calculates the real birth chart
                          from DOB/Time/Place using chart_calculator.py

  3. KNOWLEDGE AGENT    - retrieves relevant passages from the ingested
                          astrology PDFs (KP System + Astrology for
                          Beginners) using the existing ChromaDB (RAG)

  4. SYNTHESIS AGENT    - combines chart data + retrieved book knowledge +
                          the user's question, and asks qwen3:0.6b to
                          produce the final grounded answer

Flow:
    START -> router -> [chart_agent if personal] -> knowledge_agent -> synthesis -> END
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

from chart_calculator import calculate_birth_chart, chart_to_text

CHROMA_DB_DIR = "chroma_db_astrology"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:0.6b"
TOP_K = 5


# ---------------- Shared State ----------------
class AgentState(TypedDict):
    name: Optional[str]
    dob: Optional[str]
    tob: Optional[str]
    place: Optional[str]
    question: str
    intent: Optional[str]          # "general" or "personal"
    chart_summary: Optional[str]
    retrieved_context: Optional[str]
    sources: Optional[list]
    answer: Optional[str]
    error: Optional[str]


# ---------------- Shared resources (loaded once) ----------------
_embeddings = OllamaEmbeddings(model=EMBED_MODEL)
_vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=_embeddings)
_retriever = _vectorstore.as_retriever(search_kwargs={"k": TOP_K})
_llm = ChatOllama(model=CHAT_MODEL, temperature=0.3)


# ---------------- Agent 1: Router ----------------
def router_agent(state: AgentState) -> AgentState:
    """Decides whether birth-chart calculation is needed for this question.
    If the user has filled in their birth details (DOB, time, place),
    ALL agents run for every question - the chart is always calculated
    and used alongside the book knowledge. Only when birth details are
    missing does it fall back to pure general-knowledge mode."""

    has_birth_details = bool(state.get("dob") and state.get("tob") and state.get("place"))

    state["intent"] = "personal" if has_birth_details else "general"
    return state


# ---------------- Agent 2: Chart Calculator ----------------
def chart_agent(state: AgentState) -> AgentState:
    """Only runs when intent == personal. Computes the real birth chart."""
    try:
        chart = calculate_birth_chart(
            name=state.get("name") or "User",
            dob=state["dob"],
            tob=state["tob"],
            place=state["place"],
        )
        state["chart_summary"] = chart_to_text(chart)
    except Exception as e:
        state["error"] = f"Could not calculate birth chart: {e}"
        state["chart_summary"] = None
    return state


def route_after_router(state: AgentState):
    """Conditional edge: go to chart_agent only if personal intent."""
    return "chart_agent" if state["intent"] == "personal" else "knowledge_agent"


# ---------------- Agent 3: Knowledge Retrieval (RAG) ----------------
def knowledge_agent(state: AgentState) -> AgentState:
    """Retrieves relevant passages from the astrology books."""
    query = state["question"]
    # If we have a chart, enrich the retrieval query with sign/planet info
    if state.get("chart_summary"):
        query = f"{state['question']} {state['chart_summary'][:200]}"

    docs = _retriever.invoke(query)
    context = "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source')}, Page: {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )
    state["retrieved_context"] = context
    state["sources"] = [
        f"{d.metadata.get('source')} p.{d.metadata.get('page')}" for d in docs
    ]
    return state


# ---------------- Agent 4: Synthesis ----------------
def synthesis_agent(state: AgentState) -> AgentState:
    """Combines everything and generates the final answer."""

    if state["intent"] == "personal" and state.get("chart_summary"):
        prompt = f"""You are an expert Vedic/KP astrologer.

Use the person's REAL birth chart below (calculated using their exact
birth date, time, and place) together with the reference book knowledge
to answer their question. Ground your interpretation in the book content
where relevant. Be clear that this is an astrological interpretation.

BIRTH CHART:
{state['chart_summary']}

REFERENCE BOOK KNOWLEDGE:
{state['retrieved_context']}

QUESTION: {state['question']}

ANSWER:"""
    else:
        prompt = f"""You are an astrology teacher. Answer the question using
ONLY the reference book knowledge below. If the answer isn't in the
content, say you don't have that information.

REFERENCE BOOK KNOWLEDGE:
{state['retrieved_context']}

QUESTION: {state['question']}

ANSWER:"""

    response = _llm.invoke(prompt)
    state["answer"] = response.content
    return state


# ---------------- Build the Graph ----------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_agent)
    graph.add_node("chart_agent", chart_agent)
    graph.add_node("knowledge_agent", knowledge_agent)
    graph.add_node("synthesis", synthesis_agent)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"chart_agent": "chart_agent", "knowledge_agent": "knowledge_agent"},
    )
    graph.add_edge("chart_agent", "knowledge_agent")
    graph.add_edge("knowledge_agent", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


# Compiled once, reused across requests
astrology_graph = build_graph()


if __name__ == "__main__":
    # Quick manual test
    result = astrology_graph.invoke({
        "name": "Test User",
        "dob": "1995-08-15",
        "tob": "14:30",
        "place": "Chennai, India",
        "question": "What does my ascendant say about my personality?",
    })
    print("Intent:", result["intent"])
    print("Answer:", result["answer"])