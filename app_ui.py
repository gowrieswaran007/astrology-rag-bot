"""
app_ui.py
-----------
Streamlit frontend - a real-time astrology chat application.

Collects the user's Name, Date of Birth, Time of Birth, Place of Birth
(once, in the sidebar), then lets them ask questions in a chat interface.
Every question is sent to the FastAPI backend, which runs the multi-agent
system and returns a grounded answer.

Setup:
    1. Start the API first (in one terminal):
         uvicorn api:app --reload --port 8000
    2. Then start this UI (in a second terminal):
         streamlit run app_ui.py
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Astrology Assistant", page_icon="🔮")
st.title("🔮 Astrology Assistant")
st.caption("Multi-Agent RAG System | KP Astrology + Beginners Guide | qwen3:4b")

# ---------------- Sidebar: Birth Details (collected once) ----------------
with st.sidebar:
    st.header("Your Birth Details")
    st.caption("Needed only for personal predictions. General astrology questions work without this.")
    name = st.text_input("Name", value="")
    dob = st.date_input("Date of Birth")
    tob = st.time_input("Time of Birth")
    place = st.text_input("Place of Birth", placeholder="e.g. Chennai, India")

    st.divider()
    st.caption("💡 Example questions:")
    st.caption("- What is the KP system in astrology?")
    st.caption("- What does my ascendant say about my personality?")
    st.caption("- Explain the 12 houses in astrology")

# ---------------- Chat history ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- Chat input ----------------
question = st.chat_input("Ask an astrology question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the agents..."):
            payload = {
                "name": name or None,
                "dob": str(dob) if dob else None,
                "tob": tob.strftime("%H:%M") if tob else None,
                "place": place or None,
                "question": question,
            }

            try:
                response = requests.post(API_URL, json=payload, timeout=300)
                response.raise_for_status()
                data = response.json()

                answer = data["answer"]
                intent = data["intent"]
                chart_summary = data.get("chart_summary")
                sources = data.get("sources")

                st.caption(f"🧭 Detected intent: **{intent}**")

                if chart_summary:
                    with st.expander("📊 Your Birth Chart (calculated)"):
                        st.text(chart_summary)

                st.markdown(answer)

                if sources:
                    st.caption(f"📚 Sources: {', '.join(sources[:3])}")

            except requests.exceptions.ConnectionError:
                answer = "⚠️ Could not connect to the API. Make sure 'uvicorn api:app --reload --port 8000' is running in another terminal."
                st.error(answer)
            except Exception as e:
                answer = f"⚠️ Error: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})