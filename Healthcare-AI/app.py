import streamlit as st
import requests

MODEL = "llama3.2:3b"
API_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """
You are a beginner-friendly healthcare education assistant.
Provide general health information, not medical diagnoses.
Never prescribe medicines or dosages.
Mention uncertainty and recommend consulting qualified clinicians.
For severe breathing difficulty, chest pain, stroke symptoms,
unconsciousness or other emergencies, advise immediate emergency care.
Use simple language.
"""

st.set_page_config(page_title="Healthcare AI Assistant")
st.title("Healthcare AI Assistant")
st.caption("Local AI Healthcare Learning Prototype")

st.warning("Educational prototype only. Do not enter real patient data.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask a healthcare question")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.write(prompt)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ] + st.session_state.messages[-10:]

    with st.chat_message("assistant"):
        try:
            response = requests.post(
                API_URL,
                json={
                    "model": MODEL,
                    "messages": messages,
                    "stream": False
                },
                timeout=180
            )
            response.raise_for_status()

            answer = response.json()["message"]["content"]
            st.write(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except requests.exceptions.ConnectionError:
            st.error("Ollama is not running. Please start Ollama.")
        except Exception as e:
            st.error(f"Error: {e}")
