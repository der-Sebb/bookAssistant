import streamlit as st
from streamlit_chat import message
from book_assistant import BookAssistant

st.set_page_config(page_title="Library")

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].answer(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

        st.session_state["user_input"] = ""

def page(model_type, vector_database, collection_name):
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = BookAssistant(model_type, vector_database, collection_name)

    st.header("BookAssistant")

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    model_type = "mistral"
    vector_database = "./book_vec_db"
    collection_name = "books"
    page(model_type, vector_database, collection_name)