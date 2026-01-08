import streamlit as st
from rag import query_rag

st.title("Tunisian Archaeological Sites Chatbot")

# Optional conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

user_query = st.text_input("Ask a question about archaeological sites in Tunisia:")

if user_query:
    # Append to history
    st.session_state.history.append(f"User: {user_query}")

    # Get RAG response
    response_text, sources = query_rag(user_query)

    # Append to history
    st.session_state.history.append(f"Bot: {response_text}")

    # Display
    st.write("Answer:", response_text)

    # Sources expander
    with st.expander("Sources"):
        if sources:
            for meta in sources:
                st.write(f"- {meta['site']} ({meta.get('delegation', 'N/A')}): {meta['source']}")

        else:
            st.write("No sources found.")

    # History expander
    with st.expander("Conversation History"):
        st.write("\n".join(st.session_state.history))