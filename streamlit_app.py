import streamlit as st
import sys
sys.path.append('.')
from utils import chat

st.title("Basketball RL Project Chatbot")
st.write("Ask me anything about Xavier Beltran's Multi-Agent RL Basketball Simulation!")

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the project..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat(prompt, st.session_state.conversation_history)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})