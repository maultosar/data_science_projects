import streamlit as st
from agent import PsyAgent
from langchain_core.messages import HumanMessage

st.title("Psy AI")

config = {"configurable": {"thread_id": "1"}}

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.agent = PsyAgent(config, knowledge_retrieval=False, debug=True)

session_agent = st.session_state.agent
session_messages = st.session_state.messages

for message in session_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("How can I help you?"):
    
    with st.chat_message("user"):
        st.markdown(user_input)
    session_messages.append({"role": "user", "content": user_input})

    user_prompt = HumanMessage(content=user_input)
    
    if "last_node" not in session_agent.graph.get_state(config).values:
        with st.spinner('Thinking...'):
            for event in session_agent.graph.stream({"messages": [user_prompt]}, config, stream_mode="values"):
                final_response = event

            with st.chat_message("assistant"):
                st.markdown(final_response["messages"][-1].content)
            session_messages.append({"role": "assistant", "content": final_response["messages"][-1].content})

    else:
        last_node = session_agent.graph.get_state(config).values["last_node"]
        session_agent.graph.update_state(config, {"messages": [user_prompt]}, as_node=last_node)
        
        with st.spinner('Thinking...'):
            for event in session_agent.graph.stream(None, config, stream_mode="values"):
                final_response = event
            
            with st.chat_message("assistant"):
                st.write(final_response["messages"][-1].content)
            session_messages.append({"role": "assistant", "content": final_response["messages"][-1].content})