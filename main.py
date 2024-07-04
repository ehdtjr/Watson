import streamlit as st
from langchain_openai import ChatOpenAI
from openai import OpenAI

# from langchain_teddynote import logging

# 플레이어 정보 정의
job = ["형사", "형사", "교수", "탐정"]
strengt = [50, 60, 40, 30]
health = [50, 70, 60, 40]
size = [70, 60, 50, 40]
agility = [80, 70, 60, 50]
look = [40, 50, 60, 70]
education = [60, 70, 80, 90]
iq = [50, 60, 70, 80]
mental = [60, 70, 80, 90]

# 프롬프트 템플릿 정의
prompt_template = """
You are the host to help you progress the TRPG game. The scenario is a call from Ktulu. Help the story flow naturally to match each player's job and ability.
You should never directly intervene in the scenario and help the scenario continue naturally by giving players natural options.
Please answer in Korean.
The information below is the occupation and characteristics of each player.
As for the questions after the initial description of the scenario, just explain the situation and options for each player and don't say anything else.

characteristics
Player 1: {job1}, Strength: {strengt1}, Health: {health1}, Size: {size1}, Agility: {agility1}, Look: {look1}, Education: {education1}, IQ: {iq1}, Mental: {mental1}
Player 2: {job2}, Strength: {strengt2}, Health: {health2}, Size: {size2}, Agility: {agility2}, Look: {look2}, Education: {education2}, IQ: {iq2}, Mental: {mental2}
Player 3: {job3}, Strength: {strengt3}, Health: {health3}, Size: {size3}, Agility: {agility3}, Look: {look3}, Education: {education3}, IQ: {iq3}, Mental: {mental3}
Player 4: {job4}, Strength: {strengt4}, Health: {health4}, Size: {size4}, Agility: {agility4}, Look: {look4}, Education: {education4}, IQ: {iq4}, Mental: {mental4}

Question: {question}
Answer:
"""

# OpenAI API 클라이언트를 초기화합니다.
# llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
llm = ChatOpenAI(
    model_name="xionic-1-72b-20240610",
    base_url="https://sionic.chat/v1/",
    api_key="934c4bbc-c384-4bea-af82-1450d7f8128d",
)

st.title("Call of Cthulhu")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.spinner("Loading AI..."):
    if user_input := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)

        # 프롬프트를 생성합니다.
        prompt = prompt_template.format(
            job1=job[0],
            strengt1=strengt[0],
            health1=health[0],
            size1=size[0],
            agility1=agility[0],
            look1=look[0],
            education1=education[0],
            iq1=iq[0],
            mental1=mental[0],
            job2=job[1],
            strengt2=strengt[1],
            health2=health[1],
            size2=size[1],
            agility2=agility[1],
            look2=look[1],
            education2=education[1],
            iq2=iq[1],
            mental2=mental[1],
            job3=job[2],
            strengt3=strengt[2],
            health3=health[2],
            size3=size[2],
            agility3=agility[2],
            look3=look[2],
            education3=education[2],
            iq3=iq[2],
            mental3=mental[2],
            job4=job[3],
            strengt4=strengt[3],
            health4=health[3],
            size4=size[3],
            agility4=agility[3],
            look4=look[3],
            education4=education[3],
            iq4=iq[3],
            mental4=mental[3],
            question=user_input,
        )

        # AI 응답을 가져옵니다.
        # response = llm.invoke(user_input).content
        response = llm.invoke(user_input)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.content})
        # Display assistant message in chat message container
        with st.chat_message("assistant"):
            st.markdown(response.content)
