import os
import streamlit as st
from langchain_openai import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# .env 파일을 로드합니다.
load_dotenv()

# from langchain_teddynote import logging
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("watson")

# 플레이어 정보 정의
job = ["형사", "형사", "교수", "탐정"]
strength = [50, 60, 40, 30]
health = [50, 70, 60, 40]
size = [70, 60, 50, 40]
agility = [80, 70, 60, 50]
look = [40, 50, 60, 70]
education = [60, 70, 80, 90]
iq = [50, 60, 70, 80]
mental = [60, 70, 80, 90]

# 프롬프트 템플릿 정의
prompt_template = """
당신은 20년차 TRPG게임 사회자입니다. Playere들의 선택에 따라 시나리오를 자연스럽게 진행하여주세요.
각 Player들의 직업과 능력치는 특성치에 적혀있으며 상황에 따라 적절하게 활용해주세요.
답변은 한글로 작성하세요.

다음 사항을 준수하세요:
- 답변은 진행상황에 대한 짧은 요약 및 Player들에게 제안할 선택지로만 구성하세요.

다음 내용은 Player들의 특성치입니다.:
Player 1: {job1}, Strength: {strength1}, Health: {health1}, Size: {size1}, Agility: {agility1}, Look: {look1}, Education: {education1}, IQ: {iq1}, Mental: {mental1}
Player 2: {job2}, Strength: {strength2}, Health: {health2}, Size: {size2}, Agility: {agility2}, Look: {look2}, Education: {education2}, IQ: {iq2}, Mental: {mental2}
Player 3: {job3}, Strength: {strength3}, Health: {health3}, Size: {size3}, Agility: {agility3}, Look: {look3}, Education: {education3}, IQ: {iq3}, Mental: {mental3}
Player 4: {job4}, Strength: {strength4}, Health: {health4}, Size: {size4}, Agility: {agility4}, Look: {look4}, Education: {education4}, IQ: {iq4}, Mental: {mental4}

# 질문: {question}

# 답변:
"""


class StreamCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)


# LLM 및 출력 파서 설정
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    streaming=True,
    verbose=True,
    callbacks=[StreamCallback()],
)
output_parser = StrOutputParser()

# 채팅 기록을 저장할 메모리 초기화
chat_history = ChatMessageHistory()

# 프롬프트 설정
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Runnable 객체 생성
runnable = RunnablePassthrough.assign(
    history=lambda _: chat_history.messages,
    question=lambda x: x["question"],
    job1=lambda _: job[0],
    strength1=lambda _: strength[0],
    health1=lambda _: health[0],
    size1=lambda _: size[0],
    agility1=lambda _: agility[0],
    look1=lambda _: look[0],
    education1=lambda _: education[0],
    iq1=lambda _: iq[0],
    mental1=lambda _: mental[0],
    job2=lambda _: job[1],
    strength2=lambda _: strength[1],
    health2=lambda _: health[1],
    size2=lambda _: size[1],
    agility2=lambda _: agility[1],
    look2=lambda _: look[1],
    education2=lambda _: education[1],
    iq2=lambda _: iq[1],
    mental2=lambda _: mental[1],
    job3=lambda _: job[2],
    strength3=lambda _: strength[2],
    health3=lambda _: health[2],
    size3=lambda _: size[2],
    agility3=lambda _: agility[2],
    look3=lambda _: look[2],
    education3=lambda _: education[2],
    iq3=lambda _: iq[2],
    mental3=lambda _: mental[2],
    job4=lambda _: job[3],
    strength4=lambda _: strength[3],
    health4=lambda _: health[3],
    size4=lambda _: size[3],
    agility4=lambda _: agility[3],
    look4=lambda _: look[3],
    education4=lambda _: education[3],
    iq4=lambda _: iq[3],
    mental4=lambda _: mental[3],
)

# LCEL 체인 구성
chain = runnable | prompt | llm | output_parser

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
        formatted_prompt = prompt_template.format(
            job1=job[0],
            strength1=strength[0],
            health1=health[0],
            size1=size[0],
            agility1=agility[0],
            look1=look[0],
            education1=education[0],
            iq1=iq[0],
            mental1=mental[0],
            job2=job[1],
            strength2=strength[1],
            health2=health[1],
            size2=size[1],
            agility2=agility[1],
            look2=look[1],
            education2=education[1],
            iq2=iq[1],
            mental2=mental[1],
            job3=job[2],
            strength3=strength[2],
            health3=health[2],
            size3=size[2],
            agility3=agility[2],
            look3=look[2],
            education3=education[2],
            iq3=iq[2],
            mental3=mental[2],
            job4=job[3],
            strength4=strength[3],
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
        response = chain.invoke({"question": user_input})

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Display assistant message in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
