import streamlit as st
from dotenv import load_dotenv

load_dotenv()

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

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


class StreamCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)

# LLM 및 출력 파서 설정
llm = ChatOpenAI(
    model="gpt-4o",
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
        (
            "system",
            prompt_template.format(
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
                question="{question}",
            ),
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Runnable 객체 생성
runnable = RunnablePassthrough.assign(
    history=lambda _: chat_history.messages,
    # context=lambda x: ensemble_retriever.invoke(x["question"])
)
# LCEL 체인 구성
chain = runnable | prompt | llm | output_parser


def rag_chain(question):
    response = chain.invoke({"question": question})
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)
    return response


# 텔레그램 봇 설정 및 핸들러 정의
import telegram
import os
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram.constants import ChatAction, ParseMode

# 텔레그램 봇 토큰을 환경 변수에서 가져옵니다.
bot = telegram.Bot(os.getenv("BOT_TOKEN"))

# 날짜와 시간 함수
from datetime import datetime

get_current_datetime = lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# RAG 체인을 사용하여 답변을 생성하는 함수
def generate_response(message):
    return rag_chain(message)


# 텍스트를 Telegram Markdown V2 형식으로 이스케이프하는 함수
escape_markdown_v2 = lambda text: "".join(
    ["\\" + char if char in r"\`*_{}[]()#+-.!|>=" else char for char in text]
)

# 응답을 나누어 마크다운 V2 형식으로 포맷팅하는 함수
split_response = lambda response: [
    escape_markdown_v2(part) if i % 2 == 0 else f"```{part}```"
    for i, part in enumerate(response.split("```"))
]


# 봇의 /start 명령에 대한 핸들러 함수
async def start(update, context):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="안녕하세요, 테스트중인 챗봇입니다!! 🧑‍💻",
    )


# 텔레그램 메시지에 대한 응답을 생성하는 핸들러 함수
async def chat_bot(update, context):
    message = update.message.text
    user = update.message.from_user
    user_identifier = (
        user.username
        if user.username
        else f"{user.first_name} {user.last_name if user.last_name else ''}"
    )
    date_time = get_current_datetime()

    print(f"\n[User_Info] uid: {user.id}, name: {user_identifier}, date: {date_time}")
    print(f"\n[Question] {message}\n[Answer]\n")

    loading_message = await context.bot.send_message(
        chat_id=update.effective_chat.id, text="처리 중입니다... 🧑‍💻"
    )
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    try:
        response = generate_response(message)
    except Exception as e:
        await context.bot.delete_message(
            chat_id=update.effective_chat.id, message_id=loading_message.message_id
        )
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=f"오류가 발생했습니다: {str(e)}"
        )
        return

    await context.bot.delete_message(
        chat_id=update.effective_chat.id, message_id=loading_message.message_id
    )
    formatted_response_parts = split_response(response)

    for part in formatted_response_parts:
        if part.strip():  # part가 비어있지 않은 경우에만 메시지 전송
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=part,
                parse_mode=ParseMode.MARKDOWN_V2,
            )


# 텔레그램 봇 애플리케이션 생성 및 핸들러 추가
application = Application.builder().token(os.getenv("BOT_TOKEN")).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_bot))

# 봇 실행
application.run_polling()
