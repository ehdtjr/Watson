from dotenv import load_dotenv
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote import logging
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from openai import OpenAI
from operator import itemgetter
import re
import os

load_dotenv()

strength = [50, 60, 40, 30]
health = [50, 70, 60, 40]
size = [70, 60, 50, 40]
agility = [80, 70, 60, 50]
look = [40, 50, 60, 70]
education = [60, 70, 80, 90]
iq = [50, 60, 70, 80]
mental = [60, 70, 80, 90]

# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH12-RAG")

# 채팅 기록을 저장할 메모리 초기화
# chat_history = ChatMessageHistory()

# 세션 기록을 저장할 딕셔너리
store = {}


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


# 시나리오 진행과 선택지를 추출하는 함수
def extract_scenario_and_choices(text):
    # "시나리오 진행" 부분을 추출하는 정규 표현식 패턴

    scenario_pattern = r"\*\*시나리오 진행:\*\*\s*(.*?)(?=\n\n|$)"
    scenario_match = re.search(scenario_pattern, text, re.DOTALL)

    # 선택지 부분을 추출하는 정규 표현식 패턴
    choices_pattern = r"(\d+\.\s[^\n]+)"
    choices = re.findall(choices_pattern, text)

    if "새 학기의 봄" in text or scenario_match:
        scenario_progress = scenario_match.group(
            1
        ).strip()  # "시나리오 진행" 부분만 반환
    else:
        scenario_progress = None

    dice = False

    if re.findall("주사위 입력값을 기다립니다", text):
        scenario_progress += "\n\n주사위를 굴려주세요."
        dice = True

    return scenario_progress, choices, dice


# 시작 템플릿과 시나리오 진행 템플릿 선택
def select_prompt_template(user_input, is_intro):
    if is_intro:
        return intro_prompt_template
    elif "주사위 입력값" in user_input:
        return dice_prompt
    else:
        return prompt


# 체인(Chain) 생성
def build_chain(user_input, is_intro):
    prompt_template = select_prompt_template(user_input, is_intro)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            # MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    return (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | runnable
        | prompt
        | llm
        | StrOutputParser()
    )


runnable = RunnablePassthrough.assign(
    question=lambda x: x["question"],
    strength1=lambda _: strength[0],
    health1=lambda _: health[0],
    size1=lambda _: size[0],
    agility1=lambda _: agility[0],
    look1=lambda _: look[0],
    education1=lambda _: education[0],
    iq1=lambda _: iq[0],
    mental1=lambda _: mental[0],
)

# RAG 사전 단계
# # 단계 1: 문서 로드(Load Documents)
# loader = PyMuPDFLoader("./data/Afterschool_Adventure_Time-20-52.pdf")
# docs = loader.load()

# # 단계 2: 문서 분할(Split Documents)
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
# split_documents = text_splitter.split_documents(docs)

# # 단계 3: 임베딩(Embedding) 생성
embeddings = OpenAIEmbeddings()

# # 단계 4: DB 생성(Create DB) 및 저장
# # 벡터스토어를 생성합니다. (메모리에 저장하는것이고 아직 디스크에 저장하는 것은 아님)
# vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
# # 로컬에 "MY_VECTORSTORE"라는 이름으로 데이터베이스를 저장합니다.
# vectorstore.save_local("MY_VECTORSTORE")

vectorstore = FAISS.load_local(
    "MY_VECTORSTORE", embeddings, allow_dangerous_deserialization=True
)

# 단계 5: 검색기(Retriever) 생성
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

# 시작 템플릿
intro_prompt_template = """
당신은 30년 차 TRPG 게임 사회자입니다.
현재 "방과후 탐사활동기록" 시나리오의 사회자를 맡게 되었습니다.
아래 예제1과 검색된 context를 참고하여 시나리오의 도입부를 7줄 이내로 작성해주세요.
답변은 한글로 작성하세요.

다음 사항을 준수하세요:
- 답변은 시나리오 진행으로 구성됩니다.
- 답변 형식은 [예제1]을 참고하여 작성하여 7줄 이내로 작성해주세요. 

# 예제1:
    **시나리오 진행:**
    새 학기의 봄.
    여러가지 식물들이 형형색색 학교의 화단 또는 가는 길목마다 꽃을 피우고, 꽃가루가 흩날립니다.
    이 장면이 장관일 수도 있고, 어떤 이들에게는 어떤 때 보다도 괴로운 계절일 수도 있죠.
    어느 정도 시기가 흐르고 새로운 학년, 새 반, 새 친구들. 각자 저마다의 그룹이 짜여졌을 무렵입니다.
    학교, 하면 동아리 활동에 열심인 학생들도 있겠지요. 그래요, 여러분처럼요.
    그날따라 여러분은 저마다의 이유로 동아리 활동을 하거나, 또는 부실에서 잠들었다거나, 또는 다른 게임을 했다던가의 등등의 이유로 늦게 하교하는 날이 되었습니다.

#context:
{context}

#Previous Chat History:
{chat_history}
"""

# 시나리오 진행 템플릿
prompt = """당신은 30년 차 TRPG 게임 사회자입니다.
현재 "방과후 탐사활동기록" 시나리오의 사회자를 맡게 되었습니다.
Player들의 선택지를 참고하여 시나리오를 진행하려고 합니다.
다음과 같은 검색된 context를 사용하여 시나리오를 자연스럽게 만들어주세요.
답변은 한글로 작성하세요.

다음 사항을 준수하세요:
- 답변은 상황에 대한 시나리오 진행과 3가지 선택지 제공으로 구성됩니다.
- 답변 형식은 [예제1]을 참고하여 작성해주세요. 단,시나리오 진행 도중 주사위를 굴려야 하는 상황이 온다면 답변 형식은 [예제2]를 참고하여 작성해주세요.
- 주사위 입력값이 들어오면 플레이어의 특성값을 확인하여 결과를 알려주고 시나리오를 진행합니다.

# 예제1:
    **시나리오 진행:**
    여러분은 조심스럽게 2층으로 올라가기로 합니다. 계단을 오르면서 소리가 점점 더 크게 들려옵니다. 2층에 도착하자, 소리는 복도 끝에 있는 문 뒤에서 나는 것 같습니다. 문은 살짝 열려있고, 안쪽에서 희미한 빛이 새어나오고 있습니다.

    **선택지:**
    1. 문을 열어본다.
    2. 문을 열지 않고 돌아간다.
    3. 다른 방향으로 이동한다.

# 예제2:
    **시나리오 진행:**
    여러분은 경찰과 흰 가운의 남자가 대화를 나누는 것을 발견하고, 그들의 대화를 엿듣기 위해 은밀하게 접근하기로 합니다. 주변은 어둡고 조용하여 작은 소리도 쉽게 들릴 수 있는 상황입니다. 은밀하게 접근하기 위해서는 은밀행동 판정이 필요합니다.

    주사위를 굴려 은밀행동 판정을 해주세요.

    **주사위 입력값을 기다립니다.**

#Question: 
{question} 

#Previous Chat History:
{chat_history}

#Context: 
{context} 

#Answer:"""

# 주사위 템플릿
dice_prompt = """
당신은 30년 차 TRPG 게임 사회자입니다.
현재 "방과후 탐사활동기록" 시나리오의 사회자를 맡게 되었습니다.
답변은 한글로 작성하세요.

다음 사항을 준수하세요:
- [Characteristic value]를 참고하여 주사위 판정을 한 후, 이전 대화를 이어서 진행해주세요.
- 답변은 주사위 결과 대한 시나리오 진행과 3가지 선택지 제공으로 구성됩니다.
- 답변 형식은 [예제1]을 참고하여 작성해주세요.

#Characteristic value:
  Strength: {strength1}, Health: {health1}, Size: {size1}, Agility: {agility1}, Look: {look1}, Education: {education1}, IQ: {iq1}, Mental: {mental1}

# 예제1:
    **시나리오 진행:**
    은밀행동 판정에 성공했습니다! 당신은 경찰과 흰 가운의 남자가 대화를 나누는 것을 엿들었습니다. 그들은 무언가를 교환하고 있는 것 같습니다.

    **선택지:**
    1. 대화를 더 듣기 위해 더 가까이 접근한다.
    2. 대화를 더 듣기 위해 뒤로 물러난다.
    3. 다른 방향으로 이동한다.

#Previous Chat History:
{chat_history}
    
#Question: 
{question} 

#Answer:
"""

# 단계 7: 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

st.title("방과후 탐사활동기록")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.is_intro = True

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

        chain = build_chain(user_input, st.session_state.is_intro)
        st.session_state.is_intro = False  # 이후부터는 시나리오 진행으로 전환

        # 대화를 기록하는 RAG 체인 생성
        rag_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,  # 세션 기록을 가져오는 함수
            input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
            history_messages_key="chat_history",  # 기록 메시지의 키
        )

        # AI 응답을 가져옵니다.
        response = rag_with_history.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": "rag123"}},
        )

        # 시나리오 진행과 선택지를 추출합니다.(dice가 true일 경우 주사위 굴리기?)
        scenario_progress, choices, dice = extract_scenario_and_choices(response)

        print(scenario_progress)
        print(choices)
        print(dice)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display assistant message in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

print(store)
