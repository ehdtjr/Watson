# TRPG AI Game Master

TRPG 게임의 사회자 역할을 대체하는 AI입니다. LangChain과 Streamlit을 활용하여 실시간으로 게임을 진행하고, 플레이어가 선택지를 선택함으로써 자유롭게 시나리오와 상호작용할 수 있는 기능을 제공합니다.

## 목차

- [소개](#소개)
- [기술 스택](#기술-스택)
- [설치](#설치)
- [사용법](#사용법)
- [실행](#실행)
- [핵심 기능](#핵심-기능)

## 소개

TRPG AI Game Master는 전통적인 TRPG(Tabletop Role-Playing Game)에서 사회자의 역할을 인공지능으로 대체하는 프로젝트입니다. 이 AI는 플레이어와 상호작용하고, 스토리를 생성하며, 게임의 진행을 돕습니다. 주요 기능은 다음과 같습니다:
- 플레이어와의 대화 및 상호작용
- 게임 스토리 생성 및 전개
- 실시간 게임 진행 및 상태 관리

이 프로젝트는 TRPG 게임의 시나리오 내용을 벡터 데이터베이스(Vector DB)에 인덱싱한 후 이를 검색(Retriever)하여 사용하는 RAG(Retrieval-Augmented Generation) 방식을 사용합니다. 또한, 상황별 프롬프트 템플릿을 선택할 수 있도록 프롬프트 엔지니어링을 적용하였으며, 프롬프트 템플릿 내에는 벡터 DB에 저장된 내용을 context로 가져오고, 이전 대화 내용을 chat_history로 불러와 시나리오가 자연스럽게 이어지도록 설계되었습니다.

## 기술 스택

이 프로젝트에서 사용되는 주요 기술 스택은 다음과 같습니다:
- [Python](https://www.python.org/): 주요 프로그래밍 언어
- [LangChain](https://langchain.com/): 언어 모델 chain 프레임워크
- [LangSmith](https://www.langchain.com/langsmith): chain 추적을 통한 효율적인 디버깅 제공
- [Streamlit](https://streamlit.io/): 웹 애플리케이션 프레임워크
- [OpenAI GPT-4o](https://openai.com/research/gpt-4): 언어 모델
- [Vector DB(FAISS)](https://www.vector-db.com/): 시나리오 인덱싱 및 검색을 위한 벡터 데이터베이스

## 설치

프로젝트를 설치하고 실행하는데 필요한 의존성을 설치합니다.

```bash
# 프로젝트 생성 및 진입
mkdir [프로젝트 파일명]
cd [프로젝트 파일명]

# 클론 명령어
git clone https://github.com/ehdtjr/Watson.git

# 가상 환경 생성
python -m venv [가상환경명]

# 가상 환경 활성화
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 필요한 의존성 설치
pip install -r requirements.txt


## 사용법
.env 파일과 .streamlit/secrets.toml 에서 프로젝트를 실행하는데 필요한 API_KEY를 지정해줍니다.

- [OPENAI_API_KEY](https://platform.openai.com/api-keys) : GPT를 사용하기 위한 key 발급
- [LANGCHAIN_API_KEY](https://smith.langchain.com/o/77c5a6a5-2792-56e1-ac37-fe1d529f8673/settings) : langsmith를 활용한 답변 생성 과정을 추적하기 위한 key 발급



## 실행

프로젝트를 실행하는 방법을 설명합니다.

```bash
# Streamlit 애플리케이션 실행
streamlit run main.py
```

## 핵심 기능

### 시나리오 인덱싱 및 검색

TRPG AI Game Master는 시나리오 내용을 벡터 데이터베이스에 인덱싱하여 저장합니다. 이를 통해 필요한 시나리오 정보를 실시간으로 검색하고, 플레이어의 질문이나 상황에 맞는 답변을 제공합니다.

### 프롬프트 엔지니어링

상황에 따라 다른 프롬프트 템플릿을 선택하여 게임의 진행을 돕습니다. 프롬프트 템플릿은 벡터 DB에서 검색된 시나리오 내용을 컨텍스트로 가져오며, 이전 대화 내용을 chat_history로 포함하여 자연스럽게 이야기가 이어질 수 있도록 합니다.

이 프로젝트를 통해 전통적인 TRPG 게임의 사회자 역할을 AI가 대체하여, 더 많은 사람들이 쉽게 TRPG를 즐길 수 있기를 바랍니다.

