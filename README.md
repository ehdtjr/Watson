# TRPG AI Game Master

TRPG 게임의 사회자 역할을 대체하는 AI입니다. LangChain과 Streamlit을 활용하여 실시간으로 게임을 진행하고, 플레이어와 상호작용할 수 있는 기능을 제공합니다.

## 목차

- [소개](#소개)
- [기술 스택](#기술-스택)
- [설치](#설치)
- [사용법](#사용법)
- [실행](#실행)


## 소개

TRPG AI Game Master는 전통적인 TRPG(Tabletop Role-Playing Game)에서 사회자의 역할을 인공지능으로 대체하는 프로젝트입니다. 이 AI는 플레이어와 상호작용하고, 스토리를 생성하며, 게임의 진행을 돕습니다. 주요 기능은 다음과 같습니다:
- 플레이어와의 대화 및 상호작용
- 게임 스토리 생성 및 전개
- 실시간 게임 진행 및 상태 관리

## 기술 스택

이 프로젝트에서 사용되는 주요 기술 스택은 다음과 같습니다:
- [Python](https://www.python.org/): 주요 프로그래밍 언어
- [LangChain](https://langchain.com/): 언어 모델 체인 프레임워크
- [Streamlit](https://streamlit.io/): 웹 애플리케이션 프레임워크
- [OpenAI GPT-4](https://openai.com/research/gpt-4): 언어 모델

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
```

## 사용법
.env 파일과 .streamlit/secrets.toml 에서 프로젝트를 실행하는데 필요한 API_KEY를 지정해줍니다.

# OPENAI_API_KEY
[open ai](https://platform.openai.com/api-keys)

# LANGCHAIN_API_KEY
[langsmith](https://smith.langchain.com/o/77c5a6a5-2792-56e1-ac37-fe1d529f8673/settings)



## 실행

프로젝트를 실행하는 방법을 설명합니다.

```bash
# Streamlit 애플리케이션 실행
streamlit run app.py
