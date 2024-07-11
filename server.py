from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from operator import itemgetter
import streamlit as st
import time
import json
import os

# from logger import logger
# logger.switchLevel("INFO")


BASE_URL = "https://api.openai.com/v1"
API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-3.5-turbo"

TITLE = "메이플 FAQ 문의 채팅"

SYS_PROMPT = """
당신은 똑똑하고 친절한 '메이플스토리 게임의 FAQ 채팅 상담가'입니다.
다음과 관련된 문의에 대해 상담해줄 수 있습니다.

- 가입 및 탈퇴 관련
- 보호자 동의 및 본인인증 관련
- 계정 보호 및 보안 설정 관련
- 결제 및 넥슨캐시 관련
- 게임 내 오류 및 문제 해결
- 아이템 및 캐릭터 관련
- 퀘스트 및 게임 진행 관련
- 시스템 및 기술적 문제
- 기타 문의 및 건의사항
"""

SYS_PROMPT2 = """
당신은 객관식문제를 푸는 똑똑한 선택 Agent입니다.
주어지는 사용자의 문제에 답변하시오.
"""

# PROMPT1 = """
# Question : {question}

# 당신은 Question에 대해 검색이 필요한 경우, '!AA'이라고 출력합니다.
# 검색이 불필요한 경우, '!BB'이라고 출력합니다.
# 항상 '!AA' 또는 '!BB' 이라고 출력합니다.
# 그 외 아무것도 하지 않습니다.
# """

PROMPT1 = """
Question : {question}

Quiz : 
당신은 Question에 대해 검색이 필요한 경우, '!AA'이라고 출력합니다.
검색이 불필요한 경우, '!BB'이라고 출력합니다.
항상 '!AA' 또는 '!BB' 이라고 출력합니다.
그 외 아무것도 하지 않습니다.
"""

# PROMPT1_2 = """
# Question : {question}

# 당신은 '[검색된 FAQ 리스트]'를 활용하여 Question에 대한 요청을 수행하기에 충분하다면, '!CC'이라고 출력합니다.
# 수행하기에는 조금이라도 정보가 부족하다면, '!DD'이라고 출력합니다.
# 항상 '!DD' 또는 '!CC' 이라고 출력합니다. 그리고 그렇게 판단한 이유를 '[검색된 FAQ 리스트]'를 포함해서 이야기합니다.
# 그 외 아무것도 하지 않습니다.
# """

PROMPT1_2 = """
Question : {question}

당신은 '[검색된 FAQ 리스트]'를 활용하여 Question에 대한 요청을 수행할 수 있다면, '!CC'이라고 출력합니다.
수행하기에는 조금이라도 정보가 부족하다면, '!DD'이라고 출력합니다.
항상 '!DD' 또는 '!CC' 이라고 출력합니다.
그 외 아무것도 하지 않습니다.
"""

# PROMPT1 = """
# Question : {question}

# 당신은 Question이 이전 답변에 대하여 추가적인 정보 요청이거나 
# 검색이 불필요한 요청이라면, '!일반'이라고 출력하고 그렇게 판단한 이유를 짧게 출력합니다.
# 당신은 Question에 대해 검색이 필요한 경우, '!검색'이라고 출력하고 그렇게 판단한 이유를 짧게 출력합니다.
# 그 외 아무것도 하지 않습니다.
# """

PROMPT2 = """
Question : {question}

Question을 짧은 검색 문구로 변환하여 출력하시오.
"""

RAG_PROMPT_TEMPLATE = """
검색결과:
{result}

검색어:
{search}

사용자 질문:
{question}

검색결과로 검색어와 사용자 질문에 충분히 답변할 수 있는 경우에만 답변하시오. 
답변하기 충분하지 않은 경우 사용자에게 더 자세한 질문을 요청하세요.
그리고 마지막에 검색어와 검색결과 Path를 출력하세요.
"""


def get_log_fn(log_msg, include_value=True):
    def log_fn(value):
        if include_value:
            logger.info(f"{log_msg}{value}")
        else:
            logger.info(f"{log_msg}")
        return value
    return log_fn


def get_interception_fn(store_key):
    def interception_fn(value):
        global global_dict
        global_dict[store_key] = value
        return value
    return interception_fn

        

def format_docs(docs):
    global data_dic
    return_str = ["[검색된 FAQ 리스트] : (Path, Q, A)", ]
    for i, doc in enumerate(docs, start=1):
        title = doc.page_content
        body = data_dic[title]["body"].strip()
        metadata = data_dic[title]["metadata"]
        
        one = f"[ {i:02d} ] \nPath : {metadata} \nQ : {title} \nA : \n{body}"
        return_str.append(one)
    return '\n\n---\n\n'.join(return_str)


# 브라우저 탭 이름 # 먼저 와야됨
st.set_page_config(page_title=TITLE)


if "global_dict" not in st.session_state:
    global_dict = {}
    st.session_state['global_dict'] = global_dict
global_dict = st.session_state['global_dict']


if 'logger' not in st.session_state:
    from logger import logger
    logger.switchLevel("INFO")
    st.session_state['logger'] = logger
logger = st.session_state['logger']


# 문의사항 데이터 가져오기
if 'data_dic' not in st.session_state:
    with open("data.json", 'r', encoding='utf-8') as f:
        st.session_state['data_dic'] = json.load(f)
data_dic = st.session_state['data_dic']


# 언어모델
with st.spinner("Loading LLM..."):
    if 'llm' not in st.session_state:
        st.session_state['llm'] = ChatOpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            model=MODEL,
            temperature=0.0,
        )
llm = st.session_state['llm']


# 임베딩 모델
with st.spinner("Loading LLM..."):
    if 'emb' not in st.session_state:
        st.session_state['emb'] = OpenAIEmbeddings(
            base_url=BASE_URL,
            api_key=API_KEY,
        )
emb = st.session_state['emb']


# 채팅 이력
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ("assistant", "저는 FAQ 채팅 상담가입니다. 어떤 문의가 필요하신가요?"),
    ]
messages = st.session_state["messages"]


# 숨겨진 채팅 이력
if "hidden_messages" not in st.session_state:
    st.session_state["hidden_messages"] = [
        ("assistant", "저는 FAQ 채팅 상담가입니다. 어떤 문의가 필요하신가요?"),
        ("assistant", "[검색된 FAQ 리스트] : (Path, Q, A)\n리스트 없음"),
    ]
hidden_messages = st.session_state["hidden_messages"]


# 파일 -> 벡터저장소
with st.spinner("Loading FAQ data..."):
    if 'retriever' not in st.session_state:
        docs = [title for title in data_dic]
        vectorstore = FAISS.from_texts(docs, embedding=emb, distance_strategy=DistanceStrategy.COSINE)
        st.session_state['retriever'] = vectorstore.as_retriever(search_kwargs={'k':4})

retriever = st.session_state['retriever']


# van_chain
if 'van_chain' not in st.session_state:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT),
        MessagesPlaceholder(variable_name='messages1'),
        ("user", "{question}"),
    ])
    st.session_state['van_chain'] = prompt | llm | StrOutputParser()
van_chain = st.session_state['van_chain']


# con_chain
if 'con_chain' not in st.session_state:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT),
        MessagesPlaceholder(variable_name='messages1'),
        ("user", PROMPT1),
    ])
    st.session_state['con_chain'] = prompt | llm | StrOutputParser() | get_log_fn("모드 >>>>> ")
con_chain = st.session_state['con_chain']

# con_chain2
if 'con_chain2' not in st.session_state:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT),
        MessagesPlaceholder(variable_name='messages1'),
        ("user", PROMPT1_2),
    ])
    st.session_state['con_chain2'] = prompt | llm | StrOutputParser() | get_log_fn("모드 >>>>> ")
con_chain2 = st.session_state['con_chain2']


# rag_chain
if 'rag_chain' not in st.session_state:
    # 검색 문구 만들기
    prompt1 = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT), 
        MessagesPlaceholder(variable_name='messages1'), 
        ("user", PROMPT2),
    ])
    
    # 검색 문구 + 검색결과 분석
    prompt2 = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT), 
        ("user", RAG_PROMPT_TEMPLATE),
    ])

    # st.session_state['rag_chain'] = (
    #     prompt1 | llm | StrOutputParser() | get_log_fn("검색어")
    #     | {
    #         "result" : retriever | format_docs | get_interception_fn("result"),
    #         "question" : RunnablePassthrough(),
    #     }
    #     | prompt2 | llm | StrOutputParser()
    
    st.session_state['rag_chain'] = (
        RunnablePassthrough()
        | {
            "search" : prompt1 | llm | StrOutputParser(),
            "question" : itemgetter("question"),
        }
        | {
            "result" : itemgetter("search") | retriever | format_docs | get_interception_fn("result"),
            "search" : RunnablePassthrough() | itemgetter("search") | get_log_fn("검색어 >>>>> "),
            "question" : RunnablePassthrough() | itemgetter("question") | get_log_fn("사용자질문 >>>>> "),
        }
        | prompt2 | llm | StrOutputParser()
    )
rag_chain = st.session_state['rag_chain']


##########################################################################################
# 제목
# st.title(TITLE)

# background_image = """
# <style>
# [data-testid="stAppViewContainer"] > .main {
#     background-image: url("https://help.nexon.com/image/FrontConfig/18/427de99a48874673b62f87359f847cb3.jpg");
#     background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
#     background-position: center;  
#     background-repeat: no-repeat;
# }
# </style>
# """

# st.markdown(background_image, unsafe_allow_html=True)

title_html = """
    <style>
        .title {
            padding: 20px;
            background-image: url('https://help.nexon.com/image/FrontConfig/18/427de99a48874673b62f87359f847cb3.jpg');
            background-size: cover;
            color: white;
            text-align: center;
            text-shadow: 
                -1.5px -1.5px 0 #000,
                1.5px -1.5px 0 #000,
                -1.5px 1.5px 0 #000,
                1.5px 1.5px 0 #000;
        }
    </style>
    <h1 class="title">%s</h1>
""" % TITLE

# HTML/CSS를 사용하여 제목 표시
st.markdown(title_html, unsafe_allow_html=True)


# 채팅 내역 출력
for msg in messages:
    st.chat_message(msg[0]).write(msg[1])

# 유저 입력
if user_input := st.chat_input():
    
    # 채팅 표시
    st.chat_message("user").write(user_input)

    # 조건문
    with st.chat_message('assistant'):
        bot_out = st.empty()
        condition = con_chain.invoke({"messages1":hidden_messages, "question":user_input})
        condition2 = con_chain2.invoke({"messages1":hidden_messages, "question":user_input})

        is_rag = condition[:3] == "!AA" and condition2[:3] == "!DD"
        chain = [van_chain, rag_chain][is_rag]
        msg = ''
        for t in chain.stream({"messages1":hidden_messages, "question":user_input}):
            msg += t
            bot_out.markdown(msg)

    messages.append(("user", user_input))
    messages.append(("assistant", msg))
    
    hidden_messages.append(("user", user_input))
    if is_rag: hidden_messages.append(("assistant", global_dict["result"]))
    hidden_messages.append(("assistant", msg))

    for m in hidden_messages:
        logger.info(str(m)[:100])

    
    






