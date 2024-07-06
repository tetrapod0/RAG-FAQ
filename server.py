from langchain_core.messages import ChatMessage, HumanMessage, AIMessage, SystemMessage
# from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


from langserve.pydantic_v1 import BaseModel, Field
from typing import List, Union

from openai import OpenAI
from operator import itemgetter
from typing import List

import streamlit as st
import time
import json
import os


# BASE_URL = "http://localhost:1234/v1",
# API_KEY = "lm-studio",
# MODEL = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
BASE_URL = "https://api.openai.com/v1"
API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-3.5-turbo"


TITLE = "메이플 FAQ 문의 채팅"

SYS_PROMPT = """
당신은 똑똑하고 친절한 '메이플스토리 게임의 FAQ 채팅 상담가'입니다.
"""

PROMPT1 = """
Question : {question}

당신은 Question에 대해 검색이 필요한 경우 '!검색'이라고 외칩니다.
그렇지 않은 경우 '!일반'이라고 외칩니다.
그 외 아무것도 하지 않습니다.
"""

PROMPT2 = """
Question : {question}

Question을 검색할 문구로 변환하여 출력하시오.
"""

RAG_PROMPT_TEMPLATE = """
검색결과:
{result}

사용자 질문:
{question}

검색결과로 사용자 질문에 충분히 답변할 수 있는 경우에만 답변하시오. 
답변하기 충분하지 않은 경우 사용자에게 더 자세한 질문을 요청하세요.
"""




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


# 브라우저 탭 이름
st.set_page_config(page_title=TITLE)


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
        # ChatMessage(role="assistant", content="저는 FAQ 채팅 상담가입니다. 어떤 문의가 필요하신가요?"),
        ("assistant", "저는 FAQ 채팅 상담가입니다. 어떤 문의가 필요하신가요?"),
    ]
messages = st.session_state["messages"]


# 파일 -> 벡터저장소
with st.spinner("Loading FAQ data..."):
    if 'retriever' not in st.session_state:
        # docs = [Document(page_content=title, metadata={"path":value['metadata']}) 
        #         for title, value in data_dic.items()]
        docs = [title for title in data_dic]
        vectorstore = FAISS.from_texts(docs, embedding=emb, distance_strategy=DistanceStrategy.COSINE)
        st.session_state['retriever'] = vectorstore.as_retriever(search_kwargs={'k':5})

        # data_str = []
        # for key, value in data_dic.items():
        #     q = key
        #     a = value["body"]
        #     metadata = value["metadata"]
        #     one = f"**Metadata** : {metadata}\n**Question** : {q}\n**Answer** : \n{a}"
        #     data_str.append(one)
        # data_str = "\n\n---\n\n".join(data_str)
        
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        # splits = text_splitter.split_text(data_str)
        # vectorstore = FAISS.from_texts(splits, embedding=emb, distance_strategy=DistanceStrategy.COSINE)
        # st.session_state['retriever'] = vectorstore.as_retriever(search_kwargs={'k':10})
retriever = st.session_state['retriever']



# van_chain
if 'van_chain' not in st.session_state:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT + " 관련되지 않은 질문은 받지 않습니다."),
        MessagesPlaceholder(variable_name='messages1'),
        ("user", "{question}"),
    ])
    st.session_state['van_chain'] = prompt | llm | StrOutputParser()
van_chain = st.session_state['van_chain']

# con_chain
if 'con_chain' not in st.session_state:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT + " 관련되지 않은 질문은 받지 않습니다."),
        MessagesPlaceholder(variable_name='messages1'),
        ("user", PROMPT1),
    ])
    st.session_state['con_chain'] = prompt | llm | StrOutputParser()
con_chain = st.session_state['con_chain']

# rag_chain
if 'rag_chain' not in st.session_state:
    prompt1 = ChatPromptTemplate.from_messages([("system", SYS_PROMPT), ("user", PROMPT2),])
    prompt2 = ChatPromptTemplate.from_messages([("system", SYS_PROMPT), ("user", RAG_PROMPT_TEMPLATE),])

    # ch1 = prompt1 | llm | StrOutputParser() | {"question" : RunnablePassthrough()}
    
    st.session_state['rag_chain'] = (
        prompt1 | llm | StrOutputParser()
        | {
            "result" : retriever | format_docs,
            "question" : RunnablePassthrough(),
        }
        | prompt2 | llm | StrOutputParser()
    )
rag_chain = st.session_state['rag_chain']


##########################################################################################

# 제목
st.title(TITLE)

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
        condition = con_chain.invoke({"messages1":messages, "question":user_input})
        print(condition)

        chain = [van_chain, rag_chain][condition[:3] == "!검색"]
        msg = ''
        for t in chain.stream({"messages1":messages, "question":user_input}):
            msg += t
            bot_out.markdown(msg)

        
    messages.append(("user", user_input))
    messages.append(("assistant", msg))
    
    






