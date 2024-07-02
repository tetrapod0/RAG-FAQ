from langchain_core.messages import ChatMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_openai import ChatOpenAI


from langserve.pydantic_v1 import BaseModel, Field
from typing import List, Union

from openai import OpenAI
from typing import List

import streamlit as st
import json
import os


TITLE = "메이플 FAQ 문의 채팅"

RAG_PROMPT_TEMPLATE = "You always answer into Korean. You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. \nContext: {context} \nQuestion: {question} (You must answer in Korean.) \nAnswer:"
SYS_PROMPT = "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability. You always answer succinctly. You must always answer in Korean."


with open("data.json", 'r', encoding='utf-8') as f:
    data_dic = json.load(f)


class MyEmbeddings(Embeddings):
    def __init__(self, base_url, api_key="lm-studio"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def embed_documents(self, texts: List[str], model="nomic-ai/nomic-embed-text-v1.5-GGUF") -> List[List[float]]:
        texts = list(map(lambda text:text.replace("\n", " "), texts))
        datas = self.client.embeddings.create(input=texts, model=model).data
        return list(map(lambda data:data.embedding, datas))
        
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


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


# 언어모델
if 'llm' not in st.session_state:
    with st.spinner("Loading LLM..."):
        st.session_state['llm'] = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
            temperature=0.1,
        )
llm = st.session_state['llm']


# 임베딩 모델
if 'emb' not in st.session_state:
    with st.spinner("Loading LLM..."):
        st.session_state['emb'] = MyEmbeddings(base_url="http://localhost:1234/v1")
emb = st.session_state['emb']

# 보이는 채팅 이력
if "visible_messages" not in st.session_state:
    st.session_state["visible_messages"] = [
        ("assistant", "저는 FAQ 채팅 상담가입니다. 어떤 문의가 필요하신가요?"),
    ]
visible_messages = st.session_state["visible_messages"]

# 숨겨진 채팅 이력
if "hidden_messages" not in st.session_state:
    st.session_state["hidden_messages"] = [
        ("assistant", "저는 FAQ 채팅 상담가입니다. 어떤 문의가 필요하신가요?"),
    ]
hidden_messages = st.session_state["hidden_messages"]


# 파일 -> 벡터저장소
if 'retriever' not in st.session_state:
    with st.spinner("Loading FAQ data..."):
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


# # chat 체인
if 'chat_chain' not in st.session_state:
    prompt = ChatPromptTemplate.from_messages([
        # ("system", SYS_PROMPT),
        ("system", """
        너는 FAQ 채팅 상담가이다.
        [검색된 FAQ 리스트]를 참고하여 metadata를 포함하여 사용자 질문에 답변해라.
        특정 주제나 문제와 관련이 없는 일반적인 질문인 경우, 답변하지 말것.
        그리고 한국어를 사용해라.
        """),
        MessagesPlaceholder(variable_name='messages1'),
        ("user", """
        내 질문에 한국어로 답변해줘.
        Question : {question}
        """)
    ])
    st.session_state['chat_chain'] = prompt | llm | StrOutputParser()
chain = st.session_state['chat_chain']

# # chat 체인
# if 'chat_chain' not in st.session_state:
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", SYS_PROMPT),
#         MessagesPlaceholder(variable_name='messages1'),
#     ])
#     st.session_state['chat_chain'] = prompt | llm | StrOutputParser()
# chain = st.session_state['chat_chain']

# chat 체인
# if 'chat_chain' not in st.session_state:
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", SYS_PROMPT),
#         MessagesPlaceholder(variable_name='messages1'),
#         ("user", RAG_PROMPT_TEMPLATE),
#     ])
#     st.session_state['chat_chain'] = prompt | llm | StrOutputParser()
# chain = st.session_state['chat_chain']



##########################################################################################

# 제목
st.title(TITLE)


# 채팅 내역 출력
for msg in visible_messages:
    st.chat_message(msg[0]).write(msg[1])


# 유저 입력
if user_input := st.chat_input():
    
    # 채팅 표시
    st.chat_message('user').write(user_input)

    # 조건문
    # hidden_messages.append(('system', CON_PROMPT))
    # chain.innoke(hidden_messages)
    
    # rag 검색
    # format_docs = lambda docs:"[검색된 데이터] : \n\n" + "\n\n".join(doc.page_content for doc in docs)
    rag = retriever | format_docs
    result = rag.invoke(user_input)
    
    # 채팅 저장
    # visible_messages.append(("user", user_input))
    hidden_messages.append(("user", result))
    # hidden_messages.append(("user", user_input))
        
    with st.chat_message('assistant'):
        bot_out = st.empty()
        msg = ''
        # for t in chain.stream(hidden_messages):
        for t in chain.stream({
            "messages1" : hidden_messages,
            # "context" : result,
            "question" : user_input,
        }):
            msg += t
            bot_out.markdown(msg)

    
    visible_messages.append(("user", user_input))
    # hidden_messages.append(("user", result))
    hidden_messages.append(("user", user_input))

    visible_messages.append(('assistant', msg))
    hidden_messages.append(('assistant', msg))
    







