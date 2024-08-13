# 2024.08.13 ch6 외부 검색과 히스토리를 바탕으로 응답하는 웹 앱 구현하기
import os
import streamlit as st
# 대화 기록 보여주는 StreamlitChatMessageHistory 클래스 가져오기 -> pip install langchain-community
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
# .env 파일에서 환경변수 가져오기
from dotenv import load_dotenv
# chatGPT API 가져오기
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

# 6.8 Agents를 사용하여 필요에 따라 외부 정보 검색하게 하기
from langchain import hub
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langsmith import Client
# 채팅 대화 기록 바탕으로 응답하기 위한 langchain.memory 모듈 가져오기
from langchain.memory import ConversationBufferMemory
load_dotenv()

### Agent 생성하는 함수
def create_agent_chain(history):
  chat = ChatOpenAI(
      model_name=os.environ["OPENAI_API_MODEL"],
      temperature=os.environ["OPENAI_API_TEMPERATURE"],
    )
  tools = load_tools(["ddg-search", "wikipedia"])
  client = Client(api_key=os.environ["LANGSMITH_API_KEY"])
  prompt = client.pull_prompt("hwchase17/openai-tools-agent")
  # OpenAI Functions Agent가 사용할 수 있는 설정으로 memory 초기화
  memory = ConversationBufferMemory(
    chat_memory=history,memory_key="chat_history",return_messages=True
  )
  agent = create_openai_tools_agent(chat,tools,prompt)
  return AgentExecutor(agent=agent, tools=tools, memory=memory)
st.title("langchain-streamlit-app")
# history 객체 생성
history = StreamlitChatMessageHistory()

for message in history.messages:
  st.chat_message(message.type).write(message.content)
### 프롬프트 입력값 받기
prompt = st.chat_input("안녕?")


### 프롬프트 화면에 출력

if prompt: 
  with st.chat_message("user"):
    st.markdown(prompt)
  
  with st.chat_message("assistant"):
    callback = StreamlitCallbackHandler(st.container())
    
    agent_chain = create_agent_chain(history)
    res = agent_chain.invoke(
      {"input": prompt},
      {"callbacks": [callback]}
    )
    
    st.markdown(res["output"])