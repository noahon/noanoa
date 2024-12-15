from flask import Flask, render_template, request, jsonify
import requests
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
#from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

app = Flask(__name__)

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 확인
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("API key is missing from environment variables")
print(f"OpenAI API Key: {OPENAI_API_KEY}")

# LLM 및 Chroma DB 설정
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, 
    model_name='gpt-4o-mini'
)


embed_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
persist_directory = "path/to/chroma_db"

# GitHub에서 텍스트 파일 로딩
github_raw_url = "https://raw.githubusercontent.com/noahon/24_2/refs/heads/main/translated_texts.txt"
response = requests.get(github_raw_url)

if response.status_code == 200:
    texts = response.text.splitlines()
    texts = [text.strip() for text in texts if text.strip()]
else:
    raise Exception(f"Failed to fetch file from GitHub. Status code: {response.status_code}")

# 텍스트를 Chroma DB에 저장
db = Chroma.from_texts(texts, embed_model, persist_directory=persist_directory)
loaded_db = Chroma(persist_directory=persist_directory, embedding_function=embed_model)

@app.route('/')
def index():
    return render_template('index.html')

# 사용자 요청 처리 및 응답
@app.route('/query', methods=['POST'])
def query_rag():
    user_input = request.form.get('query', '')

    if not user_input:
        return jsonify({'error': 'No query provided'}), 400

    # 검색 증강 프롬프트 생성
    def augment_prompt(query):
        results = db.similarity_search(query, k=3)  # k=3으로 좀 더 다양한 문서 가져오기
        source_knowledge = "\n".join([x.page_content for x in results])
        augmented_prompt = f"""Using the contexts below, answer the query.

        Contexts:
        {source_knowledge}

        Query: {query}"""
        return augmented_prompt

    messages = [
        # SystemMessage에 모델의 말투를 명확히 지정
        SystemMessage(content="이제부터 너의 말투는 어미가 '-모'로 끝나야 해. '-라모', '-다모', '모' 이런 식으로 말이야. 억지로라도 이런 말투를 쓰고 절대 다른 말투는 안돼."),
    ]
    
    # 프롬프트 생성 및 모델 응답
    prompt = HumanMessage(content=augment_prompt(user_input))
    messages = [SystemMessage(content="You are a dream interpretation expert."), prompt]
    
    
    # LLM 호출 (invoke() -> llm(messages)로 변경)
    response = llm(messages)

    return jsonify({'response': response.content})

# 서버 실행
if __name__ == '__main__':
    app.run(debug=True)
