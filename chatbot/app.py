from flask import Flask, render_template, request, jsonify
import requests
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

db = Chroma.from_texts(texts, embed_model, persist_directory=persist_directory)
loaded_db = Chroma(persist_directory=persist_directory, embedding_function=embed_model)

def augment_prompt(query: str):
    results = db.similarity_search(query, k=2)
    source_knowledge = "\n".join([x.page_content for x in results])
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt

@app.route('/')
def home():
    return render_template('index.html')  # HTML 파일 렌더링

@app.route('/ask', methods=['POST'])
def ask():
    query = request.form['query']
    prompt = HumanMessage(content=augment_prompt(query))
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        prompt
    ]
    res = llm.invoke(messages)
    return jsonify({"response": res.content})
