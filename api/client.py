from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)


def get_llamacpp_llm():
    from langchain_openai import ChatOpenAI
    base_url = os.getenv("LLAMA_CPP_URL", "http://127.0.0.1:8080/v1")
    model = os.getenv("LLAMA_CPP_MODEL", "ggml-org/gemma-3-1b-it-GGUF")
    logger.info(f"llama.cpp LLM: model={model}, base_url={base_url}")
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key="not-needed",
        temperature=0.7,
    )


def get_gemini_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    logger.info(f"Gemini LLM: model={model}")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7,
    )


llm = get_llamacpp_llm()
gemini_llm = get_gemini_llm()

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} for a 5 year old child with 100 words")

add_routes(
    app,
    prompt1 | llm,
    path="/essay"
)

add_routes(
    app,
    prompt2 | gemini_llm,
    path="/poem"
)

logger.info("Routes registered: /essay (llama.cpp), /poem (Gemini)")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
