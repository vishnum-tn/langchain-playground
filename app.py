import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

app = FastAPI(title="LangChain Chatbot")
templates = Jinja2Templates(directory="templates")


def get_llm(provider: str = None):
    provider = provider or os.getenv("LLM_PROVIDER", "ollama")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"), temperature=0.7)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-pro"), temperature=0.7)
    elif provider == "llama.cpp":
        from langchain_openai import ChatOpenAI
        base_url = os.getenv("LLAMA_CPP_URL", "http://127.0.0.1:8080/v1")
        return ChatOpenAI(model="local-model", base_url=base_url, api_key="not-needed", temperature=0.7)
    else:  # ollama
        from langchain_ollama import ChatOllama
        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=0.7)


chat_history = []
system_prompt = SystemMessage(content="You are a helpful AI assistant. Answer concisely and clearly.")


class ChatRequest(BaseModel):
    message: str
    provider: str = None


class ChatResponse(BaseModel):
    reply: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    llm = get_llm(req.provider)

    prompt = ChatPromptTemplate.from_messages([
        system_prompt,
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm

    result = chain.invoke({
        "history": chat_history,
        "input": req.message
    })

    chat_history.append(HumanMessage(content=req.message))
    chat_history.append(AIMessage(content=result.content))

    if len(chat_history) > 20:
        chat_history[:] = chat_history[-20:]

    return ChatResponse(reply=result.content)


@app.delete("/chat/history")
async def clear_history():
    chat_history.clear()
    return {"status": "History cleared"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
