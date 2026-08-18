import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="LangChain Router API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer clearly and concisely."),
    ("human", "{input}")
])


def get_gemini_chain():
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7,
    )
    return prompt | llm | StrOutputParser()


def get_ollama_chain():
    from langchain_ollama import ChatOllama
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.7,
    )
    return prompt | llm | StrOutputParser()


class ChatRequest(BaseModel):
    input: str


class ChatResponse(BaseModel):
    model: str
    reply: str


@app.get("/")
def root():
    return {
        "routes": {
            "gemini": "POST /gemini/invoke",
            "ollama": "POST /ollama/invoke",
            "both":   "POST /router/invoke",
        }
    }


@app.post("/gemini/invoke", response_model=ChatResponse)
async def gemini_invoke(req: ChatRequest):
    try:
        reply = get_gemini_chain().invoke({"input": req.input})
        return ChatResponse(model="gemini", reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ollama/invoke", response_model=ChatResponse)
async def ollama_invoke(req: ChatRequest):
    try:
        reply = get_ollama_chain().invoke({"input": req.input})
        return ChatResponse(model="ollama", reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/router/invoke")
async def router_invoke(req: ChatRequest):
    results = {}
    for name, get_chain in [("gemini", get_gemini_chain), ("ollama", get_ollama_chain)]:
        try:
            results[name] = get_chain().invoke({"input": req.input})
        except Exception as e:
            results[name] = f"Error: {e}"
    return {"input": req.input, "responses": results}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
