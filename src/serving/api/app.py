from fastapi import FastAPI
from ..inference.generator import ConstrainedGenerator
from ...agents.med_agent import MedAgent

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(req: dict):
    # construct (retriever, generator, safety) via DI container
    agent: MedAgent = app.state.agent
    out = agent.respond(req["query"], context=req.get("context", {}))
    return out