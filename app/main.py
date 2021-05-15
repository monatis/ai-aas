import time

from fastapi import Depends, FastAPI, HTTPException
from starlette.requests import Request
import redis
import uuid
import json
from schemas import (ZSLTextInput, SingleTextInput,
                     QAInput)

import logging
queue = redis.StrictRedis(host="redis")
from qdrant_client import QdrantClient
qdrant = QdrantClient(host="qdrant", port=6333)
logging.warning("Connected to qdrant")

app = FastAPI(
    title="Intelligence as a Service",
    description="Fast, easy-to-use, consistent AI APIs APIs for developers. Use one of our SDKs or simply make an http request to our endpoints from anywhere you need it.",
    version="0.0.1-alpha"
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/")
async def get_status_version():
    return {"status": "ok", "version": "0.0.1-alpha", "author": "M. Yusuf Sarıgöz", "license": "Apache 2.0"}


@app.post('/text/classification')
async def classify_text(zsl_input: ZSLTextInput):
    """
    Post a list of texts with a list of possible labels
    for zero shot text classification. Get the recognized label and score.
    """
    resp = {"success": False}
    k = str(uuid.uuid4())
    zsl_input = zsl_input.dict()
    zsl_input["id"] = k
    queue.rpush("zsl", json.dumps(zsl_input))
    num_tries, max_tries = 0, 20
    while num_tries < max_tries:
        num_tries += 1
        output = queue.get(k)
        if output is not None:
            resp["predictions"] = json.loads(output)
            queue.delete(k)
            break
        time.sleep(0.08)
        resp["success"] = True
    else:
        raise HTTPException(status_code=400, detail=f"Request failed after {max_tries} tries")

    return resp

    
@app.post('/text/summarization')
async def summarize_text(ats_input: SingleTextInput):
    """
    Post a long text, and get an abstractive textual summary.
    """
    resp = {"success": False}
    k = str(uuid.uuid4())
    ats_input = ats_input.dict()
    ats_input["id"] = k
    queue.rpush("ats", json.dumps(ats_input))
    num_tries, max_tries = 0, 100
    while num_tries < max_tries:
        num_tries += 1
        output = queue.get(k)
        if output is not None:
            resp["predictions"] = json.loads(output)
            queue.delete(k)
            break
        time.sleep(0.1)
        resp["success"] = True
    else:
        raise HTTPException(status_code=400, detail=f"Request failed after {max_tries} tries")

    return resp

    
@app.post('/text/qg')
async def generate_questions(qg_input: SingleTextInput):
    """
    Post a long text, and get a set of questions and their answers.
    """
    resp = {"success": False}
    k = str(uuid.uuid4())
    qg_input = qg_input.dict()
    qg_input["id"] = k
    queue.rpush("qaqg", json.dumps(qg_input))
    num_tries, max_tries = 0, 300
    while num_tries < max_tries:
        num_tries += 1
        output = queue.get(k)
        if output is not None:
            resp["predictions"] = json.loads(output)
            queue.delete(k)
            break
        time.sleep(0.2)
        resp["success"] = True
    else:
        raise HTTPException(status_code=400, detail=f"Request failed after {max_tries} tries")

    return resp    


@app.post('/text/qa')
async def answer_question(qa_input: QAInput):
    """
    Post a long text and a question, and get an extractive answer.
    """
    resp = {"success": False}
    k = str(uuid.uuid4())
    qa_input = qa_input.dict()
    qa_input["id"] = k
    queue.rpush("qaqg", json.dumps(qa_input))
    num_tries, max_tries = 0, 100
    while num_tries < max_tries:
        num_tries += 1
        output = queue.get(k)
        if output is not None:
            resp["predictions"] = json.loads(output)
            queue.delete(k)
            break
        time.sleep(0.1)
        resp["success"] = True
    else:
        raise HTTPException(status_code=400, detail=f"Request failed after {max_tries} tries")

    return resp

        
@app.post('/text/ner')
async def recognize_named_entities(ner_input: SingleTextInput):
    """
    Post a text, and get a set of named entities.
    """
    resp = {"success": False}
    k = str(uuid.uuid4())
    ner_input = ner_input.dict()
    ner_input["id"] = k
    queue.rpush("ner", json.dumps(ner_input))
    num_tries, max_tries = 0, 20
    while num_tries < max_tries:
        num_tries += 1
        output = queue.get(k)
        if output is not None:
            resp["predictions"] = json.loads(output)
            queue.delete(k)
            break
        time.sleep(0.08)
        resp["success"] = True
    else:
        raise HTTPException(status_code=400, detail=f"Request failed after {max_tries} tries")

    return resp    

