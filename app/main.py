import asyncio
import json
import logging
import time
import uuid

import aioredis
import async_timeout
from fastapi import Depends, FastAPI, HTTPException
from qdrant_client import QdrantClient
from starlette.requests import Request

from schemas import QAInput, SingleTextInput, ZSLTextInput

app = FastAPI(
    title="Intelligence as a Service",
    description="Fast, easy-to-use, consistent AI APIs APIs for developers. Use one of our SDKs or simply make an http request to our endpoints from anywhere you need it.",
    version="0.0.1-alpha"
)


@app.on_event('startup')
async def app_startup():
    app.state.redis = await aioredis.create_redis_pool('redis://redis:6379/0?encoding=utf-8')
    logging.warning('Connected to redis')
    app.state.qdrant = QdrantClient(host="qdrant", port=6333)
    logging.warning('connected to Qdrant')


@app.on_event('shutdown')
async def app_shutdown():
    app.state.redis.close()
    await app.state.redis.wait_closed()
    app.state.qdrant.close()


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
    await app.state.redis.rpush("zsl", json.dumps(zsl_input))
    try:
        async with async_timeout.timeout(3.0):
            while True:
                output = await app.state.redis.get(k)
                if output is not None:
                    resp["predictions"] = json.loads(output)
                    await app.state.redis.delete(k)
                    break
                asyncio.sleep(0.1)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408, detail="Request timed out after waiting for 3 seconds")

    resp["success"] = True
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
    await app.state.redis.rpush("ats", json.dumps(ats_input))
    try:
        async with async_timeout.timeout(10.0):
            while True:
                output = await app.state.redis.get(k)
                if output is not None:
                    resp["predictions"] = json.loads(output)
                    await app.state.redis.delete(k)
                    break
                asyncio.sleep(0.2)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408, detail="Request timed out after waiting for 10 seconds")

    resp["success"] = True
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
    await app.state.redis.rpush("qaqg", json.dumps(qg_input))
    try:
        async with async_timeout.timeout(60.0):
            while True:
                output = await app.state.redis.get(k)
                if output is not None:
                    resp["predictions"] = json.loads(output)
                    await app.state.redis.delete(k)
                    break
                asyncio.sleep(0.2)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408, detail="Request timed out after waiting for 60 seconds")

    resp["success"] = True
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
    await app.state.redis.rpush("qaqg", json.dumps(qa_input))
    try:
        async with async_timeout.timeout(10.0):
            while True:
                output = await app.state.redis.get(k)
                if output is not None:
                    resp["predictions"] = json.loads(output)
                    await app.state.redis.delete(k)
                    break
                asyncio.sleep(0.1)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408, detail="Request timed out after waiting for 10 seconds")

    resp["success"] = True
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
    await app.state.redis.rpush("ner", json.dumps(ner_input))
    try:
        async with async_timeout.timeout(3.0):
            while True:
                output = await app.state.redis.get(k)
                if output is not None:
                    resp["predictions"] = json.loads(output)
                    await app.state.redis.delete(k)
                    break
                asyncio.sleep(0.1)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408, detail="Request timed out after waiting for 3 seconds")

    resp["success"] = True
    return resp
