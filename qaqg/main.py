import asyncio
import logging

import aioredis
import ujson
from pipelines import pipeline  # pipelines.py script in the cloned repo
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


async def task():
    tokenizer = AutoTokenizer.from_pretrained(
        "ozcangundes/mt5-multitask-qa-qg-turkish")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "ozcangundes/mt5-multitask-qa-qg-turkish")
    multimodel = pipeline("multitask-qa-qg", tokenizer=tokenizer, model=model)

    queue = await aioredis.create_redis_pool("redis://redis:6379/0?encoding=utf-8")
    logging.warning("Connected to Redis")

    logging.warning("QAQG task is running asynchronously...")
    while True:
        pipe = queue.pipeline()
        pipe.lrange("qaqg", 0, 7)
        pipe.ltrim("qaqg", 8, -1)
        requests, _ = await pipe.execute()

        for r in requests:
            r = ujson.loads(r)
            results = {}
            if r.get("question", None) is None:
                results = multimodel(r["text"])
            else:
                results = multimodel(
                    {"context": r["text"], "question": r["question"]})

            await queue.set(r["id"], ujson.dumps(results))

        asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(task())
