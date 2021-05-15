import asyncio
import logging

import aioredis
import ujson
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          TokenClassificationPipeline)


async def task():
    tokenizer = AutoTokenizer.from_pretrained(
        'Alaeddin/convbert-base-turkish-ner-cased')
    model = AutoModelForTokenClassification.from_pretrained(
        'Alaeddin/convbert-base-turkish-ner-cased')
    ner = TokenClassificationPipeline(
        model=model, tokenizer=tokenizer, grouped_entities=True)

    queue = await aioredis.create_redis_pool("redis://redis:6379/0?encoding=utf-8")
    logging.warning("Connected to Redis")

    logging.warning("NER task is running asynchronously...")
    while True:
        pipe = queue.pipeline()
        pipe.lrange("ner", 0, 7)
        pipe.ltrim("ner", 8, -1)
        requests, _ = await pipe.execute()

        for r in requests:
            r = ujson.loads(r)
            results = ner(r["text"])
            for i in range(len(results)):
                results[i]['start'] = int(results[i]['start'])
                results[i]['end'] = int(results[i]['end'])

            await queue.set(r["id"], ujson.dumps(results))

        asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(task())
