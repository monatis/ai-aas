import asyncio
import logging

import aioredis
import ujson
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def generate_summary(main_news, tokenizer, model):
    source_encoding = tokenizer(
        main_news,
        max_length=784,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt")

    generated_ids = model.generate(
        input_ids=source_encoding["input_ids"],
        attention_mask=source_encoding["attention_mask"],
        num_beams=3,
        max_length=120,
        repetition_penalty=2.5,
        length_penalty=2.0,
        early_stopping=True,
        use_cache=True
    )

    preds = [tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
             for gen_id in generated_ids]

    return "".join(preds)


async def task():
    tokenizer = AutoTokenizer.from_pretrained(
        "ozcangundes/mt5-small-turkish-summarization")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "ozcangundes/mt5-small-turkish-summarization")

    queue = await aioredis.create_redis_pool("redis://redis:6379/0?encoding=utf-8")
    logging.warning("Connected to Redis")

    logging.warning("ATS task is running asynchronously...")
    while True:
        pipe = queue.pipeline()
        pipe.lrange("ats", 0, 7)
        pipe.ltrim("ats", 8, -1)
        requests, _ = await pipe.execute()

        for r in requests:
            r = ujson.loads(r)
            results = {"summary": generate_summary(
                r["text"], tokenizer, model)}

            await queue.set(r["id"], ujson.dumps(results))

        asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(task())
