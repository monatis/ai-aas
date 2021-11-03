import asyncio
import logging

import aioredis
import numpy as np
import tensorflow as tf
import ujson
from transformers import AutoTokenizer, TFAutoModel

#model_name = 'mys/bert-base-turkish-cased-nli-mean'
model_name = 'mys/bert-base-turkish-cased-nli-mean'


def label_text(model, tokenizer, texts, labels):
    texts_length = len(texts)
    tokens = tokenizer(texts + labels, padding=True, return_tensors='tf')
    embs = model(**tokens)[0]

    attention_masks = tf.cast(tokens['attention_mask'], tf.float32)
    sample_length = tf.reduce_sum(attention_masks, axis=-1, keepdims=True)
    masked_embs = embs * tf.expand_dims(attention_masks, axis=-1)
    masked_embs = tf.reduce_sum(masked_embs, axis=1) / tf.cast(sample_length, tf.float32)

    dists = tf.experimental.numpy.inner(masked_embs[:texts_length], masked_embs[texts_length:])
    scores = tf.nn.softmax(dists)
    results = list(zip(labels, scores.numpy().squeeze().tolist()))
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    sorted_results = [{"label": label, "score": f"{score:.4f}"} for label, score in sorted_results]
    return sorted_results


async def task():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModel.from_pretrained(model_name)

    queue = await aioredis.create_redis("redis://redis:6379/0?encoding=utf-8")
    logging.warning("Connected to Redis")
    logging.warning("ZSL task is running asynchronously...")
    
    while True:
        pipe = queue.pipeline()
        pipe.lrange("zsl", 0, 7)
        pipe.ltrim("zsl", 8, -1)
        requests, _ = await pipe.execute()
        
        for r in requests:
            r = ujson.loads(r)

            results = []
            for i in range(len(r["texts"])):
                sorted_results = label_text(model, tokenizer, [r["texts"][i]], r["labels"])
                results.append({'text': r["texts"][i], 'results': sorted_results})

            await queue.set(r["id"], ujson.dumps(results))

        await asyncio.sleep(0.1)


if __name__ == "__main__":
    logging.warning("ZSL started")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(task())
