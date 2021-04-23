import json
import logging
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import redis
from pipelines import pipeline #pipelines.py script in the cloned repo


def run_qaqg():
    while True:
        with queue.pipeline() as pipe:
            pipe.lrange("qaqg", 0, 7)
            pipe.ltrim("qaqg", 8, -1)
            requests, _ = pipe.execute()

        for r in requests:
            start = time.time()
            r = json.loads(r)
            results = {}
            if r.get("question", None) is None:
                results = multimodel(r["text"])
            else:
                results = multimodel({"context": r["text"], "question": r["question"]})
            
            queue.set(r["id"], json.dumps(results))
            logging.warning(f"{r['id']} processed in {time.time() - start}")

        time.sleep(0.1)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("ozcangundes/mt5-multitask-qa-qg-turkish") 
    model = AutoModelForSeq2SeqLM.from_pretrained("ozcangundes/mt5-multitask-qa-qg-turkish")
    multimodel = pipeline("multitask-qa-qg",tokenizer=tokenizer,model=model)

    queue = redis.StrictRedis(host="redis")
    logging.warning("Connected to Redis")

    run_qaqg()