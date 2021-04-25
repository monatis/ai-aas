import json
import logging
import time
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import redis

def run_ner():
    while True:
        with queue.pipeline() as pipe:
            pipe.lrange("ner", 0, 7)
            pipe.ltrim("ner", 8, -1)
            requests, _ = pipe.execute()

        for r in requests:
            start = time.time()
            r = json.loads(r)
            results = ner(r["text"])
            for i in range(len(results)):
                results[i]['start'] = int(results[i]['start'])
                results[i]['end'] = int(results[i]['end'])
            
            queue.set(r["id"], json.dumps(results))
            logging.warning(f"{r['id']} processed in {time.time() - start}")

        time.sleep(0.1)



if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained('Alaeddin/convbert-base-turkish-ner-cased')
    model = AutoModelForTokenClassification.from_pretrained('Alaeddin/convbert-base-turkish-ner-cased')
    ner = TokenClassificationPipeline(model=model, tokenizer=tokenizer, grouped_entities=True)

    queue = redis.StrictRedis(host="redis")
    logging.warning("Connected to Redis")

    run_ner()
