import json
import logging
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import redis

def generate_summary(main_news):
  source_encoding=tokenizer(
    main_news,
    max_length=784,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    add_special_tokens=True,
    return_tensors="pt")

  generated_ids=model.generate(
      input_ids=source_encoding["input_ids"],
      attention_mask=source_encoding["attention_mask"],
      num_beams=2,
      max_length=96,
      repetition_penalty=2.5,
      length_penalty=2.5,
      early_stopping=True,
      use_cache=True
  )

  preds=[tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
         for gen_id in generated_ids]

  return "".join(preds)


def run_ats():
    while True:
        with queue.pipeline() as pipe:
            pipe.lrange("ats", 0, 7)
            pipe.ltrim("ats", 8, -1)
            requests, _ = pipe.execute()

        for r in requests:
            start = time.time()
            r = json.loads(r)
            results = {"summary": generate_summary(r["text"])}

            queue.set(r["id"], json.dumps(results))
            logging.warning(f"{r['id']} processed in {time.time() - start}")

        time.sleep(0.1)



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("ozcangundes/mt5-small-turkish-summarization")
    model = AutoModelForSeq2SeqLM.from_pretrained("ozcangundes/mt5-small-turkish-summarization")

    queue = redis.StrictRedis(host="redis")
    logging.warning("Connected to Redis")

    run_ats()