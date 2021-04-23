FROM ailabscomtr/transformers

RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('ozcangundes/mt5-small-turkish-summarization'); AutoModelForSeq2SeqLM.from_pretrained('ozcangundes/mt5-small-turkish-summarization')"

COPY ./requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY main.py /app/main.py

CMD ["python", "/app/main.py"]