FROM ailabscomtr/transformers

RUN python -c "from transformers import AutoTokenizer, AutoModelForTokenClassification; AutoTokenizer.from_pretrained('Alaeddin/convbert-base-turkish-ner-cased'); AutoModelForTokenClassification.from_pretrained('Alaeddin/convbert-base-turkish-ner-cased')"

COPY ./requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY main.py /app/main.py

CMD ["python", "/app/main.py"]