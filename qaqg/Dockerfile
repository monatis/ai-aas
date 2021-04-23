FROM ailabscomtr/transformers

RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('ozcangundes/mt5-multitask-qa-qg-turkish'); AutoModelForSeq2SeqLM.from_pretrained('ozcangundes/mt5-multitask-qa-qg-turkish')"

COPY ./requirements.txt /app/requirements.txt

RUN apt -y update && \
    apt -y --no-install-recommends install git && \
    git clone https://github.com/ozcangundes/multitask-question-generation.git /app/multitask-question-generation && \
    apt autoremove -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir -r /app/requirements.txt

WORKDIR /app/multitask-question-generation

COPY main.py /app/multitask-question-generation/main.py

CMD ["python", "/app/multitask-question-generation/main.py"]