FROM ailabscomtr/transformers

# Cache models
RUN python3 -c "from transformers import AutoTokenizer, TFAutoModel; AutoTokenizer.from_pretrained('mys/bert-base-turkish-cased-nli-mean'); TFAutoModel.from_pretrained('mys/bert-base-turkish-cased-nli-mean')"

# copy scripts
COPY main.py /app/

# run script
CMD ["python3", "/app/main.py"]