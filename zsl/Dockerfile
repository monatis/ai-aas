FROM ailabscomtr/transformers

# install extra dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# copy models
COPY bert-base-turkish-cased-nli-mean-tokens /bert-base-turkish-cased-nli-mean-tokens

# copy scripts
COPY main.py /app/

# run script
CMD ["python", "/app/main.py"]