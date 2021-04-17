import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("ozcangundes/mt5-multitask-qa-qg-turkish") 
model = AutoModelForSeq2SeqLM.from_pretrained("ozcangundes/mt5-multitask-qa-qg-turkish")

from pipelines import pipeline #pipelines.py script in the cloned repo
multimodel = pipeline("multitask-qa-qg",tokenizer=tokenizer,model=model)

# sample text
text = """Özcan Gündeş, 1993 yılı Tarsus doğumludur. Orta Doğu Teknik Üniversitesi 
Endüstri Mühendisliği bölümünde 2011 2016 yılları arasında lisans eğitimi görmüştür.
Yüksek lisansını ise 2020 Aralık ayında, 4.00 genel not ortalaması ile 
Boğaziçi Üniversitesi, Yönetim Bilişim Sistemleri bölümünde tamamlamıştır.
Futbolla yakından ilgilenmekle birlikte, Galatasaray kulübü taraftarıdır."""

logging.warn(multimodel(text))
