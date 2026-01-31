# this file uses Finbert model which is used for sentiment analysis os financial data
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
class FinBERT_Sentiment():
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model.eval()

    def score(self, text: str) -> float: # Score written by AI
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        with torch.no_grad(): # written byb using documentation
            outputs = self.model(**inputs) # input then output
            probab = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
            sentiment_score = probab[2] - probab[0]
            return float(sentiment_score)
