from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TextClassificationPipeline,
)
from datasets import load_dataset, load_from_disk, load_metric
import numpy as np


def infer(model_name: str, text: str):
    model_path = "model/" + model_name

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    pipe = TextClassificationPipeline(
        model=model, tokenizer=tokenizer, return_all_scores=True, device=0
    )
    print(pipe(text))
