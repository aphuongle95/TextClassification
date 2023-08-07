import numpy as np
from datasets import load_dataset, load_from_disk, load_metric

def strip_text(example):
    example["text"] = example["text"].strip()
    return example

metric = load_metric("accuracy")
def compute_metrics(eval_pred) :
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)