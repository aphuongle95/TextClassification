#!/usr/bin/env python
# coding: utf-8


from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, TextClassificationPipeline
from datasets import load_dataset, load_from_disk, load_metric
import numpy as np

model_name = "FPTAI/vibert-base-cased"


model = AutoModelForSequenceClassification.from_pretrained(model_name , num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)


dataset = load_dataset('csv', data_files={'negative_data.csv', 'positive_data.csv'})


dataset.shuffle(seeds=20)

dataset = dataset.remove_columns("Rate")
dataset = dataset.rename_column("Review", "text")
dataset = dataset.rename_column("Label", "label")


dataset = dataset['train']
def strip_text(example):
    example["text"] = example["text"].strip()
    return example

dataset = dataset.map(strip_text)
dataset = dataset.train_test_split(test_size=0.1)


print(dataset)



def tokenize_function(examples):
  return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns='text')


tokenized_datasets

metric = load_metric("accuracy")
def compute_metrics(eval_pred) :
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# training_args = TrainingArguments(
#     "test", evaluation_strategy="steps", eval_steps=500, disable_tqdm=True)
# trainer = Trainer(
#     args=training_args,
#     tokenizer=tokenizer,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     model_init=model_init,
#     compute_metrics=compute_metrics,
# )

# # Default objective is the sum of all metrics
# # when metrics are provided, so we have to maximize it.
# trainer.hyperparameter_search(
#     direction="maximize",
#     backend="ray",
#     n_trials=10 # number of trials
# )




import wandb

sweep_config = {
    'method': 'random'
}



parameters_dict = {
    'epochs': {
        'value': 5
        },
    'batch_size': {
        'values': [8, 16]
        },
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 1e-5,
        'max': 1e-3
    },
    'weight_decay': {
        'values': [0.001, 0.01, 0.1, 0.5]
    },
}


sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project='seq-classification-sweeps')

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, return_dict=True)

def train(config=None):
  with wandb.init(config=config):
    # set sweep configuration
    config = wandb.config


    # set training arguments
    training_args = TrainingArguments(
        output_dir='seq-classification-sweeps',
        report_to='wandb',
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=8,
    logging_strategy='epoch',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,

        # remove_unused_columns=False,
        # fp16=True
    )


    # define training loop
    trainer = Trainer(
        # model,
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics
    )


    # start training loop
    trainer.train()

wandb.agent(sweep_id, train, count=20)


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_strategy='epoch',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    # data_collator=data_collator,
)

trainer.train()

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=0)
pipe("Món ăn ở đây quá tệ")


# **Kết quả mong đợi** (Xấp xỉ)

# ```python
# [[{'label': 'LABEL_0', 'score': 0.6195073127746582},
#   {'label': 'LABEL_1', 'score': 0.36591729521751404},
#   {'label': 'LABEL_2', 'score': 0.005223034415394068},
#   {'label': 'LABEL_3', 'score': 0.004311065189540386},
#   {'label': 'LABEL_4', 'score': 0.00504127936437726}]]
# ```

# # TODO 9: Lập bảng so sánh kết quả training của hai mô hình này
# 
# |Mô hình   | Độ chính xác   |
# |---|---|
# |vinai/xphonebert-base   |   |
# | FPTAI/vibert-base-cased |   |
# |   |   |  
# 

# Chúc mừng bạn đã lập trình xong bài toán **phân loại mô hình cảm xúc.**
