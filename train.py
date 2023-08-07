#!/usr/bin/env python
# coding: utf-8


from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TextClassificationPipeline,
    EarlyStoppingCallback,
)
from datasets import load_dataset, load_from_disk, load_metric
import numpy as np

from utils import strip_text, compute_metrics


def train_with_random_search(tokenized_dataset, model, tokenizer, model_name):
    import wandb

    sweep_config = {"method": "random"}

    parameters_dict = {
        "epochs": {"value": 5},
        "batch_size": {"values": [8, 16]},
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3,
        },
        "weight_decay": {"values": [0.001, 0.01, 0.1, 0.5]},
    }

    sweep_config["parameters"] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="seq-classification-sweeps")

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, return_dict=True
        )

    with wandb.init(config=config):
        config = wandb.config

        training_args = TrainingArguments(
            output_dir="seq-classification-sweeps",
            report_to="wandb",
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=8,
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            # remove_unused_columns=False,
            # fp16=True
        )

        trainer = Trainer(
            # model,
            model_init=model_init,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            compute_metrics=compute_metrics,
        )

        # start training loop
        trainer.train()

    wandb.agent(sweep_id, train, count=20)


def train_with_hyperparameters(tokenized_dataset, model, tokenizer, model_name):
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.1,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        # data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("model/" + model_name)


def train(model_name: str, random_search: bool):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset("csv", data_files={"negative_data.csv", "positive_data.csv"})

    dataset.shuffle(seeds=20)

    dataset = dataset.remove_columns("Rate")
    dataset = dataset.rename_column("Review", "text")
    dataset = dataset.rename_column("Label", "label")

    dataset = dataset["train"]

    dataset = dataset.map(strip_text)
    dataset = dataset.train_test_split(test_size=0.1)

    tokenize_function = lambda examples: tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=256
    )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns="text"
    )

    if random_search:
        train_with_random_search(tokenized_dataset, model, tokenizer, model_name)
    else:
        train_with_hyperparameters(tokenized_dataset, model, tokenizer, model_name)
