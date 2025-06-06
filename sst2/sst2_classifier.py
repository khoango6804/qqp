import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from pathlib import Path
import evaluate
import numpy as np
import pandas as pd

model_name = Path(r"K:\checkpoint-116000")
num_labels = 2
id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}
model = AutoModelForSequenceClassification.from_pretrained(
model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    trust_remote_code=True,
    local_files_only=True )
tokenizer = AutoTokenizer.from_pretrained(model_name
                                          ,local_files_only=True, trust_remote_code=True)


train_path = r"K:\checkpoint-116000\sst2\train-00000-of-00001.parquet"
val_path = r"K:\checkpoint-116000\sst2\train-00000-of-00001.parquet"

train_df = pd.read_parquet(train_path)
val_df = pd.read_parquet(val_path)

train_df, test_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=42)
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'validation': Dataset.from_pandas(val_df),
    'test': Dataset.from_pandas(test_df)
})

def format_prompt(premise, hypothesis):
    return f"Premise: {premise}\nHypothesis: {hypothesis}"

def preprocess_function(examples):
    return tokenizer(
        examples['sentence'],
        truncation=True,
        padding=True,
        max_length=256
    )

tokenized_datasets = dataset.map(preprocess_function, batched=True)
print("Tokenized dataset:", tokenized_datasets)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer

training_args = TrainingArguments(
    output_dir="SST2_classifier",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    lr_scheduler_type="cosine",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    bf16=True,  
    load_best_model_at_end=True,
    push_to_hub=False,  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer.train()